use core::ffi::{c_char, c_int};
use core::sync::atomic::{AtomicBool, Ordering};
use std::ffi::{CStr, CString};
use std::sync::Once;

use bun_core::Output;

static INITIALIZED: Once = Once::new();
static STARTED: AtomicBool = AtomicBool::new(false);

#[unsafe(no_mangle)]
extern "C" fn BunEmbedInitialize() -> c_int {
    if unsafe { libc::pthread_main_np() } == 0 {
        return -1;
    }
    INITIALIZED.call_once(|| {
        // Like `bun`, so a write to a closed pipe fails with EPIPE instead of killing the
        // app. A disposition the app chose is kept.
        for signal in [libc::SIGPIPE, libc::SIGXFSZ] {
            let mut current: libc::sigaction = unsafe { core::mem::zeroed() };
            if unsafe { libc::sigaction(signal, core::ptr::null(), &mut current) } == 0
                && current.sa_sigaction == libc::SIG_DFL
            {
                unsafe { libc::signal(signal, libc::SIG_IGN) };
            }
        }
        crate::bin_entry::use_mimalloc_in_dependencies();
        Output::stdio::init();
    });
    0
}

/// # Safety
/// `argv` points to `argc` NUL-terminated strings, read before this returns.
#[unsafe(no_mangle)]
unsafe extern "C" fn BunEmbedStart(argc: c_int, argv: *const *const c_char) -> c_int {
    if !INITIALIZED.is_completed() || argc < 1 || argv.is_null() {
        return -1;
    }
    let mut args = Vec::with_capacity(argc as usize);
    for i in 0..argc as usize {
        let arg = unsafe { *argv.add(i) };
        if arg.is_null() {
            return -1;
        }
        args.push(unsafe { CStr::from_ptr(arg) }.to_owned());
    }
    if STARTED.swap(true, Ordering::AcqRel) {
        return -2;
    }
    match std::thread::Builder::new()
        .name("Bun".into())
        .stack_size(18 * 1024 * 1024)
        .spawn(move || run(args))
    {
        Ok(_) => 0,
        Err(_) => {
            STARTED.store(false, Ordering::Release);
            -3
        }
    }
}

fn run(args: Vec<CString>) {
    // Bun reads argv until the process exits.
    let argv: &'static [*const c_char] =
        Vec::leak(args.into_iter().map(|arg| arg.into_raw().cast_const()).collect());
    unsafe { bun_core::init_argv(argv.len() as c_int, argv.as_ptr()) };
    Output::stdio::init();
    bun_core::StackCheck::configure_thread();
    crate::cli::Cli::start();
    bun_core::Global::exit(0)
}
