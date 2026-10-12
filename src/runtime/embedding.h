#pragma once
#ifdef __cplusplus
extern "C" {
#endif
// Initialize once on the application's main thread. Like `bun`, it ignores SIGPIPE and
// SIGXFSZ (writes then fail with EPIPE/EFBIG) unless the application set their dispositions.
// 0: initialized (also when called again); -1: not called on the main thread.
int BunEmbedInitialize(void);
// Runs `bun` with these arguments on its own 18 MiB thread and returns once the thread
// starts. argv[0] is the program name, e.g. {"bun", "run", "/absolute/path/app.js"}; the
// arguments are copied. Bun then runs for the rest of the application's life and writes to
// descriptors 1 and 2. Where `bun` would exit its process (the script ends, process.exit(),
// an uncaught error), it exits the application. To restart code, run it in a Worker from a
// script that keeps running.
// Crash signal handlers set before this are called for signals Bun does not handle. Darwin does
// not report SA_RESETHAND to Bun, so such a handler runs on every delivery; to end the process
// it restores the previous disposition and re-raises, as crash reporters do.
// 0: started; -1: invalid arguments or not initialized; -2: already started;
// -3: the thread could not be created.
int BunEmbedStart(int argc, const char *const *argv);
#ifdef __cplusplus
}
#endif
