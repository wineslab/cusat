/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Async-signal-safe crash handler with backtrace.
 * Catches SIGSEGV, SIGABRT, SIGBUS, SIGFPE — prints stack trace to stderr,
 * then re-raises with default handler for core dump.
 */

#ifndef INCLUDED_CUSAT_DSP_CRASH_HANDLER_H
#define INCLUDED_CUSAT_DSP_CRASH_HANDLER_H

#ifdef __linux__
#include <execinfo.h>
#include <csignal>
#include <cstring>
#include <unistd.h>

namespace cusat_dsp {

/* Best-effort stderr write from a signal handler: nothing to do on failure. */
inline void crash_write(const char* buf, size_t len)
{
    ssize_t r = write(STDERR_FILENO, buf, len);
    (void)r;
}

inline void crash_signal_handler(int sig)
{
    const char* msg = "\n=== CRASH: signal ";
    crash_write(msg, strlen(msg));

    // Write signal number as ASCII (async-signal-safe)
    char num[16];
    int len = 0;
    int s = sig;
    if (s == 0) {
        num[len++] = '0';
    } else {
        char tmp[16];
        int tlen = 0;
        while (s > 0) {
            tmp[tlen++] = '0' + (s % 10);
            s /= 10;
        }
        for (int i = tlen - 1; i >= 0; i--)
            num[len++] = tmp[i];
    }
    crash_write(num, len);

    const char* msg2 = " ===\nBacktrace:\n";
    crash_write(msg2, strlen(msg2));

    void* frames[64];
    int n = backtrace(frames, 64);
    backtrace_symbols_fd(frames, n, STDERR_FILENO);

    const char* msg3 = "=== End backtrace ===\n";
    crash_write(msg3, strlen(msg3));

    // Re-raise with default handler to produce core dump
    signal(sig, SIG_DFL);
    raise(sig);
}

inline void install_crash_handler()
{
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = crash_signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND;

    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
    sigaction(SIGFPE, &sa, nullptr);
}

} // namespace cusat_dsp

#else
// Non-Linux: no-op
namespace cusat_dsp {
inline void install_crash_handler() {}
} // namespace cusat_dsp
#endif

#endif /* INCLUDED_CUSAT_DSP_CRASH_HANDLER_H */
