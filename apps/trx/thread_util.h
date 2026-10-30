/*
 *
 * This file is part of cuSAT.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Thread pinning and signal handling for dvbs2-uhd-trx.
 */

#pragma once

#include <pthread.h>
#include <sched.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <vector>

/* ---------- Thread pinning ---------- */
inline std::vector<int> g_available_cores; // filled at startup from cpuset

inline void detect_available_cores()
{
#ifdef __linux__
    cpu_set_t cpuset;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) == 0) {
        for (int i = 0; i < CPU_SETSIZE; i++) {
            if (CPU_ISSET(i, &cpuset))
                g_available_cores.push_back(i);
        }
    }
    fprintf(stderr, "dvbs2-uhd-trx: available cores (%zu):", g_available_cores.size());
    for (int c : g_available_cores) fprintf(stderr, " %d", c);
    fprintf(stderr, "\n");
#endif
}

// Pin current thread to the Nth available core (0-indexed) and set SCHED_FIFO.
// SCHED_FIFO prevents kernel softirqs (SCHED_OTHER) from preempting app threads.
// Hardware IRQs can still preempt but are very brief (~us).
inline void pin_thread(const char* name, int index, bool realtime = false)
{
    if (index < 0) return;
#ifdef __linux__
    if (index >= (int)g_available_cores.size()) {
        fprintf(stderr, "dvbs2-uhd-trx: cannot pin %s to slot %d (only %zu cores available)\n",
                name, index, g_available_cores.size());
        return;
    }
    int core = g_available_cores[index];
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc == 0)
        fprintf(stderr, "dvbs2-uhd-trx: pinned %s to core %d (slot %d)", name, core, index);
    else
        fprintf(stderr, "dvbs2-uhd-trx: failed to pin %s to core %d (rc=%d)", name, core, rc);

    if (!realtime) {
        fprintf(stderr, "\n");
        return;
    }

    // Optional SCHED_FIFO priority (requires root or CAP_SYS_NICE). Keep this
    // independent from affinity: a spinning CUDA worker at FIFO priority can
    // starve ksoftirqd and UHD flow-control processing on its pinned CPU.
    struct sched_param sp = {};
    sp.sched_priority = 80; // high but below kernel migration threads (99)
    int rt_rc = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
    if (rt_rc == 0)
        fprintf(stderr, " SCHED_FIFO(%d)\n", sp.sched_priority);
    else
        fprintf(stderr, " (SCHED_FIFO failed: %s)\n", strerror(rt_rc));
#else
    fprintf(stderr, "dvbs2-uhd-trx: thread pinning not supported on this platform (%s slot %d)\n", name, index);
#endif
}

/* ---------- Signal handling ---------- */
inline std::atomic<bool> g_stop{false};
inline void signal_handler(int) { g_stop.store(true); }
