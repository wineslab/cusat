/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA utilities: device query, stream management, timing helpers.
 */

#include "cuda_util.cuh"
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <cuda_runtime.h>

namespace cusat_dsp {
namespace cuda {

// Graceful-fault mode: once a long-running consumer (the deployed RX) is up, a
// CUDA error (esp. an illegal access / 700) poisons the whole context and is
// unrecoverable in-process — so the only safe response is to exit cleanly and
// let a supervisor restart us (which re-acquires the link = "reattach"). During
// setup, and in offline tools, we keep abort() (fail-fast surfaces real bugs).
// Distinct exit code so a wrapper can tell a GPU fault from a normal exit.
static std::atomic<bool> g_cuda_graceful{false};
int CUDA_FAULT_EXIT_CODE = 42;

void set_cuda_fault_graceful(bool on) { g_cuda_graceful.store(on); }

void check_cuda_error(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA FATAL at %s:%d: %s (%d)\n", file, line,
                cudaGetErrorString(err), (int)err);
        if (g_cuda_graceful.load()) {
            // Context is poisoned; do NOT touch CUDA again (cleanup would
            // recurse). _Exit skips destructors -> immediate, restartable.
            fprintf(stderr, "cusat: GPU fault during operation — exiting "
                    "cleanly (code %d) for supervised restart/reattach\n",
                    CUDA_FAULT_EXIT_CODE);
            fflush(stderr);
            std::_Exit(CUDA_FAULT_EXIT_CODE);
        }
        abort();
    }
}

bool init_cuda()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "cuda: no CUDA devices found\n");
        return false;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "cuda: %s (SM %d.%d, %d SMs, %.0f MB)\n",
            prop.name, prop.major, prop.minor,
            prop.multiProcessorCount,
            prop.totalGlobalMem / (1024.0 * 1024.0));

    return true;
}

} // namespace cuda
} // namespace cusat_dsp
