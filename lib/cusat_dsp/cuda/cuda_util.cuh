/*
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef INCLUDED_CUSAT_DSP_CUDA_UTIL_CUH
#define INCLUDED_CUSAT_DSP_CUDA_UTIL_CUH

#include <cuda_runtime.h>

#define CUDA_CHECK(call) ::cusat_dsp::cuda::check_cuda_error((call), __FILE__, __LINE__)
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaPeekAtLastError())

namespace cusat_dsp {
namespace cuda {

void check_cuda_error(cudaError_t err, const char* file, int line);

/** Enable graceful CUDA-fault handling: on a CUDA error, exit cleanly with
 *  CUDA_FAULT_EXIT_CODE (42) instead of abort(), so a supervisor can restart
 *  and re-acquire. Enable once the RX is running; leave off during setup. */
void set_cuda_fault_graceful(bool on);
extern int CUDA_FAULT_EXIT_CODE;

/** Initialize CUDA device and print info. Returns false if no GPU available. */
bool init_cuda();

} // namespace cuda
} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_CUDA_UTIL_CUH */
