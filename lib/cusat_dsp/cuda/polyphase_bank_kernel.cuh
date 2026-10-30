/*
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * GPU polyphase filterbank energy search for timing estimation.
 *
 * Evaluates K timing arms spread over one symbol period (offsets
 * a * stride / K samples, a = 0..K-1) and picks the arm with maximum
 * matched-filter output energy. Works at any fractional stride
 * (samples per symbol): per evaluated symbol the fractional position
 * selects the polyphase subfilter, exactly like the interpolation kernel.
 *
 * The kernel writes state_um[0..5] in the same format as the skip path, so
 * batch_interp_kernel runs unchanged downstream.
 */

#ifndef INCLUDED_CUSAT_DSP_CUDA_POLYPHASE_BANK_KERNEL_CUH
#define INCLUDED_CUSAT_DSP_CUDA_POLYPHASE_BANK_KERNEL_CUH

#include <cuda_runtime.h>

namespace cusat_dsp {
namespace cuda {

static constexpr int ENERGY_K_DEFAULT = 32;  // Default polyphase arms (T/32 raw resolution)
static constexpr int ENERGY_K_MAX = 64;      // Max supported polyphase arms
static constexpr int ENERGY_N_BLOCKS = 512;  // Grid size for energy kernel
                                             // (4 blocks/SM on GH200: enough
                                             // warps to hide FIR FMA latency)

/**
 * @brief Launch polyphase energy search kernel.
 *
 * Evaluates K timing arms across the input batch, finds the arm with
 * maximum matched-filter energy, and writes timing parameters to
 * state_um in the format consumed by batch_interp_kernel:
 *   [0]=tau (symbols, [0,1)), [1]=strength, [2]=n_syms,
 *   [3]=base (fractional sample offset of symbol 0 relative to
 *   ss_history, anchored mod-stride to prev_base), [4]=frac diag,
 *   [5]=drift_accum.
 *
 * @param samples_dev     Input IQ samples (device memory, interleaved re/im).
 * @param subfilters_dev  Full subfilter tap array [n_phases * subfilt_len].
 * @param state_um        Unified memory state output [8 doubles].
 * @param partial_energy  Device buffer for partial energies [N_BLOCKS * K].
 * @param done_cnt        Device atomic counter (reset to 0 before each call).
 * @param N               Total input samples.
 * @param stride          Samples per symbol (fractional).
 * @param n_phases       Number of subfilters (128).
 * @param subfilt_len     Taps per subfilter.
 * @param ss_history      Symbol sync history in samples.
 * @param max_syms_limit  Max symbols to produce.
 * @param max_syms_buf    Max symbols the output buffer can hold.
 * @param prev_base       NCO-predicted base offset (for mod-stride anchoring
 *                        and drift tracking).
 * @param prev_drift_accum Previous accumulated drift.
 * @param prev_init       Whether previous state was initialized.
 * @param energy_k        Number of timing arms to evaluate (8..64).
 * @param energy_search_cap  Max symbols for energy search (0 = no cap).
 * @param stream          CUDA stream.
 */
void launch_polyphase_energy_search(
    const float* samples_dev,
    const float* subfilters_dev,
    double* state_um,
    float* partial_energy,
    unsigned int* done_cnt,
    int N, double stride, int n_phases, int subfilt_len,
    int ss_history, int max_syms_limit, int max_syms_buf,
    double prev_base, double prev_drift_accum, int prev_init,
    int track_mode, double base_trim,
    int energy_k,
    int energy_search_cap,
    cudaStream_t stream);

} // namespace cuda
} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_CUDA_POLYPHASE_BANK_KERNEL_CUH */
