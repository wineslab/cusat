/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * GPU bulk PLHEADER search used during acquisition.
 */

#ifndef INCLUDED_CUSAT_DSP_CUDA_PLHEADER_SEARCH_KERNEL_CUH
#define INCLUDED_CUSAT_DSP_CUDA_PLHEADER_SEARCH_KERNEL_CUH

#include "plrx/pl_header_candidates.h"
#include <cuda_runtime.h>
#include <cusat_dsp/compat.h>
#include <vector>

namespace cusat_dsp {
namespace cuda {

class plheader_search_cuda
{
public:
    /* max_symbols: largest buffer ever searched (overlap + batch). */
    explicit plheader_search_cuda(int max_symbols);
    ~plheader_search_cuda();

    plheader_search_cuda(const plheader_search_cuda&) = delete;
    plheader_search_cuda& operator=(const plheader_search_cuda&) = delete;

    /* Enqueue a search on `stream` (no synchronization, no allocation).
     * sym is device-readable interleaved (re, im) floats (device memory or
     * pinned host memory), n_symbols long. index_offset is subtracted from
     * every reported index so that indices are relative to the caller's
     * batch start (pass the overlap length). */
    void launch(const float* sym, int n_symbols, int index_offset,
                cudaStream_t stream);

    /* Result of the last launch(); valid once `stream` has been
     * synchronized. */
    pl_header_candidates result() const;

    /* Blocking convenience for host-resident symbols (uses an internal
     * stream). */
    pl_header_candidates search(const complexf* sym_host, int n_symbols,
                                int index_offset);

    /* Test/study hook: buffer positions (not offset-corrected) that survived
     * the per-block top-K stage of the last launch (blocking copy). */
    std::vector<int> stage1_positions() const;
    /* ... and those selected by the mid statistic (two distinct positions per group). */
    std::vector<int> stage2_positions() const;

private:
    int d_max_symbols = 0;
    int d_max_blocks = 0;
    float2* d_sym_dev = nullptr;          // staging for search() (pageable input)
    int* d_cand_pos = nullptr;            // [max_blocks * K] stage-1 survivors
    float* d_block_pow = nullptr;         // [max_blocks]
    float* d_vscore = nullptr;            // score per retained hypothesis
    int* d_vpos = nullptr;                // position per retained hypothesis
    float2* d_rot_mid = nullptr;          // rotation tables of the verify kernel
    float2* d_rot_fine = nullptr;
    float2* d_rot_pf = nullptr;           // tile counter-rotation table
    float2* d_cand_win = nullptr;         // [max_blocks * K] survivor windows (derotated)
    int* d_mid_pos = nullptr;             // two mid-selected positions per group
    unsigned* d_done = nullptr;           // group counter (last group selects)
    unsigned* d_pair_done = nullptr;      // per-group producer counters
    int d_last_cands = 0;
    int d_last_groups = 0;
    bool d_pdl = false;                   // programmatic dependent launch available
    pl_header_candidates* d_result_host = nullptr; // pinned, written by the GPU
    cudaStream_t d_stream = nullptr;      // for search()
};

} // namespace cuda
} // namespace cusat_dsp

#endif /* INCLUDED_CUSAT_DSP_CUDA_PLHEADER_SEARCH_KERNEL_CUH */
