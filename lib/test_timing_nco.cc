/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone check for the fractional-stride timing NCO (needs a GPU).
 *
 * Interpolates a known complex sinusoid x[n] = exp(j*theta*n) through the
 * GPU timing recovery at a fractional stride (e.g. 4.096 samples/symbol)
 * across many process() calls. Because the input phase is an exact linear
 * function of the (fractional) sample position, the output symbol phases
 * must advance by ~theta*stride per symbol — including across batch
 * boundaries, where the integer n_consumed / fractional NCO-carry split
 * happens. Any carry bug shows up as a phase step at a batch boundary or
 * as cumulative phase drift over many batches.
 *
 * Two tiers of checks:
 *
 *  1. EXACT check (tight tolerance): the kernel's output for symbol k is a
 *     deterministic FIR (filter-bank convention of interpolator.h): with
 *     p = base + k*stride, q = round(frac(p)*n_phases), the window ends at
 *     n = ss_history + floor(p) + (q > 0) and uses phase
 *     ph = (n_phases - q) mod n_phases, taps newest-first. For the tone,
 *     the true output phase is exactly
 *         theta*(abs0 + n) + arg(G_ph),
 *         G_ph = sum_j taps[ph][j] * exp(-j*theta*j).
 *     This is checked to ~1e-4 rad, so any NCO/indexing/carry error of even
 *     ONE subfilter step (theta/128 = 3.9e-4 rad) fails it.
 *
 *  2. UNIFORMITY checks (loose tolerance): per-symbol phase increments vs.
 *     the ideal theta*stride. These can NOT be tight: the branches are
 *     shifted copies of a TRUNCATED prototype, so their phase response at
 *     theta deviates slightly from the ideal i/n_phases delay law. The
 *     tolerance is therefore derived from the measured peak-to-peak branch
 *     phase ripple of the actual taps, plus quantization and margin.
 *
 * Run manually (not part of ctest):
 *   ./build-local/lib/test_timing_nco
 */

#include "cuda/timing_recovery_kernel.cuh"
#include "interpolator.h"
#include <cusat_dsp/compat.h>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>

using namespace cusat_dsp;

static double wrap_pi(double x)
{
    // std::remainder wraps to [-pi, pi] in O(1); the naive subtract-loop
    // is quadratic here (the cumulative check feeds it phases that grow
    // by ~0.2 rad/symbol over millions of symbols).
    return std::remainder(x, 2.0 * M_PI);
}

static int run_stride(double stride, int n_batches)
{
    const float rolloff = 0.2f;
    const int rrc_delay = 5;
    const int n_phases = 128;
    const double theta = 0.05; // rad per input sample (well within passband)

    filter_bank_design design;
    design.samples_per_symbol = stride;
    design.rolloff = rolloff;
    design.span_symbols = rrc_delay;
    design.n_phases = n_phases;
    polyphase_filter_bank interp(design);
    const int nf = (int)interp.n_phases();
    const int fl = (int)interp.taps_per_phase();
    std::vector<float> flat = interp.flat_taps();

    // Per-branch true phase response of the bank at the tone frequency:
    // branch_arg[i] = arg(G_i), G_i = sum_j taps[i][j] e^{-j theta j}.
    // Also measure the bank's peak-to-peak deviation from the ideal linear
    // i/nf branch-delay law — this sets the uniformity-check tolerance.
    std::vector<double> branch_arg(nf);
    double dev_min = 1e9, dev_max = -1e9;
    for (int i = 0; i < nf; i++) {
        double re = 0.0, im = 0.0;
        for (int j = 0; j < fl; j++) {
            double a = -theta * (double)j;
            re += (double)flat[i * fl + j] * std::cos(a);
            im += (double)flat[i * fl + j] * std::sin(a);
        }
        branch_arg[i] = std::atan2(im, re);
        // ideal: arg(G_i) = -theta (D + i / nf)
        double dev = branch_arg[i] + theta * (double)i / (double)nf;
        if (dev < dev_min) dev_min = dev;
        if (dev > dev_max) dev_max = dev;
    }
    const double bank_ripple = dev_max - dev_min; // rad, peak-to-peak
    printf("stride %.4f: bank branch phase ripple (p-p) = %.6f rad "
           "(%.4f samples at theta=%.2f)\n",
           stride, bank_ripple, bank_ripple / theta, theta);

    const int max_input = 131072;
    cuda::timing_recovery_cuda trc(nf, fl, flat.data(), stride,
                                   interp.group_delay_samples(), max_input);
    // Estimate once on the first call, then dead-reckon: the NCO carry is
    // exactly what this test exercises. (A pure tone has no timing
    // cyclostationarity, so re-estimating would re-anchor arbitrarily.)
    trc.set_estimation_period(1000000);

    const unsigned ss_history =
        interp.history() + (unsigned)std::ceil(stride / 2.0) + 2;
    const int chunk = 32768;
    const int buf_cap = chunk + (int)ss_history + 64;
    std::vector<complexf> buf(buf_cap);
    const int max_syms = (int)(buf_cap / stride) + 2;
    std::vector<complexf> syms(max_syms);

    cuda::timing_recovery_cuda::state_t st{};
    st.tau = 0.0;
    st.nco_base = 0.0;
    st.drift_accum = 0.0;
    st.init = false;

    uint64_t abs0 = 0;        // absolute index of buf[0]
    int n_buffered = 0;
    uint64_t total_syms = 0;
    uint64_t total_consumed = 0;

    bool have_ref = false;
    double ref_phase = 0.0;     // measured phase of reference symbol
    uint64_t ref_sym = 0;       // its global symbol index
    double last_phase = 0.0;    // phase of last symbol of previous batch
    bool have_last = false;

    double max_intra_err = 0.0, max_boundary_err = 0.0, max_cum_err = 0.0;
    double max_exact_err = 0.0;
    double min_carry = 1e9, max_carry = -1e9;
    const double dphi = theta * stride; // expected per-symbol phase step

    int failures = 0;

    for (int b = 0; b < n_batches; b++) {
        // Refill the buffer with the continuing sinusoid
        while (n_buffered < buf_cap) {
            uint64_t n_abs = abs0 + (uint64_t)n_buffered;
            double ph = theta * (double)n_abs;
            buf[n_buffered] = complexf(0.5 * std::cos(ph), 0.5 * std::sin(ph));
            n_buffered++;
        }

        int n_consumed = 0;
        auto t0 = std::chrono::steady_clock::now();
        int n = trc.process(buf.data(), n_buffered, syms.data(), max_syms,
                            st, n_consumed, ss_history);
        auto t1 = std::chrono::steady_clock::now();
        if (b < 10 || (b % 200) == 0) {
            double wall_us =
                std::chrono::duration<double, std::micro>(t1 - t0).count();
            fprintf(stderr,
                    "PROF batch %d: wall=%.0fus gpu_est=%uus gpu_interp=%uus\n",
                    b, wall_us, trc.last_timing_us(), trc.last_interp_us());
        }
        if (n <= 0) {
            fprintf(stderr, "stride %.4f batch %d: no symbols produced\n",
                    stride, b);
            return 1;
        }

        // Reconstruct the fractional base the kernel used for this batch
        // from the post-batch NCO carry (exact in double up to ~1e-11):
        // st.nco_base = base + n*stride - n_consumed.
        double base_b =
            st.nco_base + (double)n_consumed - (double)n * stride;

        // --- Verify phases ---
        for (int k = 0; k < n; k++) {
            double ph = std::atan2(syms[k].imag(), syms[k].real());

            // EXACT reference: replicate the kernel's basepoint/subfilter
            // selection and use the branch's true phase response.
            {
                double p = base_b + (double)k * stride;
                int ip = (int)std::floor(p);
                double fr = p - (double)ip;
                int q = (int)(fr * (double)nf + 0.5);
                int newest = ip + (q > 0 ? 1 : 0);
                int ph = (q > 0 && q < nf) ? nf - q : 0;
                double n_abs = (double)abs0 +
                               (double)((int)ss_history + newest);
                double expect = theta * n_abs + branch_arg[ph];
                double e = std::fabs(wrap_pi(ph - expect));
                if (e > max_exact_err) max_exact_err = e;
            }
            if (k > 0) {
                double prev = std::atan2(syms[k - 1].imag(), syms[k - 1].real());
                double e = std::fabs(wrap_pi(ph - prev - dphi));
                if (e > max_intra_err) max_intra_err = e;
            } else if (have_last) {
                // batch boundary: symbol spacing must still be exactly stride
                double e = std::fabs(wrap_pi(ph - last_phase - dphi));
                if (e > max_boundary_err) max_boundary_err = e;
            }
            // cumulative drift vs. the global symbol counter
            uint64_t g = total_syms + (uint64_t)k;
            if (!have_ref) {
                have_ref = true;
                ref_phase = ph;
                ref_sym = g;
            } else {
                double expect = ref_phase + dphi * (double)(g - ref_sym);
                double e = std::fabs(wrap_pi(ph - expect));
                if (e > max_cum_err) max_cum_err = e;
            }
        }
        last_phase = std::atan2(syms[n - 1].imag(), syms[n - 1].real());
        have_last = true;

        total_syms += (uint64_t)n;
        total_consumed += (uint64_t)n_consumed;
        if (st.nco_base < min_carry) min_carry = st.nco_base;
        if (st.nco_base > max_carry) max_carry = st.nco_base;

        if (b < 5 || (b % 200) == 0) {
            printf("stride %.4f batch %5d: n_syms=%5d n_consumed=%5d "
                   "carry=%.6f tau=%.4f\n",
                   stride, b, n, n_consumed, st.nco_base, st.tau);
        }

        // Shift the unconsumed tail to the front (same as the apps do)
        n_buffered -= n_consumed;
        if (n_consumed > 0 && n_buffered > 0)
            memmove(buf.data(), buf.data() + n_consumed,
                    n_buffered * sizeof(complexf));
        abs0 += (uint64_t)n_consumed;
    }

    printf("stride %.4f: %d batches, %lu symbols, %lu samples consumed\n"
           "  max exact-reference phase err= %.3e rad\n"
           "  max intra-batch phase err   = %.6f rad\n"
           "  max batch-boundary phase err= %.6f rad\n"
           "  max cumulative phase err    = %.6f rad\n"
           "  carry range                 = [%.6f, %.6f]\n",
           stride, n_batches, (unsigned long)total_syms,
           (unsigned long)total_consumed,
           max_exact_err,
           max_intra_err, max_boundary_err, max_cum_err,
           min_carry, max_carry);

    // EXACT check: the GPU output must match the deterministic FIR phase
    // to fp32 accumulation noise (~1e-5 rad). One subfilter step of NCO
    // error is theta/128 = 3.9e-4 rad, so tol_exact = 2e-4 catches even a
    // single-step indexing/carry bug; a 0.05-sample carry bug is 2.5e-3.
    const double tol_exact = 2e-4;
    if (max_exact_err > tol_exact) {
        fprintf(stderr, "FAIL: exact-reference phase error %.6f > %.6f "
                "(NCO position/subfilter selection wrong)\n",
                max_exact_err, tol_exact);
        failures++;
    }

    // UNIFORMITY tolerances: per-symbol increments deviate from the ideal
    // theta*stride by (a) the 1/128 position quantization, theta/256 per
    // symbol, and (b) the bank's branch group-delay ripple (truncated
    // prototype, measured above from the actual taps). Differential errors are bounded
    // by the p-p ripple plus two quantization steps.
    const double tol = bank_ripple + theta / 128.0 + 1e-3;
    if (max_intra_err > tol) {
        fprintf(stderr, "FAIL: intra-batch phase error %.6f > %.6f\n",
                max_intra_err, tol);
        failures++;
    }
    if (max_boundary_err > tol) {
        fprintf(stderr, "FAIL: batch-boundary phase error %.6f > %.6f "
                "(NCO carry broken)\n", max_boundary_err, tol);
        failures++;
    }
    if (max_cum_err > 2.0 * tol) {
        fprintf(stderr, "FAIL: cumulative phase drift %.6f > %.6f "
                "(NCO carry drifts)\n", max_cum_err, 2.0 * tol);
        failures++;
    }
    if (min_carry < -1e-9 || max_carry >= 1.0 + 1e-9) {
        fprintf(stderr, "FAIL: NCO carry left [0, 1): [%.6f, %.6f]\n",
                min_carry, max_carry);
        failures++;
    }
    return failures;
}

int main()
{
    int failures = 0;
    failures += run_stride(4.096, 1200); // fractional (OTA hw_sps)
    failures += run_stride(4.0, 300);    // integer sanity
    failures += run_stride(2.0, 300);    // minimum stride sanity
    if (failures == 0) {
        printf("test_timing_nco: PASS\n");
        return 0;
    }
    fprintf(stderr, "test_timing_nco: FAIL (%d checks)\n", failures);
    return 1;
}
