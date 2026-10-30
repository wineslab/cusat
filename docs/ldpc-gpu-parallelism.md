# LDPC GPU Parallelism — Analysis & Trade-offs

Notes on the CUDA LDPC decoder across GH200 and GB10 (DGX Spark).

---

## 1. Shared Memory Constraint: GH200 vs GB10

The layered decoder loads the full `K+R` LLR working set into shared memory per
block to get ~5ns access latency vs ~30-40ns for L2.

| GPU | Smem optin limit | Normal frame (K+R)×2 | Path |
|-----|-----------------|----------------------|------|
| GH200 (sm_90) | ~228 KB | ~126 KB | shared memory ✓ |
| GB10 (sm_121) | ~99 KB | ~126 KB | L2-pinned global ✗ |

**Fix** (`ldpc_decode_kernel.cu`): at init the decoder queries
`cudaDevAttrMaxSharedMemoryPerBlockOptin` and selects between two kernel
template instantiations — `ldpc_decode_layered_kernel<true>` (shared memory)
and `<false>` (global memory, only 4 bytes of shared for `s_converged_p`).
`__syncthreads()` correctly fences global writes between parallel-layer steps.

### Occupancy implication

Shared memory is a finite SM resource. At 126 KB/block, GH200 fits **1 block
per SM**. With global memory (4 bytes/block), the limit shifts to
registers/threads — potentially **2-4 blocks per SM**, processing that many
codewords concurrently and hiding L2 latency. Worth benchmarking
`d_use_smem=false` on GH200 for large batches.

---

## 2. MILP Parallel-Layer Schedule

### Background

DVB-S2X IRA LDPC codes have `q` layers. Two layers conflict if they share:
- **Parity nodes**: adjacent layers `|i−j| ≤ 1` or `|i−j| ≥ q−1` (wrap-around)
  — this is structural and unavoidable
- **Data variable nodes**: determined per-code from the parity-check matrix

The schedule generator (`scripts/gen_blossom_schedules.py`, Blossom maximum matching) finds a maximum matching of
non-conflicting layer pairs and orders them to minimise schedule inversions
(preserving the ascending IRA sweep for fast convergence).

### Current results (LANES_PER_STEP = 2)

| Code | q | Steps | Pairs | Speedup |
|------|---|-------|-------|---------|
| B8  (DVB-S2 4/5) | — | — | — | — |
| B18 (64APSK 11/15) | — | — | — | — |
| B19 (128APSK 3/4) | 45 | 23 | 22 | **1.96×** |
| B20 (64APSK 7/9) | — | — | — | — |

All other rates fall back to sequential (q steps, 1 layer/step).

---

## 3. Generalising to N Parallel Lanes

### Hard ceiling from the parity ring

The IRA parity structure creates a conflict cycle C_q. The independence number
of C_q is `⌊q/2⌋`. For B19: **max 22 layers can run simultaneously**, regardless
of N — this is an absolute hardware-independent ceiling.

### Theoretical step counts for B19 (q = 45)

| N lanes | Min steps | Speedup | Complexity |
|---------|-----------|---------|------------|
| 1 | 45 | 1.0× | — |
| 2 | 23 | 1.96× | max graph matching (polynomial) ← current |
| 3 | ~15 | ~3.0× | 3-dim hypergraph matching (NP-hard, tractable at q=45) |
| 4 | ~12 | ~3.75× | harder, diminishing returns |
| ∞ | ~23\* | ~2.0×\* | collapses — 22 paired + 23 singletons remaining |

\*The ∞ case is worse than N=3: running the maximum 22 independent layers
simultaneously still leaves 23 singleton steps, giving fewer total steps than
grouping into triplets.

### Convergence degradation

More parallelism moves the decoder toward "flooding" BP (all layers at once),
which converges slower or not at all. The MILP's inversion-minimisation
objective preserves the ascending IRA sweep order to counteract this. At N=2
this works well (10% disorder for B19). At N=3+ the ordering constraint is
harder to satisfy and convergence degradation may partially offset the step
reduction — requiring more trials to decode.

**Sweet spot**: likely N=3, still tractable by MILP for q≤45, with a stricter
inversion penalty to protect convergence.

---

## 4. GB10 vs GH200: Relative Benefit

| Optimization | GH200 | GB10 |
|---|---|---|
| MILP step reduction | dominant (cheap smem, 1 block/SM) | helps, but each step costs more (L2) |
| Higher occupancy (global mem path) | secondary | primary latency-hiding mechanism |
| Combined | additive | intertwined — more blocks/SM amortise per-step L2 cost |

The MILP schedule benefit (45→23 steps) is the same absolute reduction on both
GPUs. On GH200 this translates directly to faster decode. On GB10 the gain is
partially offset by higher L2 latency per step, but this is in turn compensated
by higher occupancy.

---

## 5. Open Work

- [ ] Benchmark `d_use_smem=false` on GH200 for large batches (occupancy vs smem)
- [ ] Generate MILP schedules for remaining code rates (currently only B8/B18/B19/B20)
- [ ] Investigate N=3 hypergraph matching extension for high-rate codes with large q
- [ ] Profile convergence penalty of N=3 (extra trials vs fewer steps)
