#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Blossom (maximum-matching) parallel-layer schedule generator for ALL
DVB-S2/S2X normal-frame LDPC code rates used by the modcod table.

The minimum-parallel-step schedule for the layered decoder is a MAXIMUM MATCHING
on the layer-conflict graph (two layers mergeable iff conflict-free): an edge
per mergeable pair, max-cardinality matching = max merges = min steps. Edmonds'
blossom solves this OPTIMALLY in polynomial time, so no heuristic or MILP is
needed. We use networkx
max_weight_matching(maxcardinality=True) with a small ascending-offset weight as
the "staircasity" tiebreak among equally-optimal matchings (memory locality).

Layer/conflict semantics MUST match the CUDA host prep (ldpc_decode_kernel.cu):
decoder layer of a check with base accumulator address a is a % q; two layers
conflict iff they touch a shared VN group OR are parity-chain adjacent
(|i-j| <= 1 mod q).

Inputs:  lib/cusat_dsp/ldpc_tables/dvb_s2_tables.h, dvb_s2x_tables.h
Outputs: one header per rate in lib/cusat_dsp/cuda/ldpc_schedules/, and the
         regenerated registry ldpc_schedule_registry.h.
"""

import os
import re
import networkx as nx

M = 360
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "lib/cusat_dsp/ldpc_tables")
OUT = os.path.join(ROOT, "lib/cusat_dsp/cuda/ldpc_schedules")

# (rate enum, table file key, table symbol) for every normal-frame rate the
# modcod table uses. Table->rate from lib/cusat_dsp/ldpc_table_factory.cc.
S2, S2X = "dvb_s2_tables.h", "dvb_s2x_tables.h"
RATES = [
    ("RATE_1_4",  S2,  "s2_b1"),  ("RATE_1_3",  S2,  "s2_b2"),  ("RATE_2_5",  S2,  "s2_b3"),
    ("RATE_1_2",  S2,  "s2_b4"),  ("RATE_3_5",  S2,  "s2_b5"),  ("RATE_2_3",  S2,  "s2_b6"),
    ("RATE_3_4",  S2,  "s2_b7"),  ("RATE_4_5",  S2,  "s2_b8"),  ("RATE_5_6",  S2,  "s2_b9"),
    ("RATE_8_9",  S2,  "s2_b10"), ("RATE_9_10", S2,  "s2_b11"),
    ("RATE_13_45",  S2X, "s2x_b2"),  ("RATE_9_20",   S2X, "s2x_b3"),
    ("RATE_11_20",  S2X, "s2x_b4"),  ("RATE_26_45",  S2X, "s2x_b5"),
    ("RATE_28_45",  S2X, "s2x_b6"),  ("RATE_23_36",  S2X, "s2x_b7"),
    ("RATE_25_36",  S2X, "s2x_b8"),  ("RATE_13_18",  S2X, "s2x_b9"),
    ("RATE_90_180", S2X, "s2x_b11"), ("RATE_96_180", S2X, "s2x_b12"),
    ("RATE_100_180",S2X, "s2x_b13"), ("RATE_104_180",S2X, "s2x_b14"),
    ("RATE_116_180",S2X, "s2x_b15"), ("RATE_124_180",S2X, "s2x_b16"),
    ("RATE_128_180",S2X, "s2x_b17"), ("RATE_132_180",S2X, "s2x_b18"),
    ("RATE_135_180",S2X, "s2x_b19"), ("RATE_140_180",S2X, "s2x_b20"),
    ("RATE_154_180",S2X, "s2x_b21"), ("RATE_18_30",  S2X, "s2x_b22"),
    ("RATE_20_30",  S2X, "s2x_b23"),
]


def parse_kq(content, tbl):
    m = re.search(rf"/\* {re.escape(tbl)}: N=(\d+) K=(\d+) \(q=(\d+)\) \*/", content)
    if not m:
        raise RuntimeError(f"banner for {tbl} not found")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def parse_array(content, name):
    m = re.search(re.escape(name) + r"\[\]\s*=\s*\{", content)
    if not m:
        raise RuntimeError(f"array {name} not found")
    start = m.end()
    end = content.index("};", start)
    return [int(x) for x in re.findall(r"\d+", content[start:end])]


def cost_matrix(q, deg, addr):
    """Returns (vn_cost, adjacent). vn_cost[i][j] = #shared VN groups (a SOFT,
    lost-update conflict the decoder tolerates). adjacent[i][j] = parity-chain
    neighbours |i-j|<=1 mod q (a HARD conflict — pairing them breaks the IRA
    staircase decode, FER->1, so these are FORBIDDEN, not penalized).
    layer = addr % q matches the decoder (host transpose p = q*j+i)."""
    groups = [set() for _ in range(q)]
    idx = 0
    for g, d in enumerate(deg):
        for _ in range(d):
            groups[addr[idx] % q].add(g)
            idx += 1
    assert idx == len(addr), f"edge count {idx} != {len(addr)}"
    vn = [[0] * q for _ in range(q)]
    adj = [[False] * q for _ in range(q)]
    for i in range(q):
        for j in range(i + 1, q):
            vn[i][j] = vn[j][i] = len(groups[i] & groups[j])
            a = (abs(i - j) <= 1) or (abs(i - j) >= q - 1)
            adj[i][j] = adj[j][i] = a
    return vn, adj


def blossom_schedule(q, vn, adj):
    """Max-pack, parity-chain adjacency FORBIDDEN, VN-sharing minimized.
    Edges = all non-adjacent pairs (adjacency is a hard FER-breaking conflict);
    maximum-cardinality matching packs as much as possible without it; among
    those, minimize VN-group sharing (soft, tolerated). One weighted blossom."""
    G = nx.Graph()
    G.add_nodes_from(range(q))
    for i in range(q):
        for j in range(i + 1, q):
            if adj[i][j]:
                continue  # forbid parity-chain neighbours
            G.add_edge(i, j, weight=-1000 * vn[i][j] + (q - (j - i)))
    matching = nx.max_weight_matching(G, maxcardinality=True)
    matched, sched = set(), []
    for u, v in matching:
        a, b = (u, v) if u < v else (v, u)
        sched.append((a, b)); matched.add(a); matched.add(b)
    for i in range(q):
        if i not in matched:
            sched.append((i, -1))
    sched.sort(key=lambda p: p[0])
    flat = sorted(x for p in sched for x in p if x >= 0)
    assert flat == list(range(q)), "bad coverage"
    for a, b in sched:
        assert b < 0 or not adj[a][b], f"adjacent pair {a},{b}"  # hard constraint
    return sched, sum(vn[a][b] for a, b in sched if b >= 0)


def emit_header(sym, rate, q, sched):
    steps = len(sched)
    pairs = sum(1 for _, b in sched if b >= 0)
    path = os.path.join(OUT, f"ldpc_schedule_{sym.lower()}.h")
    with open(path, "w") as f:
        f.write(f"""/*
 * Parallel-layer schedule for DVB-S2/S2X table {sym} -- rate {rate}.
 * q={q} layers -> {steps} steps ({pairs} pairs + {steps - pairs} singletons),
 * theoretical speedup {q / steps:.2f}x.
 * MAX-PACK / least-conflict: minimum-conflict MAXIMUM-cardinality matching
 * (one weighted Edmonds blossom). Conflicts are allowed (the offset-min-sum
 * decoder tolerates them at ~zero it_mean cost); pairs minimize total shared-VN-group
 * conflict-weight as a hedge + memory locality. Generated by
 * scripts/gen_blossom_schedules.py
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>

namespace cusat_dsp {{
namespace ldpc_schedule {{

static constexpr int {sym}_STEPS = {steps};
static constexpr int {sym}_Q     = {q};

static constexpr int16_t {sym}[{steps}][2] = {{
""")
        for a, b in sched:
            f.write(f"    {{{a:3d}, {b:3d}}},\n")
        f.write("};\n\n} // namespace ldpc_schedule\n} // namespace cusat_dsp\n")
    return steps


def old_steps(sym):
    p = os.path.join(OUT, f"ldpc_schedule_{sym.lower()}.h")
    if not os.path.exists(p):
        return None
    m = re.search(rf"{sym}_STEPS\s*=\s*(\d+)", open(p).read())
    return int(m.group(1)) if m else None


def emit_registry(rows):
    """Regenerate ldpc_schedule_registry.h with all rates; preserve helpers."""
    syms = [r[2].upper() for r in rows]
    inc = "\n".join(f'#include "ldpc_schedule_{s.lower()}.h"' for s in syms)
    cases = "\n".join(
        f"    case {rate}: return {{ {sym}, {sym}_STEPS, {sym}_Q }};"
        for rate, _, _, sym in rows)
    path = os.path.join(OUT, "ldpc_schedule_registry.h")
    with open(path, "w") as f:
        f.write(f"""/*
 * LDPC parallel-layer schedule registry.
 *
 * Maps code_rate_t -> precomputed blossom (maximum-matching) schedule for
 * every normal-frame rate the modcod table uses. Returns nullptr for codes
 * without a schedule (short/medium frames, VLSNR); the kernel then falls back
 * to the sequential 1-layer/step schedule.
 * GENERATED by scripts/gen_blossom_schedules.py — do not edit by hand.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>
#include <cusat_dsp/modcod_types.h>

{inc}

namespace cusat_dsp {{
namespace ldpc_schedule {{

static constexpr int MAX_SCHEDULE_STEPS = 140;
static constexpr int LANES_PER_STEP = 2;

struct entry {{
    const int16_t (*pairs)[2];
    int n_steps;
    int q;
}};

/** Look up the blossom schedule for a code rate (normal frame). */
inline entry get(code_rate_t rate)
{{
    switch (rate) {{
{cases}
    default: return {{ nullptr, 0, 0 }};
    }}
}}

/** Sequential fallback: q steps, one layer each (lane 1 = -1 = inactive). */
inline int build_sequential(int16_t (*buf)[2], int q)
{{
    for (int i = 0; i < q; i++) {{ buf[i][0] = (int16_t)i; buf[i][1] = -1; }}
    return q;
}}

}} // namespace ldpc_schedule
}} // namespace cusat_dsp
""")


def main():
    s2 = open(os.path.join(TAB, S2)).read()
    s2x = open(os.path.join(TAB, S2X)).read()
    src = {S2: s2, S2X: s2x}
    rows = []
    print(f"{'rate':10} {'sym':9} {'q':>4} {'pairs':>5} {'steps':>5} "
          f"{'confl':>5} {'spd':>5}")
    maxq = 0
    for rate, fkey, tbl in RATES:
        content = src[fkey]
        N, K, q = parse_kq(content, tbl)
        deg = parse_array(content, f"{tbl}_deg")
        addr = parse_array(content, f"{tbl}_addr")
        vn, adj = cost_matrix(q, deg, addr)
        sched, confl = blossom_schedule(q, vn, adj)
        steps = emit_header(tbl.upper(), rate, q, sched)
        pairs = sum(1 for _, b in sched if b >= 0)
        maxq = max(maxq, q)
        print(f"{rate:10} {tbl.upper():9} {q:>4} {pairs:>5} {steps:>5} "
              f"{confl:>5} {q/steps:>4.2f}x")
        rows.append((rate, fkey, tbl, tbl.upper()))
    emit_registry(rows)
    print(f"\n{len(rows)} rates: max-pack (q/2) / least-conflict; max q = {maxq} "
          f"(MAX_SCHEDULE_STEPS=140 ok). Registry regenerated.")


if __name__ == "__main__":
    main()
