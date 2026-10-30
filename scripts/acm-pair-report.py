#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate an acm-tcp-pair.sh run against the two-modem ACM acceptance criteria.

usage: acm-pair-report.py <out_dir>

(a) no hard unlock / app-visible Lock=False after first lock
(b) MODCOD follows SNR: rank correlation of the SNR reported by the
    receiving node with the spectral efficiency of the MODCOD the other
    node transmits, in both directions
(c) post-BCH FER in steady segments (TX MODCOD unchanged for +-3 s)
(d) frames lost per MODCOD transition = (data frames sent by one node -
    FECFRAMEs decoded by the other) / transitions
"""
import re
import sys

LINE = re.compile(r"^(?:([\d.]+) )?log :info: (\d+)s Lock=(\w+); pilotSNR=([-\d.]+).*?"
                  r"FECFRAMEs=(\d+); FER=([\d.e+-]+);.*?TxFrames=(\d+);.*?"
                  r"TxMC=(\S+)\((\d+)\)")
FIN = re.compile(r"finished\. ([\d.]+) sec, TX (\d+) frames .*RX (\d+) FECFRAMEs "
                 r"(\d+) errors, (\d+) PDUs")


def parse(path):
    rows, fin, hard = [], None, 0
    with open(path, errors="replace") as f:
        for ln in f:
            if "HARD unlock" in ln:
                hard += 1
            m = LINE.search(ln)
            if m:
                rows.append(dict(w=float(m[1]) if m[1] else None, t=int(m[2]),
                                 lock=m[3] == "True", snr=float(m[4]), fc=int(m[5]),
                                 fer=float(m[6]), tx=int(m[7]), mc=m[8], mci=int(m[9])))
            m = FIN.search(ln)
            if m:
                fin = dict(sec=float(m[1]), tx=int(m[2]), fc=int(m[3]),
                           fe=int(m[4]), pdus=int(m[5]))
    return rows, fin, hard


def at(rows, key, w):
    """counter `key` linearly interpolated at wall time w"""
    for a, b in zip(rows, rows[1:]):
        if a["w"] <= w <= b["w"]:
            f = (w - a["w"]) / (b["w"] - a["w"]) if b["w"] > a["w"] else 0.0
            return a[key] + f * (b[key] - a[key])
    return None


def trajectory(out):
    try:
        with open(f"{out}/trajectory.log") as f:
            return [float(ln.split()[0]) for ln in f if ln.strip()]
    except OSError:
        return []


def efficiency(name):
    """information bits per symbol from a MODCOD name like 16apsk3/5l"""
    m = re.match(r"(qpsk|8psk|8apsk|(\d+)apsk)(\d+)/(\d+)", name)
    if not m:
        return float("nan")
    bps = {"qpsk": 2, "8psk": 3, "8apsk": 3}.get(m[1])
    if bps is None:
        bps = int(m[2]).bit_length() - 1
    return bps * int(m[3]) / int(m[4])


def spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va > 0 and vb > 0 else float("nan")


def main(out):
    nodes = {n: parse(f"{out}/node{n}.log") for n in "AB"}
    ok = True
    for n, (rows, fin, hard) in nodes.items():
        first = next((i for i, r in enumerate(rows) if r["lock"]), None)
        drops = sum(1 for r in rows[first:] if not r["lock"]) if first is not None else -1
        print(f"node{n}: {len(rows)} s logged, HARD unlock={hard}, "
              f"Lock=False seconds after first lock={drops}")
        ok &= hard == 0 and drops == 0
    for tx, rx in (("A", "B"), ("B", "A")):
        trows, tfin, _ = nodes[tx]
        rrows, rfin, _ = nodes[rx]
        mcs = [r["mci"] for r in trows]
        trans = sum(1 for i in range(1, len(mcs)) if mcs[i] != mcs[i - 1])
        snr = [r["snr"] for r in rrows][:len(trows)]
        eff = [efficiency(r["mc"]) for r in trows][:len(snr)]
        rho = spearman(snr, eff) if snr else float("nan")
        # steady seconds: TX MODCOD unchanged for +-3 s; errors from the
        # cumulative FER (2 significant digits, so a bound when nonzero)
        st_fr = st_err = 0
        for i in range(4, min(len(trows), len(rrows)) - 3):
            if len(set(mcs[i - 3:i + 4])) != 1:
                continue
            d_fc = rrows[i]["fc"] - rrows[i - 1]["fc"]
            d_fe = rrows[i]["fer"] * rrows[i]["fc"] - rrows[i - 1]["fer"] * rrows[i - 1]["fc"]
            st_fr += max(d_fc, 0)
            st_err += max(d_fe, 0.0)
        line = (f"{tx}->{rx}: transitions={trans}, rank corr(pilotSNR@{rx}, TxMC bits/sym@{tx})={rho:.2f}"
                f", steady frames={st_fr}, steady errors~{st_err:.1f}")
        traj = trajectory(out)
        if rfin:
            fer = rfin["fe"] / rfin["fc"] if rfin["fc"] else float("nan")
            line += (f"\n    whole run: sent={tfin['tx'] if tfin else '?'} decoded={rfin['fc']} "
                     f"errors={rfin['fe']} (FER {fer:.1e})")
        if traj and trows and trows[0]["w"] is not None:
            # window = trajectory (both ends at the high Es/N0, same MODCOD,
            # so the frames in flight cancel); transitions inside it only
            w0, w1 = traj[0], traj[-1]
            d_tx = at(trows, "tx", w1) - at(trows, "tx", w0)
            d_fc = at(rrows, "fc", w1) - at(rrows, "fc", w0)
            tr_w = sum(1 for a, b in zip(trows, trows[1:])
                       if w0 <= b["w"] <= w1 and a["mci"] != b["mci"])
            lost = d_tx - d_fc
            per = lost / tr_w if tr_w else float("nan")
            line += (f"\n    trajectory window {w1 - w0:.0f} s: sent={d_tx:.0f} "
                     f"decoded={d_fc:.0f} lost={lost:.0f} over {tr_w} transitions "
                     f"-> {per:.2f}/transition")
            ok &= per <= 2.0
        ok &= st_fr > 0 and st_err / st_fr <= 1e-4
        print(line)
    print("RESULT:", "PASS" if ok else "FAIL (see lines above)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/acmpair")
