#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Verify the LDPC parity-check table headers in lib/cusat_dsp/ldpc_tables/:
# re-parse every table, recompute its structural invariants and compare a
# checksum of its numeric content against the manifest checksums.txt.
#
#   --etsi S2.pdf S2X.pdf  also compare every table, number by number, with
#                          the annex text of the ETSI documents (needs
#                          pdftotext; the PDFs are free from www.etsi.org)
#   --rewrite              re-emit the headers from the parsed tables (layout
#                          only; the content checksums must not change)
#
# The accumulator addresses are those published in:
#   - ETSI EN 302 307-1 V1.4.1, Annex B (normal FECFRAME) and Annex C
#     (short FECFRAME) — DVB-S2
#   - ETSI EN 302 307-2 V1.3.1, Annex B/C — DVB-S2X (incl. the three
#     medium-frame tables)

import argparse
import hashlib
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(REPO, "lib", "cusat_dsp", "ldpc_tables")
MANIFEST = os.path.join(TABLES_DIR, "checksums.txt")

# Table header per code family: (file, ETSI citation lines)
FAMILIES = {
    "s2": (
        "dvb_s2_tables.h",
        ["ETSI EN 302 307-1 V1.4.1, Annex B (normal FECFRAME, tables B.1-B.11)",
         "and Annex C (short FECFRAME, tables C.1-C.10)."],
    ),
    "s2x": (
        "dvb_s2x_tables.h",
        ["ETSI EN 302 307-2 V1.3.1, Annex B (normal FECFRAME) and Annex C",
         "(short FECFRAME), including the medium-frame tables (C.8-C.10)."],
    ),
}

class Code:
    """One LDPC code: dimensions + per-row accumulator addresses."""

    def __init__(self, name, n, k, m, rows):
        self.name = name      # e.g. "s2_b1"
        self.n = n
        self.k = k
        self.m = m
        self.rows = rows      # list[list[int]] — addresses per address row

    @property
    def r(self):
        return self.n - self.k

    @property
    def q(self):
        return self.r // self.m

    def validate(self):
        assert self.r % self.m == 0, f"{self.name}: R not divisible by M"
        assert self.k == len(self.rows) * self.m, \
            f"{self.name}: K != n_rows * M"
        for row in self.rows:
            assert 0 < len(row) <= 255, f"{self.name}: bad row degree"
            for a in row:
                assert 0 <= a < self.r, f"{self.name}: address {a} >= R"
                assert a < 65536, f"{self.name}: address {a} overflows uint16"

    def links_total(self):
        """Total Tanner edges incl. parity chain (decoder BNL sizing)."""
        bit_links = self.m * sum(len(row) for row in self.rows)
        return bit_links + 2 * self.r - 1

    def links_max_cn(self):
        """Max check-node degree incl. the 2 parity edges."""
        cnc = [0] * self.r
        for row in self.rows:
            for j in range(self.m):
                shift = j * self.q
                for a in row:
                    cnc[(a + shift) % self.r] += 1
        return max(cnc) + 2

    def canonical(self):
        """Canonical string of the numeric content (for checksums)."""
        parts = [f"{self.name} N={self.n} K={self.k} M={self.m}"]
        for row in self.rows:
            parts.append(" ".join(str(a) for a in row))
        return "\n".join(parts) + "\n"

    def checksum(self):
        return hashlib.sha256(self.canonical().encode()).hexdigest()


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

def parse_generated(path):
    text = open(path).read()
    arrays = {}
    for mm in re.finditer(
            r"inline constexpr (?:uint8_t|uint16_t|uint32_t) (\w+)\[\] = \{(.*?)\};",
            text, re.S):
        arrays[mm.group(1)] = [int(x) for x in
                               mm.group(2).replace(",", " ").split()]
    codes = []
    for mm in re.finditer(
            r"inline constexpr ldpc_code (\w+) = \{\s*"
            r"(\d+), (\d+), (\d+), (\d+), (\d+), (\d+),", text):
        name = mm.group(1)
        n, k, m, lt, lmc, n_rows = (int(mm.group(i)) for i in range(2, 8))
        deg = arrays[f"{name}_deg"]
        off = arrays[f"{name}_off"]
        addr = arrays[f"{name}_addr"]
        assert len(deg) == n_rows and len(off) == n_rows + 1
        assert off[0] == 0 and off[-1] == len(addr)
        rows = [addr[off[i]:off[i + 1]] for i in range(n_rows)]
        assert [len(row) for row in rows] == deg
        code = Code(name, n, k, m, rows)
        code.validate()
        assert code.links_total() == lt, f"{name}: links_total mismatch"
        assert code.links_max_cn() == lmc, f"{name}: links_max_cn mismatch"
        codes.append(code)
    return codes


def load_all_generated():
    out = {}
    for family, (fname, _) in FAMILIES.items():
        out[family] = parse_generated(os.path.join(TABLES_DIR, fname))
    return out


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def do_check():
    manifest = {}
    for line in open(MANIFEST):
        if line.startswith("#"):
            continue
        csum, name = line.split()
        manifest[name] = csum
    seen = 0
    for family, codes in load_all_generated().items():
        for code in codes:
            assert code.name in manifest, f"{code.name}: not in manifest"
            assert code.checksum() == manifest[code.name], \
                f"{code.name}: checksum mismatch!"
            seen += 1
    assert seen == len(manifest), "manifest has extra entries"
    print(f"OK: {seen} tables match {os.path.relpath(MANIFEST, REPO)} "
          f"and all structural invariants hold")


ETSI_DOC = {"s2": "EN 302 307-1", "s2x": "EN 302 307-2"}


def etsi_citation(name):
    family, table = name.split("_")
    return f"{ETSI_DOC[family]} Table {table[0].upper()}.{table[1:]}"


def wrap(values, per_line=16, indent="    "):
    lines = []
    for i in range(0, len(values), per_line):
        lines.append(indent + ", ".join(str(v) for v in values[i:i + per_line]) + ",")
    return "\n".join(lines)


def emit_family(family, codes):
    fname, citation = FAMILIES[family]
    guard = "INCLUDED_CUSAT_DSP_LDPC_TABLES_" + fname.replace(".", "_").upper()
    out = ["/*",
           " * SPDX-License-Identifier: Apache-2.0",
           " *",
           " * LDPC accumulator addresses from",
           " *   " + citation[0],
           " *   " + citation[1],
           " * The numbers are those of the standard; scripts/check_ldpc_tables.py",
           " * --etsi compares them with the ETSI documents. DO NOT EDIT BY HAND.",
           " */", "", f"#ifndef {guard}", f"#define {guard}", "", '#include "ldpc_code.h"', "",
           "namespace cusat_dsp {", "namespace ldpc_tables {", ""]
    for c in codes:
        deg = [len(r) for r in c.rows]
        off = [0]
        for d in deg:
            off.append(off[-1] + d)
        addr = [a for r in c.rows for a in r]
        out += [f"/* {etsi_citation(c.name)}: N = {c.n}, K = {c.k} */",
                f"inline constexpr uint8_t {c.name}_deg[] = {{", wrap(deg, 24), "};",
                f"inline constexpr uint32_t {c.name}_off[] = {{", wrap(off), "};",
                f"inline constexpr uint16_t {c.name}_addr[] = {{", wrap(addr), "};",
                f"inline constexpr ldpc_code {c.name} = {{",
                f"    {c.n}, {c.k}, {c.m}, {c.links_total()}, {c.links_max_cn()}, {len(c.rows)},",
                f"    {c.name}_deg, {c.name}_off, {c.name}_addr,", "};", ""]
    out += ["} // namespace ldpc_tables", "} // namespace cusat_dsp", "", f"#endif /* {guard} */", ""]
    return "\n".join(out)


def do_rewrite():
    before = {c.name: c.checksum() for codes in load_all_generated().values() for c in codes}
    for family, codes in load_all_generated().items():
        with open(os.path.join(TABLES_DIR, FAMILIES[family][0]), "w") as f:
            f.write(emit_family(family, codes))
    after = {c.name: c.checksum() for codes in load_all_generated().values() for c in codes}
    assert before == after, "rewrite changed table content!"
    print(f"rewrote {len(after)} tables; content checksums unchanged")


def etsi_annex_tables(pdf, family):
    """Annex B/C tables of an ETSI PDF as flat address lists. The annexes
    print tables in several columns and wrap long rows, so each page is read
    column by column and the numbers are concatenated."""
    import subprocess
    text = subprocess.run(["pdftotext", "-layout", pdf, "-"], check=True,
                          capture_output=True, text=True).stdout.split("\n")
    tables, cur, page = {}, None, []

    def flush():
        if cur is None or not page:
            return
        segs = []
        for li, line in enumerate(page):
            for mm in re.finditer(r"\S+(?: \S+)*", line):
                toks = mm.group(0).split()
                if toks and all(t.isdigit() for t in toks):
                    segs.append((mm.start(), li, [int(t) for t in toks]))
        cols = []
        for x in sorted(set(sg[0] for sg in segs)):
            if cols and x - cols[-1][-1] <= 6:
                cols[-1].append(x)
            else:
                cols.append([x])
        col_of = lambda x: next(i for i, c in enumerate(cols) if c[0] - 1 <= x <= c[-1] + 1)
        for sg in sorted(segs, key=lambda sg: (col_of(sg[0]), sg[1])):
            tables[cur].extend(sg[2])

    for line in text:
        mm = re.match(r"^\s*Table ([BC])\.(\d+):", line)
        if mm:
            flush()
            page, cur = [], f"{family}_{mm.group(1).lower()}{mm.group(2)}"
            tables[cur] = []
        elif cur is None:
            continue
        elif "ETSI EN 302 307" in line or line.strip() == "ETSI":
            flush()
            page = []
        elif re.match(r"^\s*(Annex [D-Z]|History|Table [A-Z]\.|[A-Z]\.\d+\s+\w)", line):
            flush()
            page, cur = [], None
        else:
            page.append(line)
    flush()
    return tables


def do_etsi(s2_pdf, s2x_pdf):
    etsi = {}
    etsi.update(etsi_annex_tables(s2_pdf, "s2"))
    etsi.update(etsi_annex_tables(s2x_pdf, "s2x"))
    n_codes = n_addr = 0
    for codes in load_all_generated().values():
        for c in codes:
            flat = [a for r in c.rows for a in r]
            assert etsi.get(c.name) == flat, f"{c.name}: differs from {etsi_citation(c.name)}"
            n_codes += 1
            n_addr += len(flat)
    print(f"OK: {n_codes} tables ({n_addr} addresses) identical to the ETSI annexes")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--etsi", nargs=2, metavar=("EN302307-1.pdf", "EN302307-2.pdf"))
    ap.add_argument("--rewrite", action="store_true")
    args = ap.parse_args()
    if args.rewrite:
        do_rewrite()
    do_check()
    if args.etsi:
        do_etsi(*args.etsi)


if __name__ == "__main__":
    main()
