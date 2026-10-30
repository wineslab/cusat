#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate a live DIFI stream with the DIFI Consortium's own checkers.

Binds a UDP socket, receives packets from our libradio_difi.so TX, and runs
each through the packet definitions + validate() functions from
DIFI-Consortium/DIFI-Certification (third-party ground truth). Usage:

    git clone https://github.com/DIFI-Consortium/DIFI-Certification /tmp/difi-cert
    pip install construct
    python3 difi_certify_stream.py --cert-repo /tmp/difi-cert --port 50150 --n 500

Then point the transmitter at this host:port (radio.driver: difi,
difi_tx_addr: <host>:50150).
"""
import argparse
import socket
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cert-repo", required=True,
                    help="path to a DIFI-Certification checkout")
    ap.add_argument("--port", type=int, default=50150)
    ap.add_argument("--n", type=int, default=500, help="packets to check")
    ap.add_argument("--timeout", type=float, default=20.0)
    a = ap.parse_args()

    sys.path.insert(0, a.cert_repo + "/packet_definitions")
    import difi_data_v1_1 as ddata          # noqa: E402
    import difi_context_v1_1 as dctx        # noqa: E402

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", a.port))
    s.settimeout(a.timeout)
    print(f"listening on :{a.port} for {a.n} packets ...")

    counts = {"data": 0, "context": 0, "unknown": 0}
    errors = {}
    for i in range(a.n):
        try:
            pkt, _ = s.recvfrom(65536)
        except socket.timeout:
            print(f"TIMEOUT after {i} packets")
            break
        ptype = (pkt[0] >> 4) & 0xF
        try:
            if ptype == 0x1:
                counts["data"] += 1
                parsed = ddata.difi_data_definition.parse(pkt)
                errs = ddata.difi_data_definition.validate(parsed)
            elif ptype == 0x4:
                counts["context"] += 1
                parsed = dctx.difi_context_definition.parse(pkt)
                errs = dctx.difi_context_definition.validate(parsed)
                print(f"  context: rf={parsed.rfFreq/1e9:.4f} GHz "
                      f"rate={parsed.sampleRate/1e6:.3f} Msps "
                      f"bw={parsed.bandwidth/1e6:.3f} MHz")
            else:
                counts["unknown"] += 1
                errs = [f"unknown packet type {ptype}"]
        except Exception as e:
            errs = [f"parse exception: {e}"]
        for e in errs:
            errors[e] = errors.get(e, 0) + 1

    print(f"\npackets: {counts}")
    if errors:
        print("VALIDATION ERRORS (consortium checker):")
        for e, n in sorted(errors.items(), key=lambda kv: -kv[1]):
            print(f"  [{n:>5}x] {e}")
        sys.exit(1)
    print("ALL PACKETS PASS the DIFI-Consortium v1.1 checks")


if __name__ == "__main__":
    main()
