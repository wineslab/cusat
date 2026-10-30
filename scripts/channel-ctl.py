#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
channel-ctl.py — runtime channel control for dvbs2-uhd-trx, over the SAME UDP
interface the SNR/pathloss dashboard uses (8-byte packet: uint32 magic + float32).

Four commands, distinguished by magic:
  NOIS (0x4E4F4953): SNR / direct path loss in dB
                     (--snr / --pathloss-db)
  DIST (0x44495354): distance used only to derive relative free-space path loss
                     (--pathloss-distance-km). Propagation delay remains fixed
                     at the YAML startup distance.
  LINK (0x4C494E4B): path availability, 1=up and 0=out of view (--link).
  CLER (0x434C4552): discard queued channel events (--clear-schedule).

Run one controller on the management network. Give it the local modem and
the remote modem as targets for a symmetric channel. Do not send control
traffic through the DVB path that the command changes.

Examples:
  # derive path loss from a 71572 km path on two directions:
  channel-ctl.py --pathloss-distance-km 71572 \
      127.0.0.1:5556 10.112.100.85:5556
  # add 6 dB directly, without changing distance or propagation delay:
  channel-ctl.py --pathloss-db 6 127.0.0.1:5556
  # LEO/MEO/GEO presets:
  channel-ctl.py --regime geo 127.0.0.1:5556
  # sweep edge -> zenith -> edge, 1 update/0.5s:
  channel-ctl.py --regime leo --pass-sweep 1100,4600,90 127.0.0.1:5556
  # also set SNR:
  channel-ctl.py --snr 12 --pathloss-distance-km 8000 127.0.0.1:5556
  # emulate loss and recovery of line of sight:
  channel-ctl.py --link down 127.0.0.1:5556
  channel-ctl.py --link up 127.0.0.1:5556
"""
import argparse
import socket
import struct
import sys
import time

NOIS = 0x4E4F4953
DIST = 0x44495354
LINK = 0x4C494E4B
CLEAR = 0x434C4552

# Zenith bent-pipe path presets (km): twice the orbital altitude.
REGIMES = {
    "leo": 1100,
    "meo": 16000,
    "geo": 71572,
}


def send(sock, magic, value, targets):
    pkt = struct.pack("<If", magic, float(value))
    for host, port in targets:
        sock.sendto(pkt, (host, port))


def parse_target(s):
    host, _, port = s.partition(":")
    return host, int(port or 5556)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+",
                    help="management endpoint for each selected modem")
    ap.add_argument("--pathloss-distance-km", type=float,
                    help="distance used to derive relative free-space path loss; "
                         "does not change propagation delay (km)")
    ap.add_argument("--distance-km", dest="legacy_distance_km", type=float,
                    help=argparse.SUPPRESS)
    ap.add_argument("--regime", choices=REGIMES,
                    help="LEO/MEO/GEO path-loss-distance preset; does not change delay")
    ap.add_argument("--snr", type=float, help="SNR slider value (NOIS, dB)")
    ap.add_argument("--pathloss-db", type=float,
                    help="direct differential path loss; does not change distance or delay (dB)")
    ap.add_argument("--pathloss", dest="legacy_pathloss_db", type=float,
                    help=argparse.SUPPRESS)
    ap.add_argument("--link", choices=("up", "down"),
                    help="set path availability independently of distance")
    ap.add_argument("--clear-schedule", action="store_true",
                    help="discard queued events and retain the current state")
    ap.add_argument("--pass-sweep", metavar="CLOSEST,EDGE,SECS",
                    help="ramp path-loss distance EDGE->CLOSEST->EDGE over SECS; "
                         "does not change delay")
    ap.add_argument("--period", type=float, default=0.5, help="sweep update period (s)")
    args = ap.parse_args()

    targets = [parse_target(t) for t in args.targets]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    if args.snr is not None:
        send(sock, NOIS, args.snr, targets)
        print(f"NOIS snr={args.snr} dB -> {len(targets)} target(s)")
    if (args.pathloss_db is not None and
            args.legacy_pathloss_db is not None):
        ap.error("use only --pathloss-db (legacy --pathloss was also given)")
    pathloss_db = args.pathloss_db
    if args.legacy_pathloss_db is not None:
        pathloss_db = args.legacy_pathloss_db
        print("warning: --pathloss is deprecated; use --pathloss-db",
              file=sys.stderr)
    if pathloss_db is not None:
        send(sock, NOIS, pathloss_db, targets)
        print(f"NOIS pathloss_db={pathloss_db} dB; distance and propagation "
              f"delay unchanged -> {len(targets)} target(s)")

    if (args.pathloss_distance_km is not None and
            args.legacy_distance_km is not None):
        ap.error("use only --pathloss-distance-km "
                 "(legacy --distance-km was also given)")
    dist = args.pathloss_distance_km
    if args.legacy_distance_km is not None:
        dist = args.legacy_distance_km
        print("warning: --distance-km is deprecated; use "
              "--pathloss-distance-km (propagation delay is unchanged)",
              file=sys.stderr)
    if dist is None and args.regime:
        dist = REGIMES[args.regime]

    if args.pass_sweep:
        lo, hi, secs = (float(x) for x in args.pass_sweep.split(","))
        t0 = time.time()
        while True:
            t = time.time() - t0
            if t > secs:
                break
            # Edge -> closest -> edge (approaching then receding).
            frac = t / secs
            tri = 1.0 - abs(2.0 * frac - 1.0)        # 0..1..0
            d = lo + (hi - lo) * (1.0 - tri)          # closest at mid-pass
            send(sock, DIST, d, targets)
            print(f"DIST pathloss_distance={d:8.1f} km  "
                  f"(propagation delay unchanged)  t={t:5.1f}s", flush=True)
            time.sleep(args.period)
        return

    if dist is not None:
        send(sock, DIST, dist, targets)
        print(f"DIST pathloss_distance={dist:.1f} km; propagation delay "
              f"unchanged -> {len(targets)} target(s)")

    if args.link is not None:
        available = args.link == "up"
        send(sock, LINK, 1.0 if available else 0.0, targets)
        print(f"LINK {args.link} -> {len(targets)} target(s)")

    if args.clear_schedule:
        send(sock, CLEAR, 0.0, targets)
        print(f"CLER future schedule -> {len(targets)} target(s)")

    if (args.snr is None and pathloss_db is None and dist is None and
            args.link is None and not args.clear_schedule):
        ap.error("nothing to send: give --pathloss-distance-km/--regime/"
                 "--snr/--pathloss-db/--link/--clear-schedule")


if __name__ == "__main__":
    main()
