#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Upload one time-tagged orbital channel schedule to one or more cuSAT nodes."""

import argparse
import datetime
import math
import os
import select
import socket
import struct
import sys
import time
from dataclasses import dataclass


DIST_MAGIC = 0x44495354  # DIST
LINK_MAGIC = 0x4C494E4B  # LINK
CLEAR_MAGIC = 0x434C4552  # CLER
APPLIED_ACK_MAGIC = 0x41434B31  # ACK1
QUEUED_ACK_MAGIC = 0x41434B51  # ACKQ
ACK_LATE_ON_ARRIVAL = 1 << 0
ACK_QUEUE_FULL = 1 << 1
SCHEDULED_COMMAND = struct.Struct("<IfQQ")
SCHEDULED_ACK = struct.Struct("<IIQQQ")
LEGACY_COMMAND = struct.Struct("<If")
MAX_SCHEDULE_COMMANDS = 32768

R_EARTH_KM = 6371.0
MU_EARTH_KM3_S2 = 398600.4418
REGIME_ALT_KM = {"leo": 550.0, "meo": 8000.0, "geo": 35786.0}


@dataclass(frozen=True)
class OrbitPoint:
    offset_s: float
    relative_fspl_db: float
    slant_range_km: float
    pathloss_distance_km: float
    elevation_deg: float
    link_available: bool


@dataclass(frozen=True)
class ScheduleCommand:
    offset_s: float
    magic: int
    value: float
    label: str


def env_float(name, default):
    return float(os.environ.get(name, default))


def env_int(name, default):
    return int(os.environ.get(name, default))


def parse_target(value):
    if value.isdigit():
        value = f"127.0.0.1:{value}"
    host, separator, port_text = value.rpartition(":")
    if not separator:
        host, port_text = value, "5556"
    if not host:
        raise argparse.ArgumentTypeError("a channel target requires a host")
    try:
        port = int(port_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid UDP port in {value}") from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"UDP port is outside 1..65535 in {value}")
    try:
        answers = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)
    except socket.gaierror as error:
        raise argparse.ArgumentTypeError(f"cannot resolve channel target {host}") from error
    return answers[0][4]


def parse_start_time(value):
    try:
        return float(value)
    except ValueError:
        pass
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "start time must be a Unix epoch or an ISO-8601 time with a zone"
        ) from error
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("an ISO-8601 start time requires a time zone")
    return parsed.timestamp()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Preload one orbital channel schedule into all selected cuSAT nodes"
    )
    parser.add_argument(
        "targets", nargs="*", type=parse_target, metavar="HOST[:PORT]",
        help="management endpoint for each modem; default: 127.0.0.1:5556",
    )
    parser.add_argument(
        "--regime", choices=sorted(REGIME_ALT_KM),
        default=os.environ.get("REGIME", "leo").lower(),
    )
    parser.add_argument(
        "--alt-km", type=float,
        default=float(os.environ["ALT_KM"]) if os.environ.get("ALT_KM") else None,
    )
    parser.add_argument(
        "--arc-fraction", type=float, default=env_float("ALPHA_FRAC", 0.95),
        help="fraction of the geometric horizon arc to traverse",
    )
    parser.add_argument("--passes", type=int, default=env_int("N_PASSES", 3))
    parser.add_argument(
        "--accel", type=float, default=env_float("ACCEL", 10.0),
        help="orbital speed-up when --pass-duration is not set",
    )
    parser.add_argument("--gap-sec", type=float, default=env_float("GAP_SEC", 10.0))
    parser.add_argument(
        "--pre-gap-sec", type=float, default=env_float("PRE_GAP_SEC", 0.0)
    )
    parser.add_argument(
        "--pass-duration", type=float,
        default=float(os.environ["PASS_DUR"]) if os.environ.get("PASS_DUR") else None,
        help="wall-clock duration of one visible pass",
    )
    parser.add_argument("--rate-hz", type=float, default=env_float("RATE_HZ", 5.0))
    parser.add_argument(
        "--static-elevation", type=float,
        default=float(os.environ["STATIC_ELEV_DEG"])
        if os.environ.get("STATIC_ELEV_DEG") else None,
        help="hold one elevation instead of sweeping a pass",
    )
    start = parser.add_mutually_exclusive_group()
    start.add_argument(
        "--start-at", type=parse_start_time,
        help="Unix epoch or ISO-8601 start time; default: automatic future time",
    )
    start.add_argument(
        "--start-in", type=float,
        help="seconds from now to the first point; default: automatic safe lead",
    )
    parser.add_argument(
        "--cycle", action="store_true",
        help="repeat the schedule continuously; Ctrl-C cancels future points",
    )
    parser.add_argument(
        "--upload-rate", type=float, default=1000.0,
        help="maximum schedule datagrams per second",
    )
    parser.add_argument(
        "--queue-ack-timeout-sec", type=float, default=3.0,
        help="maximum wait for schedule-upload acknowledgements",
    )
    parser.add_argument(
        "--ack-timeout-sec", type=float, default=0.5,
        help="maximum wait after each application time for ACK1",
    )
    parser.add_argument(
        "--max-lateness-ms", type=float, default=25.0,
        help="fail if a modem applies a point later than this",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the schedule without sending it",
    )
    parser.add_argument("--command-lead-sec", type=float, help=argparse.SUPPRESS)

    args = parser.parse_args(argv)
    if not 0.0 < args.arc_fraction <= 1.0:
        parser.error("--arc-fraction must be in (0, 1]")
    if args.passes < 1 or args.accel <= 0.0 or args.rate_hz <= 0.0:
        parser.error("--passes, --accel, and --rate-hz must be positive")
    if args.alt_km is not None and args.alt_km <= 0.0:
        parser.error("--alt-km must be positive")
    if args.pass_duration is not None and args.pass_duration <= 0.0:
        parser.error("--pass-duration must be positive")
    if args.gap_sec < 0.0 or args.pre_gap_sec < 0.0:
        parser.error("gap durations must not be negative")
    if args.start_in is not None and args.start_in <= 0.0:
        parser.error("--start-in must be positive")
    if args.upload_rate <= 0.0:
        parser.error("--upload-rate must be positive")
    if args.queue_ack_timeout_sec <= 0.0 or args.ack_timeout_sec <= 0.0:
        parser.error("ACK timeouts must be positive")
    if args.max_lateness_ms <= 0.0:
        parser.error("--max-lateness-ms must be positive")
    if args.static_elevation is not None and not 0.0 <= args.static_elevation <= 90.0:
        parser.error("--static-elevation must be in [0, 90]")
    if args.regime == "geo" and args.static_elevation is None:
        args.static_elevation = 90.0
    args.alt_km = args.alt_km or REGIME_ALT_KM[args.regime]
    args.targets = args.targets or [parse_target("127.0.0.1:5556")]
    if len(set(args.targets)) != len(args.targets):
        parser.error("channel targets must be unique")
    return args


def orbit_point(alt_km, alpha):
    orbital_radius = R_EARTH_KM + alt_km
    cos_alpha, sin_alpha = math.cos(alpha), math.sin(alpha)
    slant = math.sqrt(
        R_EARTH_KM ** 2 + orbital_radius ** 2
        - 2 * R_EARTH_KM * orbital_radius * cos_alpha
    )
    elevation = math.degrees(
        math.atan2(cos_alpha - R_EARTH_KM / orbital_radius, abs(sin_alpha))
    )
    relative_fspl = 20 * math.log10(slant / alt_km)
    return relative_fspl, slant, elevation


def point_at_elevation(alt_km, elevation_deg):
    orbital_radius = R_EARTH_KM + alt_km
    elevation = math.radians(elevation_deg)
    slant = -R_EARTH_KM * math.sin(elevation) + math.sqrt(
        (R_EARTH_KM * math.sin(elevation)) ** 2
        + orbital_radius ** 2 - R_EARTH_KM ** 2
    )
    return 20 * math.log10(slant / alt_km), slant, elevation_deg


def trajectory_parameters(args):
    orbital_radius = R_EARTH_KM + args.alt_km
    horizon = math.acos(R_EARTH_KM / orbital_radius)
    max_angle = horizon * args.arc_fraction
    omega = math.sqrt(MU_EARTH_KM3_S2 / orbital_radius) / orbital_radius
    real_pass = 2 * max_angle / omega
    pass_duration = args.pass_duration or real_pass / args.accel
    edge_loss, _, _ = orbit_point(args.alt_km, max_angle)
    return max_angle, real_pass, pass_duration, edge_loss


def build_trajectory(args):
    max_angle, real_pass, pass_duration, edge_loss = trajectory_parameters(args)
    interval = 1.0 / args.rate_hz

    if args.static_elevation is not None:
        loss, slant, elevation = point_at_elevation(
            args.alt_km, args.static_elevation
        )
        point = OrbitPoint(0.0, loss, slant, 2.0 * slant, elevation, True)
        return [point], pass_duration, real_pass, edge_loss

    points = []
    step = 0

    def add_outage(duration):
        nonlocal step
        for _ in range(int(round(duration * args.rate_hz))):
            points.append(OrbitPoint(
                step * interval, -1.0, math.nan, math.nan, -1.0, False
            ))
            step += 1

    add_outage(args.pre_gap_sec)
    pass_steps = max(1, int(round(pass_duration * args.rate_hz)))
    for pass_index in range(args.passes):
        for position in range(pass_steps):
            fraction = position / pass_steps
            alpha = -max_angle + 2 * max_angle * fraction
            loss, slant, elevation = orbit_point(args.alt_km, alpha)
            points.append(OrbitPoint(
                step * interval, loss, slant, 2.0 * slant, elevation, True
            ))
            step += 1
        if pass_index < args.passes - 1 or args.cycle:
            add_outage(args.gap_sec)

    if not points:
        raise ValueError("the selected parameters produce an empty schedule")
    return points, step * interval, real_pass, edge_loss


def build_commands(points):
    commands = []
    previous_link = None
    for point in points:
        if point.link_available != previous_link:
            commands.append(ScheduleCommand(
                point.offset_s, LINK_MAGIC,
                1.0 if point.link_available else 0.0,
                f"LINK {'up' if point.link_available else 'down'}",
            ))
            previous_link = point.link_available
        if point.link_available:
            commands.append(ScheduleCommand(
                point.offset_s, DIST_MAGIC, point.pathloss_distance_km,
                f"DIST {point.pathloss_distance_km:.1f} km",
            ))
    return commands


def print_schedule(args, points, period_s, real_pass_s, edge_loss_db, start_epoch):
    start_text = datetime.datetime.fromtimestamp(
        start_epoch, datetime.timezone.utc
    ).isoformat().replace("+00:00", "Z")
    print(f"# start_utc={start_text} start_epoch={start_epoch:.6f}")
    print(
        f"# regime={args.regime.upper()} altitude_km={args.alt_km:g} "
        f"period_s={period_s:.3f} real_pass_s={real_pass_s:.1f} "
        f"edge_loss_db={edge_loss_db:.2f} cycle={str(args.cycle).lower()}"
    )
    print(
        "offset_s,relative_fspl_db,slant_range_km,"
        "pathloss_distance_km,elevation_deg,link_available"
    )
    for point in points:
        print(
            f"{point.offset_s:.3f},{point.relative_fspl_db:.3f},"
            f"{point.slant_range_km:.3f},{point.pathloss_distance_km:.3f},"
            f"{point.elevation_deg:.2f},{int(point.link_available)}"
        )


class CommandSender:
    def __init__(self, targets, upload_rate, queue_ack_timeout,
                 apply_ack_timeout, max_lateness_ms):
        self.targets = tuple(targets)
        self.target_set = set(targets)
        self.upload_rate = upload_rate
        self.queue_ack_timeout = queue_ack_timeout
        self.apply_ack_timeout = apply_ack_timeout
        self.max_lateness_ns = int(max_lateness_ms * 1e6)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 0))
        self.sock.setblocking(False)
        self.sequence = 0
        self.pending = {}
        self.queue_acks = 0
        self.apply_acks = 0
        self.max_lateness_seen_ns = 0

    def _new_command(self, command, start_epoch):
        self.sequence += 1
        sequence = self.sequence
        requested_ns = int(round((start_epoch + command.offset_s) * 1e9))
        self.pending[sequence] = {
            "requested_ns": requested_ns,
            "apply_deadline": requested_ns / 1e9 + self.apply_ack_timeout,
            "expected": set(self.target_set),
            "queued": set(),
            "applied": set(),
            "label": command.label,
        }
        packet = SCHEDULED_COMMAND.pack(
            command.magic, float(command.value), requested_ns, sequence
        )
        return sequence, packet

    def _drain_acks(self):
        while True:
            try:
                packet, peer = self.sock.recvfrom(256)
            except BlockingIOError:
                return
            if len(packet) != SCHEDULED_ACK.size:
                continue
            magic, status, sequence, requested_ns, event_ns = \
                SCHEDULED_ACK.unpack(packet)
            pending = self.pending.get(sequence)
            if pending is None or peer not in pending["expected"]:
                continue
            if requested_ns != pending["requested_ns"]:
                raise RuntimeError(f"ACK timestamp mismatch for sequence {sequence}")
            if status & ACK_QUEUE_FULL:
                raise RuntimeError(
                    f"cuSAT schedule queue is full at {pending['label']}"
                )
            if status & ACK_LATE_ON_ARRIVAL:
                raise RuntimeError(
                    f"{pending['label']} reached {peer[0]} after its application time"
                )
            if magic == QUEUED_ACK_MAGIC:
                if peer not in pending["queued"]:
                    pending["queued"].add(peer)
                    self.queue_acks += 1
            elif magic == APPLIED_ACK_MAGIC:
                lateness_ns = max(0, event_ns - requested_ns)
                if lateness_ns > self.max_lateness_ns:
                    raise RuntimeError(
                        f"{pending['label']} applied {lateness_ns / 1e6:.3f} ms "
                        f"late on {peer[0]}"
                    )
                self.max_lateness_seen_ns = max(
                    self.max_lateness_seen_ns, lateness_ns
                )
                if peer not in pending["applied"]:
                    pending["applied"].add(peer)
                    self.apply_acks += 1
                if pending["applied"] == pending["expected"]:
                    del self.pending[sequence]

    def _check_apply_timeouts(self):
        now = time.time()
        for sequence, pending in self.pending.items():
            if now > pending["apply_deadline"]:
                missing = pending["expected"] - pending["applied"]
                if missing:
                    hosts = ", ".join(sorted(peer[0] for peer in missing))
                    raise RuntimeError(
                        f"no applied ACK for {pending['label']} sequence "
                        f"{sequence} from {hosts}"
                    )

    def _wait(self, seconds):
        deadline = time.monotonic() + max(0.0, seconds)
        while True:
            self._drain_acks()
            self._check_apply_timeouts()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            select.select([self.sock], [], [], min(remaining, 0.05))

    def wait_until(self, epoch):
        while True:
            self._drain_acks()
            self._check_apply_timeouts()
            remaining = epoch - time.time()
            if remaining <= 0.0:
                return
            select.select([self.sock], [], [], min(remaining, 0.05))

    def upload(self, commands, start_epoch, cycle_index):
        sequences = []
        interval = 1.0 / self.upload_rate
        next_send = time.monotonic()
        for command in commands:
            sequence, packet = self._new_command(command, start_epoch)
            sequences.append(sequence)
            for target in self.targets:
                remaining = next_send - time.monotonic()
                if remaining > 0.0:
                    self._wait(remaining)
                self.sock.sendto(packet, target)
                next_send = max(next_send + interval, time.monotonic())
            self._drain_acks()

        queue_deadline = time.monotonic() + self.queue_ack_timeout
        while True:
            self._drain_acks()
            missing = [
                sequence for sequence in sequences
                if sequence in self.pending
                and self.pending[sequence]["queued"]
                != self.pending[sequence]["expected"]
            ]
            if not missing:
                break
            if time.monotonic() >= queue_deadline:
                raise RuntimeError(
                    f"missing queued ACKs for {len(missing)} schedule commands"
                )
            select.select([self.sock], [], [], 0.05)

        print(
            f"# uploaded cycle={cycle_index} commands={len(commands)} "
            f"targets={len(self.targets)} start_epoch={start_epoch:.6f}",
            file=sys.stderr, flush=True,
        )

    def finish(self, final_epoch):
        self.wait_until(final_epoch + self.apply_ack_timeout)
        self._drain_acks()
        self._check_apply_timeouts()
        if self.pending:
            raise RuntimeError(f"missing ACKs for {len(self.pending)} commands")
        print(
            f"# queue_ACKs={self.queue_acks} apply_ACKs={self.apply_acks} "
            f"max_apply_lateness_ms={self.max_lateness_seen_ns / 1e6:.3f}",
            file=sys.stderr, flush=True,
        )

    def clear_future(self):
        packet = LEGACY_COMMAND.pack(CLEAR_MAGIC, 0.0)
        for _ in range(3):
            for target in self.targets:
                self.sock.sendto(packet, target)
            time.sleep(0.02)
        self.pending.clear()

    def close(self):
        self.sock.close()


def choose_start(args, command_count):
    upload_seconds = command_count * len(args.targets) / args.upload_rate
    minimum_lead = upload_seconds + args.queue_ack_timeout_sec + 0.5
    now = time.time()
    if args.start_at is not None:
        start_epoch = args.start_at
    elif args.start_in is not None:
        start_epoch = now + args.start_in
    else:
        start_epoch = now + max(5.0, minimum_lead)
    if not args.dry_run and start_epoch < now + minimum_lead:
        raise ValueError(
            f"the start time needs at least {minimum_lead:.2f} s of upload lead"
        )
    return start_epoch, upload_seconds


def main(argv=None):
    args = parse_args(argv)
    points, period_s, real_pass_s, edge_loss_db = build_trajectory(args)
    commands = build_commands(points)
    if len(commands) > MAX_SCHEDULE_COMMANDS:
        raise ValueError(
            f"schedule has {len(commands)} commands; maximum is "
            f"{MAX_SCHEDULE_COMMANDS}"
        )
    if args.cycle and 2 * len(commands) > MAX_SCHEDULE_COMMANDS:
        raise ValueError(
            "a cyclic schedule must leave queue space for the next cycle"
        )
    start_epoch, upload_seconds = choose_start(args, len(commands))
    if args.cycle and period_s <= upload_seconds + args.queue_ack_timeout_sec:
        raise ValueError("one cycle is too short to preload the next cycle safely")
    print_schedule(
        args, points, period_s, real_pass_s, edge_loss_db, start_epoch
    )
    if args.dry_run:
        return 0

    target_text = ", ".join(f"{host}:{port}" for host, port in args.targets)
    print(f"# management_targets={target_text}", file=sys.stderr, flush=True)
    sender = CommandSender(
        args.targets, args.upload_rate, args.queue_ack_timeout_sec,
        args.ack_timeout_sec, args.max_lateness_ms,
    )
    success = False
    try:
        sender.upload(commands, start_epoch, 0)
        if not args.cycle:
            sender.finish(start_epoch + points[-1].offset_s)
            success = True
            print("# schedule complete; final channel state retained", file=sys.stderr)
            # Retain the historical completion marker used by experiment
            # wrappers that poll the driver's combined output.
            print("# done", file=sys.stderr, flush=True)
            return 0

        cycle_index = 1
        while True:
            sender.wait_until(start_epoch + (cycle_index - 1) * period_s)
            sender.upload(
                commands, start_epoch + cycle_index * period_s, cycle_index
            )
            cycle_index += 1
    except KeyboardInterrupt:
        sender.clear_future()
        success = True
        print("\n# cycle stopped; future points cleared", file=sys.stderr)
        return 130
    finally:
        if not success:
            sender.clear_future()
        sender.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
