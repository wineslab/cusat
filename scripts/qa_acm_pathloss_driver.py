#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import datetime
import heapq
import importlib.util
import pathlib
import select
import socket
import threading
import time
import unittest


DRIVER_PATH = pathlib.Path(__file__).with_name("acm-pathloss-driver.py")
SPEC = importlib.util.spec_from_file_location("acm_pathloss_driver", DRIVER_PATH)
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class FakeModem:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.target = self.sock.getsockname()
        self.stop_event = threading.Event()
        self.scheduled = []
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop_event.is_set():
            now_ns = time.time_ns()
            while self.scheduled and self.scheduled[0][0] <= now_ns:
                requested_ns, sequence, peer = heapq.heappop(self.scheduled)
                ack = DRIVER.SCHEDULED_ACK.pack(
                    DRIVER.APPLIED_ACK_MAGIC, 0, sequence,
                    requested_ns, time.time_ns(),
                )
                self.sock.sendto(ack, peer)
            timeout = 0.01
            if self.scheduled:
                timeout = min(
                    timeout, max(0.0, (self.scheduled[0][0] - now_ns) / 1e9)
                )
            readable, _, _ = select.select([self.sock], [], [], timeout)
            if not readable:
                continue
            packet, peer = self.sock.recvfrom(256)
            if len(packet) == DRIVER.SCHEDULED_COMMAND.size:
                _magic, _value, requested_ns, sequence = \
                    DRIVER.SCHEDULED_COMMAND.unpack(packet)
                heapq.heappush(self.scheduled, (requested_ns, sequence, peer))
                ack = DRIVER.SCHEDULED_ACK.pack(
                    DRIVER.QUEUED_ACK_MAGIC, 0, sequence,
                    requested_ns, time.time_ns(),
                )
                self.sock.sendto(ack, peer)
            elif len(packet) == DRIVER.LEGACY_COMMAND.size:
                magic, _value = DRIVER.LEGACY_COMMAND.unpack(packet)
                if magic == DRIVER.CLEAR_MAGIC:
                    self.scheduled.clear()

    def close(self):
        self.stop_event.set()
        self.sock.sendto(b"x", self.target)
        self.thread.join(timeout=1.0)
        self.sock.close()


class ScheduleTest(unittest.TestCase):
    def args(self, *extra):
        return DRIVER.parse_args([
            "--regime", "leo",
            "--passes", "1",
            "--pass-duration", "4",
            "--rate-hz", "2",
            "--arc-fraction", "0.4",
            "--gap-sec", "1",
            "--dry-run",
            *extra,
            "127.0.0.1:5556",
        ])

    def test_finite_schedule_retains_last_visible_point(self):
        points, period, _real_pass, _edge_loss = DRIVER.build_trajectory(
            self.args()
        )
        commands = DRIVER.build_commands(points)

        self.assertEqual(len(points), 8)
        self.assertEqual(period, 4.0)
        self.assertTrue(points[-1].link_available)
        self.assertEqual(commands[0].magic, DRIVER.LINK_MAGIC)
        self.assertEqual(commands[-1].magic, DRIVER.DIST_MAGIC)

    def test_cyclic_schedule_includes_last_to_first_outage(self):
        points, period, _real_pass, _edge_loss = DRIVER.build_trajectory(
            self.args("--cycle")
        )
        commands = DRIVER.build_commands(points)

        self.assertEqual(len(points), 10)
        self.assertEqual(period, 5.0)
        self.assertFalse(points[-1].link_available)
        self.assertEqual(commands[-1].magic, DRIVER.LINK_MAGIC)
        self.assertEqual(commands[-1].value, 0.0)

    def test_iso_start_time_requires_and_uses_a_zone(self):
        parsed = DRIVER.parse_start_time("2026-09-02T20:00:00Z")
        expected = datetime.datetime(
            2026, 9, 2, 20, 0, tzinfo=datetime.timezone.utc
        ).timestamp()
        self.assertEqual(parsed, expected)
        with self.assertRaises(Exception):
            DRIVER.parse_start_time("2026-09-02T20:00:00")

    def test_one_controller_preloads_two_modems(self):
        modems = [FakeModem(), FakeModem()]
        sender = None
        try:
            sender = DRIVER.CommandSender(
                [modem.target for modem in modems],
                upload_rate=10000.0,
                queue_ack_timeout=1.0,
                apply_ack_timeout=0.5,
                max_lateness_ms=100.0,
            )
            commands = [
                DRIVER.ScheduleCommand(0.0, DRIVER.LINK_MAGIC, 1.0, "LINK up"),
                DRIVER.ScheduleCommand(0.05, DRIVER.DIST_MAGIC, 1100.0, "DIST"),
            ]
            start_epoch = time.time() + 0.1
            sender.upload(commands, start_epoch, 0)
            sender.finish(start_epoch + 0.05)
            self.assertEqual(sender.queue_acks, 4)
            self.assertEqual(sender.apply_acks, 4)
            self.assertFalse(sender.pending)
        finally:
            if sender is not None:
                sender.close()
            for modem in modems:
                modem.close()


if __name__ == "__main__":
    unittest.main()
