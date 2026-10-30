#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sidecar that bridges dvbs2-uhd-trx IQ/KPI data to HTTP.

Reads binary spectrum frames from a named pipe written by spectrum_tap (C++),
and serves them over HTTP. Also accepts noise commands via POST and forwards
them to the transceiver's channel emulator (--noise) via localhost UDP.

Run alongside dvbs2-uhd-trx on each signal pod:
    python3 sidecar.py [--port 8080] [--pipe /tmp/dvbs2_spectrum] [--noise-port 5556]
"""

import argparse
import logging
import os
import socket
import struct
import threading
import time

from flask import Flask, Response, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sidecar")

# --- Shared state ------------------------------------------------------------

latest_frame: bytes = b""
frame_lock = threading.Lock()

NOISE_MAGIC = 0x4E4F4953  # 'NOIS'

app = Flask(__name__)


# --- FIFO reader thread ------------------------------------------------------

def fifo_reader(pipe_path: str, frame_size: int):
    """Continuously reads fixed-size binary frames from the named pipe."""
    global latest_frame

    # Create FIFO if it doesn't exist
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
        log.info("Created FIFO %s", pipe_path)

    while True:
        try:
            # open blocks until C++ opens the write end
            with open(pipe_path, "rb") as f:
                log.info("FIFO writer connected")
                buf = b""
                while True:
                    chunk = f.read(frame_size - len(buf))
                    if not chunk:
                        log.warning("FIFO writer disconnected, reopening...")
                        break
                    buf += chunk
                    if len(buf) == frame_size:
                        with frame_lock:
                            latest_frame = buf
                        buf = b""
        except Exception as e:
            log.error("FIFO read error: %s", e)
            time.sleep(1)


# --- HTTP routes -------------------------------------------------------------

@app.route("/spectrum")
def get_spectrum():
    """Return latest binary spectrum frame (20B header + 8192B IQ)."""
    with frame_lock:
        data = latest_frame
    if not data:
        return Response(status=204)  # No content yet
    return Response(data, mimetype="application/octet-stream")


@app.route("/noise", methods=["POST"])
@app.route("/channel", methods=["POST"])
def post_channel():
    """Forward channel impairment command to C++ via localhost UDP.

    Accepts either {"snr_db": X} (legacy noise) or {"pathloss_db": X}.
    Both map to the same UDP packet — the C++ side interprets based on its mode.
    """
    try:
        body = request.get_json(force=True)
        val = float(body.get("pathloss_db", body.get("snr_db", 0.0)))
    except (TypeError, ValueError, AttributeError):
        return jsonify(error="bad request"), 400

    pkt = struct.pack("<If", NOISE_MAGIC, val)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(pkt, ("127.0.0.1", app.config["NOISE_PORT"]))
        sock.close()
    except OSError as e:
        log.warning("Failed to send channel UDP: %s", e)
        return jsonify(error=str(e)), 500

    return jsonify(ok=True)


@app.route("/health")
def health():
    with frame_lock:
        has_data = len(latest_frame) > 0
    return jsonify(healthy=True, has_data=has_data)


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Signal pod sidecar")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default 8080)")
    parser.add_argument("--pipe", default="/tmp/dvbs2_spectrum",
                        help="Named pipe path (default /tmp/dvbs2_spectrum)")
    parser.add_argument("--noise-port", type=int, default=5556,
                        help="Localhost UDP port for noise commands (default 5556)")
    args = parser.parse_args()

    # spectrum_tap header (32) + 1024 complex floats (8192) = 8224
    FRAME_SIZE = 32 + 1024 * 2 * 4

    app.config["NOISE_PORT"] = args.noise_port

    # Start FIFO reader in background thread
    reader = threading.Thread(target=fifo_reader, args=(args.pipe, FRAME_SIZE), daemon=True)
    reader.start()
    log.info("Sidecar starting on :%d, pipe=%s, noise_port=%d",
             args.port, args.pipe, args.noise_port)

    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
