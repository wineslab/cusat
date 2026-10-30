#!/usr/bin/env python3
"""Cellular-demo UE agent. Runs as root on Spark .11 (systemd unit cusat-ue-agent).

Writes /run/cusat-ue.json once a second for the dashboard:
  modem   Sierra state from AT!GSTATUS? (registration, RRC, RSRP, SINR, TX power)
  ping    RTT from the UE (netns "ue") to the core's UE gateway
  wwan    wwan0 byte counters inside netns "ue" (traffic the UE actually carried)
  iperf   UDP load through the UE: downlink (-R) and uplink, restarted if they stop
The UE interface lives in netns "ue" because Spark .11 runs a local UPF on the same
10.45.0.0/16 pool; see sierra-ue.sh.
"""
import json, os, re, socket, subprocess, threading, time
import serial

OUT = '/run/cusat-ue.json'
GATEWAY = '10.45.0.1'
NETNS = ['ip', 'netns', 'exec', 'ue']
RATES = {'dl': os.environ.get('UE_DL', '10M'), 'ul': os.environ.get('UE_UL', '2M')}
PORTS = {'dl': 5203, 'ul': 5202}
LOCK = threading.Lock()
STATE = {'modem': {}, 'ping': {'rtt_ms': None, 'time': 0.0, 'recent': []},
         'wwan': {}, 'iperf': {'dl': {}, 'ul': {}}, 'rates': RATES}
TRAFFIC = {'enabled': os.environ.get('UE_CONTROLLED') != '1', 'generation': None,
           'revision': 0, 'heartbeat': time.monotonic(), 'reason': 'Waiting for feeder'}
PROCESSES = {}


def traffic_control(enabled, generation, rates=None, reason=''):
    """Cancel old flows on a gap, pass change, or offered-load change."""
    with LOCK:
        changed = (enabled != TRAFFIC['enabled'] or generation != TRAFFIC['generation']
                   or (rates is not None and rates != RATES))
        TRAFFIC.update(heartbeat=time.monotonic(), reason=reason)
        if not changed:
            return
        if rates is not None:
            RATES.update(rates)
        TRAFFIC.update(enabled=enabled, generation=generation, revision=TRAFFIC['revision'] + 1)
        processes = list(PROCESSES.values())
        for direction in PORTS:
            STATE['iperf'][direction] = {'mbps': 0.0, 'running': False, 'paused': not enabled,
                                         'time': time.time()}
    for process in processes:
        if process.poll() is None:
            process.terminate()
            # A broken control connection must not keep a UDP sender alive.
            def reap(p=process):
                try:
                    p.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    p.kill()
            threading.Thread(target=reap, daemon=True).start()


def control_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 5560)); sock.settimeout(.25)
    while True:
        try:
            packet, source = sock.recvfrom(2048)
            if source[0] != '192.168.50.10':
                continue
            control = json.loads(packet)
            enabled = control['enabled']
            rates = control['rates']
            if type(enabled) is not bool or set(rates) != set(PORTS):
                continue
            if not all(re.fullmatch(r'[1-9][0-9]*(?:\.[0-9]+)?[KMG]?', str(v)) for v in rates.values()):
                continue
            traffic_control(enabled, control['generation'], rates, control.get('reason', ''))
        except (ValueError, KeyError, TypeError):
            continue
        except socket.timeout:
            with LOCK:
                stale = time.monotonic() - TRAFFIC['heartbeat'] > 3
                generation = TRAFFIC['generation']
            if stale and os.environ.get('UE_CONTROLLED') == '1':
                traffic_control(False, generation, reason='Controller disconnected')


def at(commands):
    port = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
    replies = {}
    try:
        for command in commands:
            port.reset_input_buffer()
            port.write((command + '\r\n').encode())
            time.sleep(0.6)
            data = b''
            while port.in_waiting:
                data += port.read(port.in_waiting)
                time.sleep(0.05)
            replies[command] = data.decode(errors='replace')
    finally:
        port.close()
    return replies


def field(text, name, number=False):
    match = re.search(re.escape(name) + r'\s*:?\s*([^\t\r\n|]+?)(?:\s{2,}|\t|\r|\n|$)', text)
    if not match:
        return None
    value = match.group(1).strip()
    if number:
        found = re.search(r'-?\d+(?:\.\d+)?', value)
        return float(found.group(0)) if found else None
    return value


def modem_loop():
    while True:
        try:
            reply = at(['AT!GSTATUS?', 'AT+CFUN?'])
            text = reply['AT!GSTATUS?']
            modem = {
                'mode': field(text, 'System mode'),
                'mm': field(text, 'MM5G state'),
                'rrc': field(text, 'RRC state'),
                'band': field(text, 'NR5G band'),
                'bw': field(text, 'NR5G dl bw'),
                'tx_power_dbm': field(text, 'NR5G Tx Power', True),
                'rsrp_dbm': field(text, 'NR5G RSRP (dBm)', True),
                'rsrq_db': field(text, 'NR5G RSRQ (dB)', True),
                'sinr_db': field(text, 'NR5G SINR (dB)', True),
                'cell': field(text, 'NR5G Cell ID'),
                'cfun': (re.search(r'\+CFUN:\s*(\d)', reply['AT+CFUN?']) or [None, None])[1],
                'time': time.time(),
            }
            with LOCK:
                STATE['modem'] = modem
        except Exception as error:  # modem re-enumerating, port busy
            with LOCK:
                STATE['modem'] = {'error': str(error)[-200:], 'time': time.time()}
        time.sleep(2)


def ping_loop():
    while True:
        try:
            process = subprocess.Popen(NETNS + ['ping', '-n', '-O', '-i', '0.5', '-W', '2', GATEWAY],
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            sent = {}  # icmp_seq -> [send time, RTT or None]; a late reply still counts as delivered
            for line in process.stdout:
                seq = re.search(r'icmp_seq=(\d+)', line)
                rtt = re.search(r'time=([0-9.]+) ms', line)
                if not seq or not (rtt or 'no answer yet' in line):
                    continue
                now, n = time.time(), int(seq.group(1))
                if rtt:
                    sent[n] = [now - float(rtt.group(1)) / 1000, float(rtt.group(1))]
                elif n not in sent:
                    sent[n] = [now - 0.5, None]  # outstanding when the next ping went out
                for old_seq in [k for k in sent if k < n - 240]:
                    del sent[old_seq]
                with LOCK:
                    # [send time, RTT or None] per ping; the dashboard windows it per pass.
                    ping = STATE['ping']
                    ping['recent'] = [sent[k] for k in sorted(sent)]
                    if rtt:
                        ping['rtt_ms'] = float(rtt.group(1))
                        ping['time'] = now
        except OSError:
            pass
        time.sleep(2)


def iperf_loop(direction):
    # UDP so the 5G and feeder links set the delivered rate; the load restarts after
    # an outage drops the control connection.
    while True:
        with LOCK:
            enabled, revision, rate = TRAFFIC['enabled'], TRAFFIC['revision'], RATES[direction]
        if not enabled:
            time.sleep(.1)
            continue
        command = NETNS + ['iperf3', '-c', GATEWAY, '-p', str(PORTS[direction]), '-u', '-b', rate,
                           '-l', '1200', '-t', '86400', '-i', '1', '--forceflush',
                           '--connect-timeout', '2000', '-w', '128K'] + (
                           ['-R', '--rcv-timeout', '3000'] if direction == 'dl' else [])
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            with LOCK:
                PROCESSES[direction] = process
                if not TRAFFIC['enabled'] or revision != TRAFFIC['revision']:
                    process.terminate()
            for line in process.stdout:
                if 'error' in line.lower():
                    with LOCK:
                        STATE.setdefault('iperf_errors', {})[direction] = line.strip()[-300:]
                    print(direction + ': ' + line.strip(), flush=True)
                match = re.search(r'([0-9.]+) ([KMG]?)bits/sec(?:\s+([0-9.]+) ms\s+(\d+)/(\d+))?', line)
                if not match or 'sender' in line or 'receiver' in line:
                    continue
                scale = {'': 1e-6, 'K': 1e-3, 'M': 1.0, 'G': 1e3}[match.group(2)]
                sample = {'mbps': float(match.group(1)) * scale, 'time': time.time(), 'running': True}
                if match.group(3):
                    lost, total = int(match.group(4)), int(match.group(5))
                    sample.update(jitter_ms=float(match.group(3)), loss=lost / total if total else None)
                with LOCK:
                    if TRAFFIC['enabled'] and revision == TRAFFIC['revision']:
                        STATE['iperf'][direction] = sample
            process.wait()
        except OSError:
            pass
        with LOCK:
            PROCESSES.pop(direction, None)
            if revision == TRAFFIC['revision']:
                STATE['iperf'][direction] = {'mbps': 0.0, 'running': False, 'time': time.time()}
        time.sleep(.25)


def counters():
    try:
        out = subprocess.run(NETNS + ['cat', '/sys/class/net/wwan0/statistics/rx_bytes',
                                      '/sys/class/net/wwan0/statistics/tx_bytes'],
                             capture_output=True, text=True, timeout=3).stdout.split()
        address = subprocess.run(NETNS + ['ip', '-4', '-br', 'addr', 'show', 'wwan0'],
                                 capture_output=True, text=True, timeout=3).stdout.split()
        return {'rx_bytes': int(out[0]), 'tx_bytes': int(out[1]),
                'ip': next((a.split('/')[0] for a in address if a.count('.') == 3), None)}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {'rx_bytes': None, 'tx_bytes': None, 'ip': None}


def main():
    for target, args in ((control_loop, ()), (modem_loop, ()), (ping_loop, ()), (iperf_loop, ('dl',)), (iperf_loop, ('ul',))):
        threading.Thread(target=target, args=args, daemon=True).start()
    while True:
        wwan = counters()
        with LOCK:
            STATE['wwan'] = dict(wwan, time=time.time())
            STATE['traffic'] = dict(TRAFFIC)
            payload = json.dumps(STATE)
        temporary = OUT + '.tmp'
        with open(temporary, 'w') as handle:
            handle.write(payload)
        os.chmod(temporary, 0o644)
        os.replace(temporary, OUT)
        time.sleep(1)


if __name__ == '__main__':
    main()
