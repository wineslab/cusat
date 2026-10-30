"""Cellular demo for dashboard-server.py: a Sierra UE over OAI and a LEO DVB-S2 feeder.

Path: Sierra EM9293 --5G NR n77 (cabled)--> OAI gNB (Spark .11, X410 .18 ch1)
      --DVB-S2 feeder through a 550-km LEO satellite (X410 ch0 <-> ch0)--> Open5GS (Spark .12).
The feeder modems run ACM on their own. This module owns the satellite pass: it sends
the bent-pipe path distance (DIST) and visibility (LINK) to both modems' channel
emulators, adds the slant-range delay with netem on the feeder TAPs, and gathers the
telemetry the cellular page shows. The 5G link itself is a cable; only the feeder is
satellite.
"""
import json, math, pathlib, re, shlex, socket, struct, subprocess, threading, time

BASE = pathlib.Path(__file__).resolve().parent.parent
HOSTS = ('11', '12')
EARTH_KM, ALT_KM, LIGHT_KM_S = 6371.0, 550.0, 299792.458
REF_PATH_KM = 2 * ALT_KM            # configs/oai-spark1x.yaml distance_km: zenith bent pipe
MASK_DEG = 20.0                     # gateway elevation mask; passes start ~7 dB below zenith,
                                    # where the feeder reacquires promptly after the gap
PASS_S, GAP_S = 60.0, 6.0           # accelerated pass (~10x real time) and out-of-view gap
DIST_MAGIC, LINK_MAGIC = 0x44495354, 0x4C494E4B
TAPS = {'11': 'dvbs2a0', '12': 'dvbs2b0'}
BACKHAUL = 'cusat-backhaul'
RATES = {'dl': '27M', 'ul': '2.5M'}   # offered UE load; congest only the DL near pass edges
                                    # carries ~40 Mb/s DL and ~12 Mb/s UL over UDP
AGENT = '/home/wineslab/cusat-ue-agent.py'


def mask_angle():
    # Earth-central angle where the satellite crosses the elevation mask.
    e = math.radians(MASK_DEG)
    return math.acos(EARTH_KM * math.cos(e) / (EARTH_KM + ALT_KM)) - e


def geometry(angle):
    r = EARTH_KM + ALT_KM
    slant = math.sqrt(EARTH_KM ** 2 + r ** 2 - 2 * EARTH_KM * r * math.cos(angle))
    elevation = math.degrees(math.atan2(math.cos(angle) - EARTH_KM / r, abs(math.sin(angle)) or 1e-9))
    return slant, elevation


class Cellular:
    def __init__(self, transport, active, lock):
        self.transport, self.active, self.lock = transport, active, lock
        self.orbit = {'animate': True, 'phase': 'pass', 't': PASS_S / 2, 'angle': 0.0, 'visible': True,
                      'pass_s': PASS_S, 'gap_s': GAP_S, 'mask_deg': MASK_DEG, 'alt_km': ALT_KM,
                      'max_angle': mask_angle(), 'pass_id': 0, 'pass_started': time.time()}
        # Per-pass statistics restart at each pass: counter baselines per direction.
        self.bler_base = {}
        self.feeder = {n: {} for n in HOSTS}
        self.gnb, self.ue, self.rates, self.counters = {}, {}, {}, {}
        self.status = {'setup': 'idle', 'error': None, 'last_restart': 0.0}
        self.outages = []          # [start, end] wall-clock seconds of feeder LINK down
        self.started = False
        self.hold = False          # restart in progress: hold the satellite at zenith, link up
        self.traffic = {'enabled': False, 'reason': 'Waiting for feeder', 'generation': None}
        self.traffic_epoch = time.time_ns()

    # ---------------------------------------------------------------- helpers
    def ssh(self, n, command, timeout=15):
        return subprocess.run(['ssh', '-S', str(self.transport.SOCKETS[n]), '-o', 'BatchMode=yes',
                               'wineslab@192.168.50.' + n, command],
                              capture_output=True, text=True, timeout=timeout)

    def root(self, n, script, timeout=60):
        # Host-namespace root shell, as the launcher does (no sudo password).
        return self.ssh(n, 'docker run --rm --pid host --network host --privileged --entrypoint nsenter '
                        'cusat:local -t 1 -m -u -i -n -p bash -c ' + shlex.quote(script), timeout)

    # ------------------------------------------------------------ setup/stop
    def setup(self, log=print):
        """Run after launcher.py --mode oai: traffic, UE session and agent."""
        with self.lock:
            # A new run must not inherit an open gap from a stopped pass.
            self.outages.clear()
        self.status.update(setup='starting', error=None)
        self.hold = True
        try:
            self._setup(log)
        finally:
            self.hold = False

    def _setup(self, log):
        for n in HOSTS:  # the launcher's backhaul iperf would compete with the UE traffic
            self.ssh(n, 'tmux kill-window -t cusat:iperf-client 2>/dev/null; '
                        'docker exec ' + BACKHAUL + ' pkill -f "iperf3 -c 10.200.0" ; true')
        self.ssh('12', 'docker exec ' + BACKHAUL + ' pkill -f "iperf3 -s -p 520" ; for p in 5202 5203; do '
                 'docker exec -d ' + BACKHAUL + ' iperf3 -s -p $p -B 10.45.0.1; done')
        log('UE iperf servers started on the core (10.45.0.1:5202/5203)')
        self.restart_ue(log)
        self.status['setup'] = 'running'

    def restart_ue(self, log=print):
        previous_hold = self.hold
        self.hold = True
        self.status['last_restart'] = time.time()
        try:
            # Stop the serial reader before changing CFUN or replacing MBIM.
            self.root('11', 'systemctl stop cusat-ue-agent 2>/dev/null || true')
            out = subprocess.run([str(BASE / 'sierra-ue.sh'), 'start'], capture_output=True, text=True, timeout=240)
            address = re.search(r'10\.45\.\d+\.\d+/\d+', out.stdout)
            if out.returncode or not address:
                raise RuntimeError('UE session not established: ' + (out.stdout + out.stderr)[-300:])
            log('UE session ' + address.group(0))
            # A radio outage can leave the old iperf control sockets half open.
            self.ssh('12', 'docker exec ' + BACKHAUL + ' pkill -f "^iperf3 -s -p 520[23]" || true; '
                     'for p in 5202 5203; do docker exec -d ' + BACKHAUL + ' iperf3 -s -p $p -B 10.45.0.1; done')
            subprocess.run(['scp', '-q', '-o', 'ControlPath=' + str(self.transport.SOCKETS['11']),
                            str(BASE / 'cellular' / 'ue-agent.py'), 'wineslab@192.168.50.11:' + AGENT], check=True)
            self.root('11', 'systemctl reset-failed cusat-ue-agent 2>/dev/null; '
                      'systemd-run --unit=cusat-ue-agent --description="cuSAT UE agent" '
                      f'-E UE_DL={RATES["dl"]} -E UE_UL={RATES["ul"]} -E UE_CONTROLLED=1 /usr/bin/python3 {AGENT}')
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                try:
                    data = json.loads(self.ssh('11', 'cat /run/cusat-ue.json').stdout)
                    flows = data.get('iperf') or {}
                    if all((flows.get(d) or {}).get('mbps', 0) > .1 and
                           time.time() - flows[d].get('time', 0) < 5 for d in ('dl', 'ul')):
                        log('UE traffic established at zenith (DL ' + RATES['dl'] + ', UL ' + RATES['ul'] + ')')
                        break
                except (ValueError, KeyError, subprocess.SubprocessError):
                    pass
                time.sleep(1)
            else:
                raise RuntimeError('UE attached, but bidirectional iperf did not establish at zenith')
            with self.lock:
                self.orbit.update(phase='pass', t=PASS_S / 2, pass_started=time.time() - PASS_S / 2)
        finally:
            self.hold = previous_hold

    def stop(self):
        with self.lock:
            self.status['setup'] = 'idle'
            self.close_outages()
        self.root('11', 'systemctl stop cusat-ue-agent 2>/dev/null; true')
        subprocess.run([str(BASE / 'sierra-ue.sh'), 'stop'], capture_output=True, text=True, timeout=120)

    # ----------------------------------------------------------------- loops
    def close_outages(self):
        """End unfinished gaps when a pass is interrupted; caller holds lock."""
        now = time.time()
        for interval in self.outages:
            if interval[1] is None:
                interval[1] = now

    def start(self):
        if self.started:
            return
        self.started = True
        loops = [self.orbit_loop, self.delay_loop, self.traffic_loop, self.counter_loop, self.watchdog_loop,
                 lambda: self.log_loop('11', 'cusat-oai', self.parse_gnb)]
        loops += [(lambda n=n: self.log_loop(n, BACKHAUL, lambda line, n=n: self.parse_feeder(n, line))) for n in HOSTS]
        for loop in loops:
            threading.Thread(target=loop, daemon=True).start()

    def orbit_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        last, sent_link = time.monotonic(), None
        while True:
            time.sleep(0.2)
            now = time.monotonic(); dt = min(now - last, 0.5); last = now
            if not self.active() or self.status['setup'] == 'idle':
                sent_link = None
                continue
            with self.lock:
                o = self.orbit
                if self.hold:  # feeder modems acquire at zenith while the stack restarts
                    self.close_outages()
                    o.update(phase='pass', t=o['pass_s'] / 2)
                if o['animate'] and not self.hold:
                    o['t'] += dt
                    if o['phase'] == 'pass' and o['t'] >= o['pass_s']:
                        o.update(phase='gap', t=0.0); self.outages.append([time.time(), None])
                    elif o['phase'] == 'gap' and o['t'] >= o['gap_s']:
                        o.update(phase='pass', t=0.0, pass_id=o['pass_id'] + 1, pass_started=time.time())
                        if self.outages and self.outages[-1][1] is None:
                            self.outages[-1][1] = time.time()
                    self.outages = [x for x in self.outages if x[1] is None or time.time() - x[1] < 300]
                if o['phase'] == 'pass':
                    o['angle'] = -o['max_angle'] + 2 * o['max_angle'] * min(1.0, o['t'] / o['pass_s'])
                slant, elevation = geometry(o['angle'])
                visible = o['phase'] == 'pass'
                o.update(visible=visible, slant_km=slant, elevation_deg=elevation, path_km=2 * slant,
                         one_way_ms=2 * slant / LIGHT_KM_S * 1000,
                         loss_db=20 * math.log10(2 * slant / REF_PATH_KM),
                         next_pass_s=(o['gap_s'] - o['t']) if not visible else None)
                link, path = visible, 2 * slant
            for n in HOSTS:
                try:
                    sock.sendto(struct.pack('<If', DIST_MAGIC, float(path)), ('192.168.50.' + n, 5556))
                    sock.sendto(struct.pack('<If', LINK_MAGIC, 1.0 if link else 0.0), ('192.168.50.' + n, 5556))
                except OSError:
                    pass
            sent_link = link

    def traffic_loop(self):
        """Fresh iperf sessions per visible pass; no offered traffic during gaps."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        servers = None
        gap_flushed = False
        while True:
            time.sleep(.2)
            with self.lock:
                running = self.active() or self.hold
                visible = self.hold or self.orbit['phase'] == 'pass'
                generation = f'{self.traffic_epoch}:{self.orbit["pass_id"]}'
                ready = self.hold or all(f.get('locked') and time.time() - f.get('time', 0) < 4
                                        for f in self.feeder.values())
            if not running:
                servers = None
                continue
            # Reacquire first; once a pass is running, let congestion remain visible.
            enable = visible and (ready or servers is True)
            reason = 'Pass in view' if enable else 'Out of view' if not visible else 'Feeder reacquiring'
            packet = {'enabled': enable and servers is True, 'generation': generation,
                      'rates': RATES, 'reason': reason}
            sock.sendto(json.dumps(packet).encode(), ('192.168.50.11', 5560))
            try:
                if not visible and not gap_flushed:
                    # Kill the reverse sender too, and discard old packets in netem.
                    out = self.ssh('12', 'docker exec ' + BACKHAUL +
                                   ' pkill -f "^iperf3 -s -p 520[23]" || true')
                    if out.returncode:
                        raise RuntimeError(out.stderr[-200:])
                    servers = False
                    with self.lock:
                        extra = max(0.0, (self.orbit.get('path_km', REF_PATH_KM) - REF_PATH_KM)
                                    / LIGHT_KM_S * 1000)
                    delay = round(extra * 4) / 4
                    for n in HOSTS:
                        out = self.ssh(n, f'docker exec {BACKHAUL} tc qdisc del dev {TAPS[n]} root 2>/dev/null; '
                                          f'docker exec {BACKHAUL} tc qdisc add dev {TAPS[n]} root netem delay {delay:.2f}ms limit 1000')
                        if out.returncode:
                            raise RuntimeError(out.stderr[-200:])
                    gap_flushed = True
                elif enable and servers is not True:
                    out = self.ssh('12', 'docker exec ' + BACKHAUL +
                                   ' pkill -f "^iperf3 -s -p 520[23]" || true; '
                                   'for p in 5202 5203; do docker exec -d ' + BACKHAUL +
                                   ' iperf3 -s -p $p -B 10.45.0.1 --idle-timeout 10 --rcv-timeout 3000; done')
                    if out.returncode:
                        raise RuntimeError(out.stderr[-200:])
                    servers = True
                    gap_flushed = False
                with self.lock:
                    self.traffic.update(enabled=enable and servers is True, reason=reason, generation=generation)
            except (OSError, subprocess.SubprocessError, RuntimeError) as error:
                with self.lock:
                    self.traffic.update(enabled=False, reason=str(error)[-200:])

    def delay_loop(self):
        # The modem's IQ delay is fixed at the 1,100-km zenith path; netem adds the rest.
        applied = {}
        while True:
            time.sleep(0.5)
            if not self.active():
                applied.clear(); continue
            with self.lock:
                extra = max(0.0, (self.orbit.get('path_km', REF_PATH_KM) - REF_PATH_KM) / LIGHT_KM_S * 1000)
            target = round(extra * 4) / 4
            for n in HOSTS:
                if applied.get(n) == target:
                    continue
                try:
                    result = self.ssh(n, f'docker exec {BACKHAUL} tc qdisc replace dev {TAPS[n]} root netem '
                                         f'delay {target:.2f}ms limit 1000', timeout=10)
                    if result.returncode == 0:
                        applied[n] = target
                except subprocess.SubprocessError:
                    pass

    def log_loop(self, n, container, parse):
        while True:
            if not self.active():
                time.sleep(1); continue
            try:
                process = subprocess.Popen(['ssh', '-S', str(self.transport.SOCKETS[n]), '-o', 'BatchMode=yes',
                                            '-o', 'ServerAliveInterval=5', 'wineslab@192.168.50.' + n,
                                            f'docker logs -f --since 2s {container} 2>&1'],
                                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, errors='replace')
                for line in process.stdout:
                    parse(line)
                    if not self.active():
                        process.kill(); break
            except OSError:
                pass
            time.sleep(2)

    FEEDER = re.compile(r'Lock=(\w+); pilotSNR=([-0-9.]+).*?FER=([-0-9.e+]+).*?UF=(\d+).*?TxMC=([a-z0-9/]+)\((\d+)\)')

    def parse_feeder(self, n, line):
        if 'log :info:' not in line:
            return
        m = self.FEEDER.search(line)
        if not m:
            return
        with self.lock:
            self.feeder[n] = {'locked': m.group(1) == 'True', 'snr_db': float(m.group(2)), 'fer': float(m.group(3)),
                              'underflows': int(m.group(4)), 'tx_mc': m.group(5), 'tx_mc_index': int(m.group(6)),
                              'time': time.time()}

    DL = re.compile(r'UE ([0-9a-f]+): dlsch_rounds (\d+)/(\d+)/(\d+)/(\d+), dlsch_errors (\d+).*?BLER ([0-9.]+) MCS \(\d\) (\d+) \(Qm (\d)')
    UL = re.compile(r'UE ([0-9a-f]+): ulsch_rounds (\d+)/(\d+)/(\d+)/(\d+), ulsch_errors (\d+).*?BLER ([0-9.]+) MCS \(\d\) (\d+) \(Qm (\d).*?NPRB (\d+) SNR ([-0-9.]+)')
    GOOD = re.compile(r'UE ([0-9a-f]+): LCID [0-9,]+, goodput DL\s+([0-9.]+) UL\s+([0-9.]+) Mbps')

    def pass_bler(self, direction, rnti, first, retx):
        # First-transmission BLER since this pass began, from OAI's HARQ round counters
        # (OAI's own BLER field is a moving average that carries over between passes).
        key = (rnti, self.orbit['pass_id'])
        base = self.bler_base.get(direction)
        if not base or base[0] != key or first < base[1]:
            self.bler_base[direction] = (key, first, retx)
            return None
        sent = first - base[1]
        return (retx - base[2]) / sent if sent >= 20 else None

    def parse_gnb(self, line):
        now = time.time()
        with self.lock:
            m = self.DL.search(line)
            if m:
                self.gnb.update(rnti=m.group(1), dl_bler=self.pass_bler('dl', m.group(1), int(m.group(2)), int(m.group(3))),
                                dl_mcs=int(m.group(8)), dl_qm=int(m.group(9)), dl_errors=int(m.group(6)), time=now)
                return
            m = self.UL.search(line)
            if m:
                self.gnb.update(rnti=m.group(1), ul_bler=self.pass_bler('ul', m.group(1), int(m.group(2)), int(m.group(3))),
                                ul_mcs=int(m.group(8)), ul_qm=int(m.group(9)), ul_prb=int(m.group(10)),
                                ul_snr_db=float(m.group(11)), ul_errors=int(m.group(6)), time=now)
                return
            m = self.GOOD.search(line)
            if m:
                self.gnb.update(goodput_dl=float(m.group(2)), goodput_ul=float(m.group(3)), time=now)
            elif 'Assertion' in line or 'Exiting execution' in line:
                self.gnb['crashed'] = now
            elif 'Remove UE context' in line:
                self.gnb['released'] = now

    def counter_loop(self):
        previous = {}
        while True:
            time.sleep(1)
            if not self.active():
                previous.clear(); continue
            try:
                a = self.ssh('11', 'cat /run/cusat-ue.json 2>/dev/null || echo {}; echo; '
                                   f'cat /sys/class/net/{TAPS["11"]}/statistics/tx_bytes /sys/class/net/{TAPS["11"]}/statistics/rx_bytes 2>/dev/null', 8).stdout.split('\n')
                b = self.ssh('12', 'cat /sys/class/net/ogstun/statistics/rx_bytes /sys/class/net/ogstun/statistics/tx_bytes '
                                   f'/sys/class/net/{TAPS["12"]}/statistics/tx_bytes 2>/dev/null', 8).stdout.split()
                ue = json.loads(a[0] or '{}')
                numbers = [int(x) for x in a[1:] if x.strip().isdigit()]
                now = time.time()
                ue_time = (ue.get('wwan') or {}).get('time')
                current = {'wwan_rx': (ue.get('wwan') or {}).get('rx_bytes'), 'wwan_tx': (ue.get('wwan') or {}).get('tx_bytes'),
                           'core_ul': int(b[0]) if len(b) > 0 else None, 'core_dl': int(b[1]) if len(b) > 1 else None,
                           'feed_up': numbers[0] if numbers else None, 'feed_down': int(b[2]) if len(b) > 2 else None}
                rates = {}
                if previous:
                    dt = max(0.2, now - previous['time'])
                    for key, value in current.items():
                        old = previous.get(key)
                        interval = dt
                        if key.startswith('wwan_') and ue_time and previous.get('ue_time'):
                            interval = ue_time - previous['ue_time']
                            if interval <= 0:
                                # The SSH poll can read the same agent sample twice.
                                rates[key] = self.rates.get(key)
                                continue
                        rates[key] = (value - old) * 8 / interval / 1e6 if value is not None and old is not None and value >= old else None
                previous = dict(current, time=now, ue_time=ue_time)
                with self.lock:
                    self.ue = ue
                    if rates:
                        self.rates = dict(rates, time=now - 0.5)
            except (subprocess.SubprocessError, OSError, ValueError, IndexError):
                pass

    def watchdog_loop(self):
        # Re-create the UE session if it goes dead while the feeder is up (a 5G
        # re-establishment or core-side release leaves stale addresses behind).
        while True:
            time.sleep(5)
            if not self.active() or self.status['setup'] != 'running':
                continue
            with self.lock:
                ping = self.ue.get('ping') or {}
                quiet = time.time() - (ping.get('time') or 0)
                visible_for = self.orbit['t'] if self.orbit['phase'] == 'pass' else 0
            if quiet > 30 and visible_for > 15 and time.time() - self.status['last_restart'] > 120:
                # Only when the gNB runs: without a cell a new UE session cannot help.
                try:
                    gnb_up = 'true' in self.ssh('11', 'docker inspect -f {{.State.Running}} cusat-oai 2>/dev/null').stdout
                except subprocess.SubprocessError:
                    gnb_up = False
                if not gnb_up:
                    continue
                self.status['setup'] = 'restarting UE'
                try:
                    self.restart_ue(lambda s: print('cellular watchdog:', s, flush=True))
                except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                    self.status['error'] = str(error)[-300:]
                    print('cellular watchdog: ' + str(error)[-300:], flush=True)
                finally:
                    self.status['setup'] = 'running'

    # --------------------------------------------------------------- payload
    def cell_info(self):
        # Band and SSB frequency of the configured cell (configs/oai-gnb.conf).
        text = (BASE / 'configs' / 'oai-gnb.conf').read_text()
        def value(name):
            m = re.search(r'\b' + name + r'\s*=\s*(\d+)', text)
            return int(m.group(1)) if m else None
        arfcn, prbs = value('absoluteFrequencySSB'), value('dl_carrierBandwidth')
        return {'band': value('dl_frequencyBand'), 'ssb_mhz': 3000 + (arfcn - 600000) * 0.015 if arfcn and arfcn >= 600000 else None,
                'prbs': prbs, 'scs_khz': 30, 'bandwidth_mhz': 20 if prbs == 51 else None}

    def pass_ping(self, ping):
        # Ping RTT and loss restart with each pass: samples are [send time, RTT or None],
        # windowed to the last 15 s of the current pass. A ping counts as lost only once
        # it is 6 s old without a reply (late replies after reacquisition still count).
        now, start = time.time(), self.orbit['pass_started']
        samples = [x for x in ping.get('recent', []) if isinstance(x, list) and x[0] >= start]
        settled = [rtt for sent, rtt in samples if now - 6 >= sent >= max(start, now - 21)]
        loss = sum(1 for rtt in settled if rtt is None) / len(settled) if len(settled) >= 4 else None
        current = next(((sent, rtt) for sent, rtt in reversed(samples) if rtt is not None), None)
        age = now - (current[0] + current[1] / 1000) if current else None
        return (current[1] if current else None), age, loss

    def payload(self):
        with self.lock:
            ue = self.ue or {}
            ping = ue.get('ping') or {}
            rtt, age, loss = self.pass_ping(ping)
            return {'orbit': dict(self.orbit), 'feeder': json.loads(json.dumps(self.feeder)), 'gnb': dict(self.gnb),
                    'ue': {'modem': ue.get('modem') or {}, 'ip': (ue.get('wwan') or {}).get('ip'),
                           'ping_ms': rtt, 'ping_age_s': age, 'ping_loss': loss,
                           'iperf': ue.get('iperf') or {}},
                    'rates': dict(self.rates), 'offered': RATES, 'traffic': dict(self.traffic),
                    'status': dict(self.status), 'cell': self.cell_info(),
                    # Only the latest, currently active gap may be open.
                    'outages': [list(x) for i, x in enumerate(self.outages)
                                if x[1] is not None or (i == len(self.outages) - 1
                                and self.orbit['phase'] == 'gap'
                                and self.status['setup'] == 'running')]}
