'use strict';
// Cellular demo page. State comes from dashboard-server.py (/api/state, "cellular" section);
// controls post to /api/control. The server owns the LEO pass and the feeder emulation.
const $ = id => document.getElementById(id);
let live = null, lastPoll = 0, inflight = 0;

async function post(action, value) {
  inflight++; render();
  try {
    const response = await fetch('/api/control', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action, value})});
    const reply = await response.json();
    if (!response.ok || !reply.ok) throw new Error(reply.error || response.statusText);
  } catch (error) { $('live-status').textContent = 'Control failed: ' + error.message; }
  finally { inflight--; render(); }
}

function fmt(v, digits=1) { return typeof v === 'number' && isFinite(v) ? v.toFixed(digits) : '—'; }
function value(id, v, unit, digits=1) { $(id).replaceChildren(document.createTextNode(fmt(v, digits) + ' ')); const u = document.createElement('small'); u.textContent = unit; $(id).append(u); }
function arc(id, fraction) { $(id).style.strokeDasharray = (Math.max(0, Math.min(1, fraction || 0)) * 100) + ' 100'; }
function facts(id, rows) {
  $(id).replaceChildren(...rows.flatMap(([k, v, cls]) => { const dt = document.createElement('dt'); dt.textContent = k; const dd = document.createElement('dd'); dd.textContent = v; if (cls) dd.className = cls; return [dt, dd]; }));
}
function mbps(value) { return typeof value === 'number' ? Math.max(0, value) : null; }
function offeredMbps(text) { const m = /^([0-9.]+)\s*([KMG]?)/.exec(text || ''); return m ? +m[1] * ({K:1e-3, M:1, G:1e3, '':1e-6})[m[2]] : null; }
function mc(name) { return name ? name.toUpperCase().replace(/^(\d*)([A-Z]+)(\d)\/(\d+)$/, '$1$2 $3/$4').replace('PSK', 'PSK').replace('APSK', 'APSK') : '—'; }
// BLER and ping loss are per pass (the server restarts them when a pass begins).
function bler(v) { return typeof v === 'number' ? (v * 100).toFixed(1) + '%' : 'new pass'; }
function feederDir(tx, rx) { return {mc: tx.tx_mc, snr: rx.snr_db, locked: !!rx.locked, fresh: fresh(rx.time, 4)}; }
function family(name) { const m = /^(qpsk|8psk|16apsk|32apsk)/.exec(name || ''); return m ? {qpsk:'qpsk', '8psk':'psk8', '16apsk':'apsk16', '32apsk':'apsk32'}[m[1]] : null; }
function fresh(t, s=5) { return typeof t === 'number' && clockOffset !== null && Date.now() - local(t) < s * 1000; }

// ---------------------------------------------------------------- render
function render() {
  const c = live && live.cellular, mode = live && live.demo_mode || {};
  const switching = !!mode.switching;
  document.body.classList.toggle('switching', switching);
  $('switch-title').textContent = switching ? ({difi:'Moving the radios to the DIFI demo…', stopped:'Stopping the demo…'}[mode.switching] || 'Starting the cellular demo…') : 'Switching demo…';
  $('switch-log').replaceChildren(...(mode.error ? [mode.error] : mode.log || []).slice(-8).map(line => { const li = document.createElement('li'); li.textContent = line; return li; }));
  renderProgress(mode, live && live.server_time);
  const busy = inflight > 0 || (live && live.busy);
  for (const id of ['restart-ue', 'restart-cell', 'demo-switch', 'demo-stop', 'animate']) $(id).disabled = !c || busy || switching;
  if (!c) { $('live-status').textContent = !live ? 'Dashboard server unreachable' : 'Waiting for the cellular demo'; ready('no cellular telemetry'); return; }

  const ue = c.ue || {}, modem = ue.modem || {}, gnb = c.gnb || {}, o = c.orbit || {}, rates = c.rates || {}, cell = c.cell || {};
  const fa = c.feeder['11'] || {}, fb = c.feeder['12'] || {};
  const offered = {dl: offeredMbps(c.offered.dl), ul: offeredMbps(c.offered.ul)};
  const registered = /Registered/.test(modem.mm || ''), connected = /Connected/.test(modem.rrc || '');
  const pingFresh = typeof ue.ping_age_s === 'number' && ue.ping_age_s < 4;
  const gnbFresh = fresh(gnb.time, 6);
  const dl = mbps(rates.wwan_rx), ul = mbps(rates.core_ul);
  // Each feeder modem reports the MODCOD it transmits and the SNR it receives: the return
  // link (.18 → .20) pairs .18's MODCOD with .20's SNR, the forward link the reverse.
  const ret = feederDir(fa, fb), fwd = feederDir(fb, fa);
  const feederRow = d => d.fresh ? (d.locked ? mc(d.mc) + ' · ' + fmt(d.snr) + ' dB' : 'reacquiring') : '—';

  // Path panel.
  facts('ue-facts', [
    ['State', registered ? (connected ? 'registered · connected' : 'registered · idle') : (modem.mm || '—'), registered ? 'good' : 'bad'],
    ['PDU', ue.ip ? ue.ip + ' (oai)' : 'none', ue.ip ? 'good' : 'bad'],
    ['TX power', typeof modem.tx_power_dbm === 'number' ? modem.tx_power_dbm.toFixed(0) + ' dBm' : '—'],
  ]);
  $('radio-title').textContent = '5G NR ' + (cell.band ? 'n' + cell.band : '') + (cell.ssb_mhz ? ' · ' + (cell.ssb_mhz / 1000).toFixed(2) + ' GHz' : '') + (cell.bandwidth_mhz ? ' · ' + cell.bandwidth_mhz + ' MHz TDD' : '');
  facts('radio-facts', [
    ['RSRP / SINR', fmt(modem.rsrp_dbm, 0) + ' dBm / ' + fmt(modem.sinr_db, 0) + ' dB'],
    ['Downlink', gnbFresh ? 'MCS ' + gnb.dl_mcs + ' · BLER ' + bler(gnb.dl_bler) : '—'],
    ['Uplink', gnbFresh ? 'MCS ' + gnb.ul_mcs + ' · BLER ' + bler(gnb.ul_bler) : '—'],
  ]);
  facts('gnb-facts', [
    ['gNB', gnbFresh ? 'serving UE ' + gnb.rnti : (gnb.crashed ? 'stopped' : 'no UE'), gnbFresh ? 'good' : 'bad'],
    ['5G cell', gnbFresh && typeof gnb.goodput_dl === 'number' ? fmt(gnb.goodput_dl) + '↓ ' + fmt(gnb.goodput_ul) + '↑ Mb/s' : '—'],
    ['Modem RX', feederRow(fwd)],
  ]);
  // Feeder: the return link (remote → gateway) is what the .20 end receives (its modem
  // reports the MODCOD .18 transmits), the forward link the reverse.
  $('sat-geometry').textContent = o.visible === false ? 'out of view · next pass in ' + fmt(o.next_pass_s, 0) + ' s'
    : fmt(o.elevation_deg, 0) + '° elevation · ' + fmt(o.slant_km, 0) + ' km';
  $('feeder-now').textContent = o.visible === false ? 'out of view' :
    fmt(mbps(rates.feed_down)) + ' ↓ · ' + fmt(mbps(rates.feed_up)) + ' ↑ Mb/s';
  facts('x18-facts', [
    ['5G ch1', cell.ssb_mhz ? (cell.ssb_mhz / 1000).toFixed(2) + ' GHz' : '—'],
    ['Feeder ch0', '↑2.05 ↓1.95 GHz'],
  ]);
  facts('x20-facts', [
    ['Feeder ch0', '↑1.95 ↓2.05 GHz'],
    ['Modem RX', feederRow(ret)],
  ]);
  facts('core-facts', [
    ['UE traffic', fmt(ul) + '↑ ' + fmt(mbps(rates.core_dl)) + '↓ Mb/s'],
    ['UPF', '10.45.0.1'],
  ]);
  $('node-ue').classList.toggle('down', !(registered && ue.ip));
  $('node-gnb').classList.toggle('down', !gnbFresh);
  $('node-sat').classList.toggle('outage', o.visible === false);
  $('legs').classList.toggle('outage', o.visible === false);

  // Gauges.
  const dlMax = Math.max(1, (offered.dl || 8) * 1.25), ulMax = Math.max(0.5, (offered.ul || 2) * 1.25);
  $('dl-max').textContent = dlMax.toFixed(0) + ' Mb/s'; $('ul-max').textContent = ulMax.toFixed(1) + ' Mb/s';
  value('dl-rate', dl, 'Mb/s'); arc('dl-arc', (dl || 0) / dlMax);
  value('ul-rate', ul, 'Mb/s', 2); arc('ul-arc', (ul || 0) / ulMax);
  const dlLoss = (ue.iperf && ue.iperf.dl && typeof ue.iperf.dl.loss === 'number') ? ue.iperf.dl.loss : null;
  $('dl-note').textContent = (offered.dl ? offered.dl + ' Mb/s offered' : '') + (dlLoss !== null ? ' · loss ' + (dlLoss * 100).toFixed(1) + '%' : '');
  $('ul-note').textContent = offered.ul ? offered.ul + ' Mb/s offered' : '—';
  if (pingFresh) value('ping-rate', ue.ping_ms, 'ms', 0); else $('ping-rate').innerHTML = '— <small>ms</small>';
  arc('ping-arc', pingFresh ? ue.ping_ms / 200 : 0);
  $('ping-note').textContent = 'feeder ' + fmt(2 * (o.one_way_ms || 0), 1) + ' ms' + (typeof ue.ping_loss === 'number' ? ' · loss ' + (ue.ping_loss * 100).toFixed(0) + '%' : '');
  const snrs = [fa, fb].filter(f => fresh(f.time, 4) && f.locked).map(f => f.snr_db);
  const snr = snrs.length ? snrs.reduce((x, y) => x + y, 0) / snrs.length : null;
  if (snr === null) $('snr-rate').innerHTML = (o.visible === false ? 'Out' : '—') + ' <small>' + (o.visible === false ? 'of view' : 'dB') + '</small>'; else value('snr-rate', snr, 'dB');
  arc('snr-arc', (snr || 0) / 30);
  $('snr-note').textContent = 'ACM ' + mc(ret.mc) + ' ↑ / ' + mc(fwd.mc) + ' ↓';
  $('graph-rate').textContent = fmt(dl) + ' ↓ · ' + fmt(ul, 2) + ' ↑ Mb/s';
  $('telemetry-note').textContent = 'DVB-S2 feeder 24 Msym/s · roll-off 0.1 · ACM · LEO 550 km, ' + (o.mask_deg || 10) + '° mask · pass ' + fmt(o.pass_s, 0) + ' s + ' + fmt(o.gap_s, 0) + ' s gap';
  $('animate').textContent = o.animate === false ? 'Resume pass' : 'Pause pass';

  // Pass readouts.
  facts('pass-readouts', [
    ['Elevation', o.visible ? fmt(o.elevation_deg, 0) + '°' : '—'],
    ['Slant range', o.visible ? fmt(o.slant_km, 0) + ' km' : '—'],
    ['Path loss vs zenith', o.visible ? '+' + fmt(o.loss_db, 1) + ' dB' : 'link off'],
    ['Feeder delay', fmt(o.one_way_ms, 1) + ' ms'],
  ]);

  // Status and readiness.
  const why = !live || performance.now() - lastPoll > 3000 ? 'no telemetry' : switching ? 'switching demo' : busy ? 'restarting' :
    !registered ? 'UE not registered' : !ue.ip ? 'no PDU session' : !gnbFresh ? 'gNB not serving' :
    (o.visible !== false && !pingFresh && !(o.t < 15)) ? 'no traffic' : '';
  ready(why);
  $('live-status').textContent = busy ? (live.last_action || 'Applying…') : o.visible === false ? 'Feeder out of view · next pass in ' + fmt(o.next_pass_s, 0) + ' s' :
    (!pingFresh && o.t < 15) ? 'New pass · feeder reacquiring' : 'Live · elevation ' + fmt(o.elevation_deg, 0) + '°';
  $('live-status').title = live.last_action || '';
}
function ready(why) {
  $('radio-ready').classList.toggle('ready', !why);
  $('radio-ready').lastChild.textContent = why ? 'Not ready · ' + why : 'UE connected';
}

// ------------------------------------------------- history on the server clock
const windowMs = 120000, points = [];
let clockOffset = null, lastKey = '', lastGap = 0;
function syncClock(t) { if (typeof t !== 'number') return; const off = t * 1000 - Date.now(); clockOffset = clockOffset === null || Math.abs(off - clockOffset) > 2000 ? off : clockOffset + .1 * (off - clockOffset); }
function local(t) { return t * 1000 - (clockOffset || 0); }
function record() {
  const c = live && live.cellular; if (!c || clockOffset === null) return;
  const r = c.rates || {}, fa = c.feeder['11'] || {}, fb = c.feeder['12'] || {};
  if (typeof r.time !== 'number') { const now = Date.now(); if (now - lastGap > 1000) { points.push({t:now}); lastGap = now; } return; }
  const key = String(r.time); if (key === lastKey) return; lastKey = key;
  const fwd = feederDir(fb, fa), ret = feederDir(fa, fb), up = d => d.fresh && d.locked;
  points.push({t:local(r.time), dl:mbps(r.wwan_rx), ul:mbps(r.core_ul),
    fwdRate:mbps(r.feed_down), retRate:mbps(r.feed_up),
    fwdSnr:up(fwd) ? fwd.snr : null, retSnr:up(ret) ? ret.snr : null,
    fwdMc:up(fwd) ? fwd.mc : null, retMc:up(ret) ? ret.mc : null});
  while (points.length && points[0].t < Date.now() - windowMs - 5000) points.shift();
}
function xOf(t, now) { return (t - (now - windowMs)) / windowMs; }
function path(key, scale, now) {
  let open = false;
  return points.map(p => { if (p[key] === null || p[key] === undefined) { open = false; return ''; } const s = (open ? 'L' : 'M') + (xOf(p.t, now) * 800).toFixed(1) + ' ' + scale(p[key]).toFixed(1); open = true; return s; }).join(' ');
}
const events = new Map();
function syncEvents(list) { const seen = new Set(); for (const e of list || []) { seen.add(e.id); events.set(e.id, e); } for (const id of [...events.keys()]) if (!seen.has(id)) events.delete(id); }
function plot() {
  const c = live && live.cellular, now = Date.now();
  const top = Math.max(2, ((c && offeredMbps(c.offered.dl)) || 8) * 1.25);
  $('axis-top').textContent = top.toFixed(0) + ' Mb/s'; $('axis-mid').textContent = (top / 2).toFixed(0);
  const y = v => 195 - Math.min(top * 1.05, Math.max(0, v)) / top * 190, ys = v => 195 - Math.min(31, Math.max(0, v)) / 30 * 190;
  $('dl-line').setAttribute('d', path('dl', y, now)); $('ul-line').setAttribute('d', path('ul', y, now));
  const ftop = 40, yf = v => 195 - Math.min(ftop * 1.05, Math.max(0, v)) / ftop * 190;
  $('fwd-rate').setAttribute('d', path('fwdRate', yf, now)); $('ret-rate').setAttribute('d', path('retRate', yf, now));
  $('fwd-snr').setAttribute('d', path('fwdSnr', ys, now)); $('ret-snr').setAttribute('d', path('retSnr', ys, now));
  lanes('lane-fwd', 'fwdMc', now); lanes('lane-ret', 'retMc', now);
  const bands = [];
  for (const [start, end] of (c && c.outages) || []) {
    const x0 = Math.max(0, xOf(local(start), now)), x1 = Math.min(1, xOf(end ? local(end) : now, now));
    if (x1 <= 0 || x0 >= 1) continue;
    const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    r.setAttribute('x', (x0 * 800).toFixed(1)); r.setAttribute('width', ((x1 - x0) * 800).toFixed(1)); r.setAttribute('y', 0); r.setAttribute('height', 200); r.setAttribute('class', 'outage-band');
    bands.push(r);
  }
  $('outage-bands').replaceChildren(...bands);
  $('feeder-outages').replaceChildren(...bands.map(b => b.cloneNode()));
  const marks = [];
  for (const e of [...events.values()].sort((a, b) => a.time - b.time)) {
    const at = e.applied || e.time, x = Math.min(100, xOf(local(at), now) * 100);
    if (x < 0) continue;
    const m = document.createElement('div'); m.className = 'chart-event ' + e.kind + ' ' + e.status + (x < 16 ? ' flip' : ''); m.style.left = x + '%';
    const s = document.createElement('span'); s.textContent = e.label + (e.status === 'pending' ? ' …' : e.status === 'failed' ? ' ✕' : e.ms ? ' · ' + Math.round(e.ms / 1000) + ' s' : ''); m.append(s); marks.push(m);
  }
  $('chart-events').replaceChildren(...marks);
}

// MODCOD lanes under the feeder graph: one block per modulation, labelled with the
// code rates ACM used inside it (ACM steps between neighbouring rates every few seconds).
function rate(name) { const m = /(\d+)\/(\d+)$/.exec(name || ''); return m ? +m[1] / +m[2] : null; }
function lanes(id, key, now) {
  const blocks = []; let current = null;
  for (const p of points) {
    const x = xOf(p.t, now); if (x < -0.01) continue;
    const v = p[key] || null, f = family(v);
    if (current && current.f === f) { current.x1 = x; if (v) current.names.add(v); continue; }
    if (current) current.x1 = x;
    current = {f, x0:Math.max(0, x), x1:x, names:new Set(v ? [v] : [])}; blocks.push(current);
  }
  const lane = $(id), label = lane.firstElementChild;
  lane.replaceChildren(label, ...blocks.filter(g => g.f && g.x1 > g.x0).map(g => {
    const names = [...g.names].sort((a, b) => rate(a) - rate(b));
    const head = mc(names[0]).split(' ')[0], rates = names.map(n => mc(n).split(' ')[1]);
    const text = head + ' ' + (rates.length > 1 ? rates[0] + '–' + rates[rates.length - 1] : rates[0]);
    const d = document.createElement('span'); d.className = 'mc-seg ' + g.f;
    d.style.left = (g.x0 * 100) + '%'; d.style.width = (Math.min(1, g.x1) - g.x0) * 100 + '%';
    if (g.x1 - g.x0 > 0.06) d.textContent = text; d.title = names.map(mc).join(', '); return d;
  }));
}

// Feeder legs: from each X410 up to the satellite, in the scene's own pixels. The
// satellite sits exactly between the two radios so the legs mirror each other.
let satShift = 0;
function drawLegs() {
  const scene = $('scene'), sat = document.querySelector('#node-sat .sat-icon');
  if (!scene || !sat) return;
  const box = scene.getBoundingClientRect();
  const radios = ['x18', 'x20'].map(id => document.querySelector('#node-' + id + ' .device').getBoundingClientRect());
  const mid = (radios[0].left + radios[0].width / 2 + radios[1].left + radios[1].width / 2) / 2;
  let s = sat.getBoundingClientRect();
  satShift += mid - (s.left + s.width / 2);
  $('node-sat').style.transform = 'translateX(' + satShift.toFixed(1) + 'px)';
  s = sat.getBoundingClientRect();
  const top = {x: s.left + s.width / 2 - box.left, y: s.bottom - box.top - 4};
  $('legs').setAttribute('viewBox', '0 0 ' + box.width.toFixed(0) + ' ' + box.height.toFixed(0));
  ['x18', 'x20'].forEach((id, i) => {
    const r = document.querySelector('#node-' + id + ' .device').getBoundingClientRect();
    const x = r.left + r.width / 2 - box.left, y = r.top - box.top + 2;
    $(i ? 'leg-b' : 'leg-a').setAttribute('d', 'M' + x.toFixed(1) + ' ' + y.toFixed(1) + ' L' + (top.x + (i ? 14 : -14)).toFixed(1) + ' ' + top.y.toFixed(1));
  });
}
window.addEventListener('resize', drawLegs);

// ------------------------------------------------------------------ the pass
// The pass spans Earth-central angles ±max_angle; draw it so the satellite crosses the
// two 20° mask lines (visual ±29.8° on the schematic orbit) and keeps going out of view.
let shown = {angle: 0, visible: true, gapT: 0};
function drawPass() {
  const o = live && live.cellular && live.cellular.orbit; if (!o) return;
  let visual;
  if (o.visible) visual = (o.angle / o.max_angle) * 29.8;
  else visual = 29.8 + Math.min(1, (o.gap_s - (o.next_pass_s || 0)) / o.gap_s) * 30;
  const a = visual * Math.PI / 180, x = 150 + 102 * Math.sin(a), y = 150 - 102 * Math.cos(a);
  $('satellite-position').setAttribute('transform', 'translate(' + x.toFixed(1) + ' ' + y.toFixed(1) + ') rotate(' + (visual + 45).toFixed(1) + ')');
  $('satellite-position').classList.toggle('faded', !o.visible);
  $('orbit-link').setAttribute('x2', x); $('orbit-link').setAttribute('y2', y); $('orbit-link').setAttribute('visibility', o.visible ? 'visible' : 'hidden');
  $('out-of-view').toggleAttribute('hidden', !!o.visible);
  $('out-of-view-text').textContent = 'OUT OF VIEW · NEXT PASS ' + fmt(o.next_pass_s, 0) + ' s';
}

// -------------------------------------------------------------------- wiring
$('restart-ue').addEventListener('click', () => post('cell_restart_ue', null));
$('restart-cell').addEventListener('click', () => { if (confirm('Restart the gNB, the feeder modems and the core services, then reattach the UE? (about 2 minutes)')) post('cell_restart', null); });
$('animate').addEventListener('click', () => post('cell_animate', !(live && live.cellular && live.cellular.orbit.animate)));
$('demo-switch').addEventListener('click', () => { if (confirm('Switch to the DIFI demo?\n\nThis stops OAI and the feeder, deletes the UE session, reflashes both X410s with the LIVESPLIT image and restarts the DIFI modems. It takes a few minutes.')) post('demo_mode', 'difi'); });
$('demo-stop').addEventListener('click', () => { if (confirm('Stop the demo?\n\nThis stops the gNB and the feeder modems, deletes the UE session and puts the Sierra in airplane mode. The radios are left idle.')) post('demo_mode', 'stopped'); });
$('fullscreen').addEventListener('click', async () => { try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); } catch {} });

async function poll() {
  try {
    const response = await fetch('/api/state?lite=1', {cache:'no-store'});
    live = await response.json(); lastPoll = performance.now();
    syncClock(live.server_time); syncEvents(live.events);
    // The server serves the page of the demo that owns the radios.
    const m = live.demo_mode || {};
    if (m.active !== 'cellular' && !m.switching) location.href = '/';
  } catch { live = live && {...live}; }
  setTimeout(poll, 250);
}
setInterval(() => { record(); render(); plot(); drawPass(); drawLegs(); }, 250);
poll();

// Demo transition progress: steps advance as the server sees each milestone; the bar
// fills against the measured step durations, so a long FPGA load visibly moves.
function clock(seconds) { seconds = Math.max(0, Math.round(seconds)); return Math.floor(seconds / 60) + ':' + String(seconds % 60).padStart(2, '0'); }
function renderProgress(mode, serverTime) {
  const p = mode && mode.progress, box = document.getElementById('progress');
  box.hidden = !(mode && mode.switching && p);
  if (box.hidden) return;
  const now = typeof serverTime === 'number' ? serverTime : Date.now() / 1000;
  const expected = p.phases.map(x => x[1]), total = expected.reduce((a, b) => a + b, 0);
  const done = expected.slice(0, p.index).reduce((a, b) => a + b, 0), inStep = now - p.phase_started;
  const current = Math.min(inStep, expected[p.index] * 0.95);
  const fraction = Math.min(0.99, (done + current) / total);
  const left = Math.max(0, expected[p.index] - inStep) + expected.slice(p.index + 1).reduce((a, b) => a + b, 0);
  document.getElementById('progress-fill').style.width = (fraction * 100).toFixed(1) + '%';
  document.getElementById('progress-bar').setAttribute('aria-valuenow', Math.round(fraction * 100));
  document.getElementById('progress-step').textContent = 'Step ' + (p.index + 1) + ' of ' + p.phases.length + ' · ' + p.phases[p.index][0];
  document.getElementById('progress-time').textContent = clock(now - p.started) + ' elapsed · ' + (inStep > expected[p.index] * 1.5 ? 'taking longer than usual' : '~' + clock(left) + ' left');
  document.getElementById('phase-list').replaceChildren(...p.phases.map(([label], k) => {
    const li = document.createElement('li'); li.className = k < p.index ? 'done' : k === p.index ? 'current' : 'pending';
    li.textContent = label; return li;
  }));
}
