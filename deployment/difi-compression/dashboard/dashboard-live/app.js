'use strict';
// Live UI. State and telemetry come from dashboard-server.py (/api/state); controls post
// to /api/control. The server owns the orbit and the emulated channel; this page only
// extrapolates the orbit between polls so the drawing stays smooth.
const txStages = [
  ['GSE + BB framing', 'CPU'], ['BCH + LDPC encode', 'CPU / GPU'], ['Interleave + PL header', 'Host / GPU'],
  ['Constellation mapping', 'Host / GPU'], ['Pilots + PL scrambling', 'Host / GPU'], ['RRC + resampling', 'Host / GPU'], ['RF output', 'RF']
];
// Spatial ordering is host-left/radio-right; RX signal traverses right-to-left.
const rxStages = [
  ['GSE + BB recovery', 'CPU'], ['LDPC + BCH decode', 'CPU / GPU'], ['Soft demap + deinterleave', 'Host / GPU'],
  ['Carrier + frame/pilot recovery', 'Host / GPU'], ['Timing recovery', 'Host / GPU'], ['Matched filter', 'Host / GPU'], ['RF input', 'RF']
];
// Every route is selectable. RX changes switch in-session (~0.5 s); TX changes restart the modems (~20 s).
const txSplits = [
  {name:'Constellation indices', letter:'C', boundary:3, detail:'FPGA maps indices, inserts pilots and PL scrambling, then shapes the waveform.', representation:'Packed indices + PL header', key:'indices'},
  {name:'Symbol IQ', letter:'B', boundary:5, detail:'Host constructs PL symbols; FPGA performs pulse shaping and interpolation.', representation:'Complex PL symbols · 1 sample/symbol', key:'symbols'},
  {name:'Sample IQ', letter:'A', boundary:6, detail:'Host generates the shaped waveform; FPGA handles radio conversion.', representation:'Waveform IQ · 4 samples/symbol', key:'samples'}
];
const rxSplits = [
  {name:'Timing-recovered symbols', letter:'C', boundary:4, detail:'FPGA matched filtering and symbol timing recovery. Carrier recovery, frame synchronization and decoding remain on the host.', representation:'Synchronized IQ · 1 sample/symbol', key:'sync'},
  {name:'Filtered IQ', letter:'B', boundary:5, detail:'FPGA matched filtering (RRC) with 2× decimation. Host recovers timing and decodes.', representation:'Filtered IQ · 2 samples/symbol', key:'filtered'},
  {name:'Sample IQ', letter:'A', boundary:6, detail:'Host performs matched filtering, synchronization and decoding.', representation:'Sample IQ · 4 samples/symbol', key:'samples'}
];
// Split-point popovers: what crosses the DIFI link at each split, drawn as a small
// diagram, with the O-RAN analog. Rates are at 15.36 Msym/s.
const tips = {
  'tx:samples': {diagram:'wave-tx', what:'The shaped transmit waveform: 16-bit I + Q at 4 samples/symbol.',
    rate:'≈ 1,966 Mb/s (the DIFI baseline)', host:'Everything through RRC pulse shaping and resampling', radio:'DAC and RF conversion only',
    oran:'Split 8: time-domain IQ, as in CPRI'},
  'tx:symbols': {diagram:'const-clean', what:'One complex value per physical-layer symbol (pilots and scrambling applied), 16-bit I + Q.',
    rate:'≈ 492 Mb/s, 4× below sample IQ', host:'Framing, FEC, mapping, pilots, scrambling', radio:'RRC pulse shaping and 4× interpolation',
    oran:'Split 7.2x downlink: the DU sends modulated IQ and the RU generates the waveform (iFFT + CP in OFDM; pulse shaping here)'},
  'tx:indices': {diagram:'const-bits', what:'Packed constellation indices (log₂M bits per symbol) plus the physical-layer header. No IQ crosses the link.',
    rate:'31–77 Mb/s (QPSK–32APSK); ≈ 46 Mb/s at 8PSK, 25–64× below sample IQ', host:'Framing, FEC and interleaving', radio:'Mapping, pilots, scrambling and pulse shaping',
    oran:'modulation compression (7.2x downlink): the DU sends constellation bits and the RU maps them to IQ'},
  'rx:samples': {diagram:'wave-rx', what:'Raw received IQ at 4 samples/symbol, with noise across the whole sample band.',
    rate:'≈ 1,966 Mb/s at 16-bit, 983 at 8-bit', host:'Matched filter, timing, carrier and frame recovery, decoding', radio:'RF and digital down-conversion only',
    oran:'Split 8: time-domain IQ, as in CPRI'},
  'rx:filtered': {diagram:'wave-filtered', what:'Matched-filtered IQ decimated to 2 samples/symbol; the out-of-band noise is gone.',
    rate:'≈ 983 Mb/s at 16-bit, 492 at 8-bit, 2× below sample IQ', host:'Timing recovery onward', radio:'RRC matched filter and 2× decimation',
    oran:'Like split 7-1 uplink: the RU does the front-end filtering and rate reduction (CP removal + FFT in OFDM); synchronization and decoding stay in the DU'},
  'rx:sync': {diagram:'const-noisy', what:'One IQ value per symbol, taken at the best sampling instant by the FPGA timing loop.',
    rate:'≈ 492 Mb/s at 16-bit, 246 at 8-bit, 4× below sample IQ', host:'Carrier and frame recovery, demapping, decoding', radio:'Matched filter and feedforward symbol timing recovery',
    oran:'Split 7.2x uplink: the RU delivers symbol-level IQ (per resource element after the FFT in OFDM); equalization and decoding stay in the DU'}
};
const SVG='http://www.w3.org/2000/svg';
function svgNode(tag,attrs,text) { const n=document.createElementNS(SVG,tag); for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v); if(text!==undefined)n.textContent=text; return n; }
function seeded(seed) { let x=seed>>>0; return ()=>{ x=(1664525*x+1013904223)>>>0; return x/4294967296; }; }
function gauss(rand) { return Math.sqrt(-2*Math.log(rand()+1e-9))*Math.cos(2*Math.PI*rand()); }
// Raised-cosine pulse (roll-off 0.35, wider than the modem's 0.2 so the sketch reads clearly).
function pulse(t) { const b=.35; if(Math.abs(t)<1e-6)return 1; const s=Math.sin(Math.PI*t)/(Math.PI*t), d=1-(2*b*t)**2; return Math.abs(d)<1e-6?s*Math.PI/4:s*Math.cos(Math.PI*b*t)/d; }
function waveDiagram(sps,noise,caption) {
  const svg=svgNode('svg',{viewBox:'0 0 300 104',role:'img','aria-label':caption});
  const symbols=[1,1,-1,1,-1,-1,-1,1,1,-1,1,-1], t0=2, t1=8, x=t=>14+(t-t0)/(t1-t0)*272, y=v=>46-Math.max(-1.45,Math.min(1.45,v))*24, I=t=>symbols.reduce((sum,a,k)=>sum+a*pulse(t-k),0);
  svg.append(svgNode('line',{x1:14,x2:286,y1:46,y2:46,stroke:'#3a414c'}));
  for(let k=t0;k<=t1;k++)svg.append(svgNode('line',{x1:x(k),x2:x(k),y1:82,y2:87,stroke:'#748197'}));
  let d=''; for(let i=0;i<=240;i++){const t=t0+(t1-t0)*i/240;d+=(i?'L':'M')+x(t).toFixed(1)+' '+y(I(t)).toFixed(1);}
  svg.append(svgNode('path',{d,fill:'none',stroke:noise>0?'#5d6a7d':'#5794f2','stroke-width':noise>0?1.2:2}));
  const rand=seeded(7+sps);
  for(let t=t0;t<=t1+1e-9;t+=1/sps){const v=I(t)+noise*gauss(rand);svg.append(svgNode('circle',{cx:x(t).toFixed(1),cy:y(v).toFixed(1),r:2.4,fill:noise>0?'#9bd98d':'#b8d5ff'}));}
  svg.append(svgNode('text',{x:14,y:100,fill:'#a7adb7','font-size':10},caption));
  svg.append(svgNode('text',{x:286,y:100,fill:'#748197','font-size':9,'text-anchor':'end'},'ticks = symbols'));
  return svg;
}
// 8PSK, Gray-coded: neighbouring points differ by one bit.
const psk8=[['001',0],['000',1],['100',2],['110',3],['010',4],['011',5],['111',6],['101',7]].map(([bits,k])=>({bits,a:k*Math.PI/4}));
function constellationDiagram(mode) {
  const svg=svgNode('svg',{viewBox:'0 0 300 124',role:'img','aria-label':mode==='bits'?'8PSK constellation with bit labels and the packed index stream':'8PSK constellation'});
  const cx=62,cy=62,r=42;
  svg.append(svgNode('line',{x1:cx-54,x2:cx+54,y1:cy,y2:cy,stroke:'#3a414c'}));svg.append(svgNode('line',{x1:cx,x2:cx,y1:cy-54,y2:cy+54,stroke:'#3a414c'}));
  svg.append(svgNode('circle',{cx,cy,r,fill:'none',stroke:'#30353e','stroke-dasharray':'2 3'}));
  if(mode!=='bits'){svg.append(svgNode('text',{x:cx+52,y:cy-4,fill:'#748197','font-size':9,'text-anchor':'end'},'I'));svg.append(svgNode('text',{x:cx+5,y:cy-50,fill:'#748197','font-size':9},'Q'));}
  const rand=seeded(11);
  for(const p of psk8) {
    const px=cx+r*Math.cos(p.a), py=cy-r*Math.sin(p.a);
    if(mode==='noisy')for(let i=0;i<14;i++)svg.append(svgNode('circle',{cx:(px+5*gauss(rand)).toFixed(1),cy:(py+5*gauss(rand)).toFixed(1),r:1.4,fill:'#9bd98d','fill-opacity':.75}));
    svg.append(svgNode('circle',{cx:px.toFixed(1),cy:py.toFixed(1),r:mode==='noisy'?2:3.4,fill:mode==='noisy'?'#e4e5e7':'#b8d5ff'}));
    if(mode==='bits')svg.append(svgNode('text',{x:(cx+(r+11)*Math.cos(p.a)).toFixed(1),y:(cy-(r+11)*Math.sin(p.a)+3).toFixed(1),fill:'#ffc27a','font-size':9,'text-anchor':'middle'},p.bits));
  }
  const right=(text,yy,color='#c2c7d0',size=10)=>svg.append(svgNode('text',{x:136,y:yy,fill:color,'font-size':size},text));
  if(mode==='bits') {
    right('Sent over DIFI (8PSK shown):',20,'#a7adb7',9);
    ['100','110','001','011','111'].forEach((bits,i)=>{svg.append(svgNode('rect',{x:136+i*31,y:28,width:28,height:18,rx:2,fill:'#3b3024',stroke:'#ff9830','stroke-opacity':.6}));svg.append(svgNode('text',{x:150+i*31,y:41,fill:'#ffd59c','font-size':10,'text-anchor':'middle'},bits));});
    right('3 bits per symbol',64);right('instead of 32-bit I/Q;',78);right('the X410 looks up the',92);right('point and builds the IQ.',106);
  } else if(mode==='clean') {
    right('Sent over DIFI:',20,'#a7adb7',9);right('(I, Q) = (0.71, 0.71)',38,'#b8d5ff');right('(I, Q) = (0, 1)',53,'#b8d5ff');right('… one per symbol',68,'#b8d5ff');
    right('16-bit I + 16-bit Q;',92);right('the X410 shapes the pulses.',106);
  } else {
    right('Sent over DIFI:',20,'#a7adb7',9);right('one noisy IQ value',38,'#b5e6aa');right('per symbol, sampled at',53,'#b5e6aa');right('the eye opening',68,'#b5e6aa');
    right('the host still corrects',92);right('carrier phase and decodes.',106);
  }
  return svg;
}
function tipDiagram(kind) {
  if(kind==='wave-tx')return waveDiagram(4,0,'4 samples/symbol · shaped waveform (I)');
  if(kind==='wave-rx')return waveDiagram(4,.32,'4 samples/symbol · raw, full-band noise');
  if(kind==='wave-filtered')return waveDiagram(2,.14,'2 samples/symbol · matched-filtered');
  return constellationDiagram(kind==='const-bits'?'bits':kind==='const-clean'?'clean':'noisy');
}
let tipNode=null;
function showTip(stop,direction,split) {
  const tip=tips[direction+':'+split.key]; if(!tip)return;
  if(!tipNode){tipNode=document.createElement('div');tipNode.className='split-tip';tipNode.id='split-tip';tipNode.setAttribute('role','tooltip');document.body.append(tipNode);}
  const title=document.createElement('h4'); const tag=document.createElement('b'); tag.textContent=direction.toUpperCase()+' '+split.letter; title.append(tag,document.createTextNode(' · '+split.name));
  const sub=document.createElement('p'); sub.className='tip-sub'; sub.textContent=tip.what;
  const facts=document.createElement('dl');
  for(const [k,v] of [['DIFI rate',tip.rate],['Host',tip.host],['X410',tip.radio],['Switch',direction==='rx'?'Live, about 0.5 s':'Restarts both modems, about 20 s']]){const dt=document.createElement('dt');dt.textContent=k;const dd=document.createElement('dd');dd.textContent=v;facts.append(dt,dd);}
  const oran=document.createElement('p'); oran.className='oran'; const lead=document.createElement('b'); lead.textContent='O-RAN analog: '; oran.append(lead,document.createTextNode(tip.oran));
  tipNode.className='split-tip '+direction; tipNode.replaceChildren(title,sub,tipDiagram(tip.diagram),facts,oran);
  stop.setAttribute('aria-describedby','split-tip');
  const rect=stop.getBoundingClientRect(), width=tipNode.offsetWidth, height=tipNode.offsetHeight;
  const left=Math.max(8,Math.min(window.innerWidth-width-8,rect.left+rect.width/2-width/2));
  const below=rect.bottom+8+height<window.innerHeight-8;
  tipNode.style.left=left+'px'; tipNode.style.top=(below?rect.bottom+8:Math.max(8,rect.top-8-height))+'px';
  tipNode.classList.add('show');
}
function hideTip() { if(tipNode)tipNode.classList.remove('show'); }
document.addEventListener('keydown',event=>{if(event.key==='Escape')hideTip();});
const modcods = [
  ['QPSK 1/2',2],['QPSK 3/4',2],['8PSK 3/5',3],['8PSK 2/3',3],
  ['16APSK 2/3',4],['16APSK 3/4',4],['32APSK 3/4',5],['32APSK 9/10',5]
];
const modcodIds = [4,7,12,13,18,19,24,28]; // DVB-S2 MODCOD numbers in the order above
const state = {tx:2,rx:2,timing:'feedforward',bits:16,modcod:0,acm:true,orbit:true,angle:-.34,altitude:550,minModcod:0,maxModcod:7,visible:true,distance:550};
const $ = id => document.getElementById(id);
const symbolRate = 15.36; // Msymbol/s
const reference = symbolRate * 4 * 32; // 16-bit sample IQ at 4 samples/symbol
let live = null, lastPoll = 0, inflight = 0;
const pending = {}; // control -> time of the user's last change; server values wait until it settles

function hold(key) { pending[key] = performance.now(); }
function settled(key) { return !pending[key] || performance.now() - pending[key] > 2500; }
function hostOf(n) { return live && live.hosts && live.hosts[n] || {}; }
function say(text) { $('action').textContent = text; }

async function post(action, value) {
  inflight++; render();
  try {
    const response = await fetch('/api/control', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action, value})});
    const reply = await response.json();
    if (!response.ok || !reply.ok) throw new Error(reply.error || response.statusText);
    say('Applied ' + action);
  } catch (error) {
    say('Failed: ' + error.message); $('live-status').textContent = 'Control failed: ' + error.message;
  } finally { inflight--; render(); }
}

function buildRow(direction, stages, splits) {
  for (const [label, processor] of stages) {
    const isTiming=direction==='rx' && label==='Timing recovery';
    const block = document.createElement(isTiming?'button':'div'); block.className='block';
    if(isTiming) { block.type='button';block.id='timing-block';block.disabled=true; }
    const title=document.createElement('strong'); title.textContent=label;
    const owner=document.createElement('small'); owner.textContent=processor;
    block.append(title,owner);
    if(isTiming) { const mode=document.createElement('span');mode.className='timing-mode';block.append(mode); }
    $(direction+'-blocks').append(block);
  }
  splits.forEach((split,index) => {
    const stop=document.createElement('button'); stop.className='stop'+(split.blocked?' unavailable':''); stop.style.left=(split.boundary/7*100)+'%';
    stop.setAttribute('aria-label',direction.toUpperCase()+' split: '+split.name);
    stop.addEventListener('mouseenter',()=>{if(!$(direction+'-split').classList.contains('dragging'))showTip(stop,direction,split);});
    stop.addEventListener('mouseleave',hideTip);stop.addEventListener('focus',()=>showTip(stop,direction,split));stop.addEventListener('blur',hideTip);
    stop.addEventListener('click',event=>{if(event.detail===0)selectSplit(direction,index);});
    $(direction+'-stops').append(stop);
  });
  for (const [name,processor,style] of [['DGX Spark','CPU/GPU','host-owner'],['X410','FPGA/RF','fpga-owner']]) {
    const group=document.createElement('div'); group.className=style;
    group.innerHTML='<svg viewBox="0 0 100 12" preserveAspectRatio="none" aria-hidden="true"><path d="M1 0 V3 Q1 6 4 6 H46 Q50 6 50 10 Q50 6 54 6 H96 Q99 6 99 3 V0"/></svg><span><b>'+name+'</b> · '+processor+'</span>';
    $(direction+'-ownership').append(group);
  }
  const rail=$(direction+'-split'); let dragging=false, chosen=state[direction];
  function point(event) {
    const rect=rail.getBoundingClientRect();
    const position=Math.max(splits[0].boundary,Math.min(splits[splits.length-1].boundary,(event.clientX-rect.left)/rect.width*7));
    chosen=0; splits.forEach((split,index)=>{if(Math.abs(split.boundary-position)<Math.abs(splits[chosen].boundary-position))chosen=index;});
    rail.style.setProperty('--drag-position',(position/7*100)+'%');
  }
  function finish(commit) {
    if(!dragging)return;
    dragging=false;rail.classList.remove('dragging');rail.style.removeProperty('--drag-position');
    if(commit)selectSplit(direction,chosen);
  }
  rail.addEventListener('pointerdown',event=>{
    if(event.button!==0)return;
    hideTip();dragging=true;rail.classList.add('dragging');rail.setPointerCapture(event.pointerId);rail.focus();point(event);
  });
  rail.addEventListener('pointermove',event=>{if(dragging)point(event);});
  rail.addEventListener('pointerup',event=>{if(dragging){point(event);finish(true);}});
  rail.addEventListener('pointercancel',()=>finish(false));
  rail.addEventListener('lostpointercapture',()=>finish(false));
  rail.addEventListener('keydown',event=>{
    let next=state[direction];
    if(event.key==='ArrowRight'||event.key==='ArrowUp')next++;
    else if(event.key==='ArrowLeft'||event.key==='ArrowDown')next--;
    else if(event.key==='Home')next=0;
    else if(event.key==='End')next=splits.length-1;
    else return;
    event.preventDefault(); selectSplit(direction,Math.max(0,Math.min(splits.length-1,next)));
  });
}
function selectSplit(direction,index) {
  if(index===state[direction]){render();return;}
  if(inflight>0||(live&&live.busy)){say('A switch is already in progress');render();return;}
  const tx=direction==='tx'?index:state.tx, rx=direction==='rx'?index:state.rx;
  if(direction==='tx'&&txSplits[tx].blocked){$('live-status').textContent=txSplits[tx].blocked;render();return;}
  state[direction]=index;hold('split');render();
  post('split',{tx:txSplits[tx].key,rx:rxSplits[rx].key});
}
function updateRow(direction,stages,splits) {
  const split=splits[state[direction]];
  $(direction+'-pipeline').style.setProperty('--boundary',(split.boundary/7*100)+'%');
  const rail=$(direction+'-split'); rail.setAttribute('aria-valuenow',state[direction]); rail.setAttribute('aria-valuetext',split.name);
  $(direction+'-description').textContent='Split '+split.letter+' · '+split.name;
  $(direction+'-description').title=split.representation+' — '+split.detail;
  rail.setAttribute('aria-description',split.detail);
  $(direction+'-transport').replaceChildren();
  const lead=document.createElement('b'); lead.textContent='DIFI '+(direction==='tx'?'→':'←')+' '+split.representation;
  $(direction+'-transport').append(lead);
  $(direction+'-transport').title=split.detail;
  [...$(direction+'-blocks').children].forEach((block,index)=>{
    block.classList.toggle('host',index<split.boundary); block.classList.toggle('rf',index===6);
    block.querySelector('small').textContent=index===6?'RF':index<split.boundary?stages[index][1]:'FPGA';
  });
  [...$(direction+'-stops').children].forEach((button,index)=>button.setAttribute('aria-pressed',index===state[direction]));
}
function rateValue(id,value,unitName='Mb/s',digits=1) { $(id).replaceChildren(document.createTextNode(value.toFixed(digits)+' ')); const unit=document.createElement('small'); unit.textContent=unitName; $(id).append(unit); }
function blank(id,unitName) { $(id).innerHTML='— <small>'+unitName+'</small>'; }
function mean(values) { const v=values.filter(x=>typeof x==='number'&&isFinite(x)); return v.length?v.reduce((a,b)=>a+b,0)/v.length:null; }
function mbps(v) { return v===null?'—':v>=100?Math.round(v).toLocaleString('en-US'):v.toFixed(1); }

function render() {
  updateRow('tx',txStages,txSplits); updateRow('rx',rxStages,rxSplits);
  const a=hostOf('11'), b=hostOf('12'), connected=!!(a.connected&&b.connected);
  // DIFI fronthaul: measured wire rate at each Spark (both directions run the same configuration).
  const tx=mean([a.tx_mbps,b.tx_mbps]), rx=mean([a.rx_mbps,b.rx_mbps]);
  recordFronthaul(a,b,connected);
  $('fronthaul-rate').textContent=tx===null&&rx===null?'—':'TX '+mbps(tx)+' · RX '+mbps(rx)+' Mb/s';
  const ratio=v=>!v?'no telemetry':reference/v<1.1?'same as sample IQ':(reference/v).toFixed(reference/v>=10?0:1).replace(/\.0$/,'')+'× below sample IQ';
  $('fronthaul-note').textContent='TX '+txSplits[state.tx].letter+': '+ratio(tx)+' · RX '+rxSplits[state.rx].letter+' '+state.bits+'-bit: '+ratio(rx)+' · sample IQ '+Math.round(reference).toLocaleString('en-US')+' Mb/s';
  const timing=$('timing-block');timing.querySelector('.timing-mode').textContent=state.rx===0?'Feedforward':'';
  timing.title=state.rx===0?'FPGA feedforward energy-search timing recovery':'Timing recovery runs on the host';
  if(state.rx===0)$('rx-description').textContent='Split C · Timing-recovered symbols · Feedforward';
  if(live&&live.split_switch_ms&&!live.busy)$('rx-transport').title='Last live RX switch: '+live.split_switch_ms+' ms';
  const snrA=b.locked?b.snr_db:null, snrB=a.locked?a.snr_db:null; // .20 RX (link .18→.20), .18 RX
  const snr=mean([snrA,snrB]), target=live&&live.orbit?live.orbit.target_snr_db:null;
  if(snr===null)blank('snr-rate','dB');else rateValue('snr-rate',snr,'dB');
  $('snr-arc').style.strokeDasharray=(snr===null?0:Math.min(100,Math.max(0,snr/30*100)))+' 100';
  $('snr-note').textContent=!state.visible?'Link lost':snr===null?'Acquiring':'.20 '+(snrA===null?'—':snrA.toFixed(1))+' · .18 '+(snrB===null?'—':snrB.toFixed(1))+(typeof target==='number'?' · channel '+target.toFixed(1):'')+' dB';
  const fer=[a.fer,b.fer].filter(x=>typeof x==='number');
  const worst=fer.length?Math.max(...fer):null;
  if(worst===null){blank('fer-rate','%');$('fer-arc').style.strokeDasharray='0 100';}
  else {$('fer-rate').replaceChildren(document.createTextNode(worst===0?'0':worst<1e-3?(worst*100).toExponential(1):(worst*100).toFixed(2)));const u=document.createElement('small');u.textContent='%';$('fer-rate').append(u);
    $('fer-arc').style.strokeDasharray=Math.min(100,Math.max(0,worst*100))+' 100';}
  $('fer-note').textContent=fer.length===2?'.20 '+(b.fer*100).toPrecision(2)+'% · .18 '+(a.fer*100).toPrecision(2)+'%':'Waiting for receivers';
  const ping=live&&live.ping&&live.ping.rtt_ms!==null&&live.ping.age_s<3?live.ping.rtt_ms:null;
  if(ping===null)blank('ping-rate','ms');else rateValue('ping-rate',ping,'ms');
  $('ping-arc').style.strokeDasharray=(ping===null?0:Math.min(100,ping/800*100))+' 100';
  $('ping-note').textContent=ping===null?(state.visible?'Waiting for ICMP replies':'No line of sight'):(live&&live.demo==='difi'?'GEO propagation ':'Model ')+(4*state.distance/299.792458).toFixed(live&&live.demo==='difi'?0:1)+' ms';
  const goodA=b.goodput_mbps??null, goodB=a.goodput_mbps??null;
  const txIndex=modcodIds.indexOf(a.tx_modcod), txIndexB=modcodIds.indexOf(b.tx_modcod);
  if(txIndex>=0&&settled('modcod'))state.modcod=txIndex;
  const good=mean([goodA,goodB]);
  if(good===null)blank('ota-rate','Mb/s');else rateValue('ota-rate',good);
  $('ota-arc').style.strokeDasharray=(good===null?0:Math.min(100,good/20*100))+' 100';
  const modulation=modcods[state.modcod][0];
  const active=txIndex>=0?modcods[txIndex][0]+(txIndexB>=0&&txIndexB!==txIndex?' / '+modcods[txIndexB][0]:''):'—';
  $('ota-note').textContent=(state.acm?'ACM ':'CCM ')+active+' · '+(live&&live.traffic_mbps||10)+' Mb/s offered';
  $('modcod-control').hidden=state.acm;$('acm-thermometer').hidden=!state.acm;
  $('modcod').disabled=state.acm||inflight>0;
  $('ccm-toggle').checked=!state.acm;$('animation-toggle').checked=state.orbit;
  for(const [id,active] of [['acm',state.acm],['ccm',!state.acm],['animation',state.orbit],['pause',!state.orbit]])$(id+'-label').classList.toggle('active',active);
  const lowest=Math.min(snrA??99,snrB??99);
  const difi=live&&live.demo==='difi';
  if(difi)$('orbit-note').textContent='Static GEO satellite at zenith · 35,786 km · ACM follows the measured SNR';
  else $('orbit-note').textContent=!state.visible?(state.orbit?'No line of sight · outage accelerated 24×':'No line of sight · below the horizon'):state.acm&&lowest<99&&lowest<acmThresholds[state.minModcod]+acmGuard?'Below guarded ACM minimum · measured':'10-s orbit · distance scale compressed';
  $('modcod').value=state.modcod;$('modcod-manual-value').textContent=modulation;$('modcod').setAttribute('aria-valuetext',modulation);
  if(settled('altitude')){$('altitude').value=Math.round(Math.log(state.altitude/550)/Math.log(35786/550)*1000);}
  $('altitude-value').textContent=Math.round(state.altitude).toLocaleString('en-US')+' km';$('altitude').setAttribute('aria-valuetext',Math.round(state.altitude)+' kilometers');
  for(const [bound,index] of [['min',state.minModcod],['max',state.maxModcod]]) {
    $('acm-'+bound).value=index;$('acm-'+bound).setAttribute('aria-valuetext',modcods[index][0]);$('acm-'+bound+'-value').textContent=modcods[index][0];
  }
  $('modcod-fill').style.height=(state.modcod/7*100)+'%';$('modcod-indicator').style.bottom=(state.modcod/7*100)+'%';
  $('ccm-fill').style.height=(state.modcod/7*100)+'%';$('ccm-indicator').style.bottom=(state.modcod/7*100)+'%';
  $('modcod-meter').setAttribute('aria-valuenow',state.modcod);$('modcod-meter').setAttribute('aria-valuetext',modulation);
  $('acm-thermometer').classList.toggle('outage',!state.visible);
  // Anything but a running DIFI demo shows the panel: switch progress, or the idle page
  // with a start button for each demo.
  const mode=live&&live.demo_mode||{active:'difi'}, switching=!!mode.switching, idle=!switching&&mode.active!=='difi';
  document.body.classList.toggle('cellular-mode',switching||idle);
  $('demo-switch').textContent=switching?'Switching demo…':'Cellular demo';
  $('demo-switch').hidden=idle;$('demo-stop').hidden=idle;
  $('demo-switch').disabled=$('demo-stop').disabled=switching||inflight>0||(live&&live.busy&&!switching);
  $('cellular-title').textContent=switching?{cellular:'Starting the cellular demo…',difi:'Starting the DIFI demo…',stopped:'Stopping the demo…'}[mode.switching]:
    mode.active===null?'The last demo switch failed':'No demo running';
  $('panel-text').textContent=switching?'Loading the FPGA image and starting the modems; this page follows along and switches when it is ready.':
    mode.active===null?'Nothing should be transmitting. Start either demo to recover.':'The radios are idle: no modem, gNB or UE transmits.';
  $('panel-actions').hidden=switching;
  for(const id of ['start-difi','start-cellular'])$(id).disabled=switching||inflight>0;
  $('cellular-log').replaceChildren(...(mode.error?[mode.error]:mode.log||[]).slice(-8).map(line=>{const li=document.createElement('li');li.textContent=line;return li;}));
  renderProgress(mode,live&&live.server_time);
  if(live&&live.phy) {
    const mc=n=>{const i=modcodIds.indexOf(hostOf(n).tx_modcod);return i>=0?modcods[i][0]:'—';};
    $('telemetry-note').textContent='DVB-S2 · '+(live.phy.sym_rate/1e6).toFixed(2)+' Msym/s · roll-off '+live.phy.rolloff+' · .18→.20 '+mc('11')+' @ '+live.phy.tx_ghz['11']+' GHz · .20→.18 '+mc('12')+' @ '+live.phy.tx_ghz['12']+' GHz';
  }
  const applying=inflight>0||(live&&live.busy);
  for(const id of ['restart-traffic','restart-radios'])$(id).disabled=applying;
  for (const button of document.querySelectorAll('[data-bits]')) { const selected=+button.dataset.bits===state.bits; button.classList.toggle('selected',selected); button.setAttribute('aria-pressed',selected); button.disabled=applying; }
  const stale=performance.now()-lastPoll>3000;
  if(!live||stale)$('live-status').textContent='Dashboard server unreachable';
  else if(!connected)$('live-status').textContent='Radios disconnected · '+(live.last_action||'');
  else if(!applying&&!$('live-status').textContent.startsWith('Control failed')&&!$('live-status').textContent.includes('not switchable'))$('live-status').textContent='Live · X410 .18 ↔ .20 · '+(a.locked&&b.locked?'both receivers locked':'acquiring');
  if(applying)$('live-status').textContent=live&&live.last_action||'Applying…';
  // Ready: both modems reporting, no restart/switch running, both receivers
  // locked (a planned orbit outage counts as ready: only the link is blanked).
  const why=!live||stale?'no telemetry':switching?'switching demo':mode.active==='cellular'?'cellular demo running':!connected?'modem disconnected':applying?'switching':
    (!state.visible||(a.locked&&b.locked))?'':'receiver unlocked';
  $('radio-ready').classList.toggle('ready',!why);
  $('radio-ready').lastChild.textContent=why?'Radios not ready · '+why:'Radios ready';
  $('live-status').title=live?live.last_action:'';
  // Idle (or a failed switch): a neutral header, and only the panel's start buttons.
  const noDemo=live&&live.demo_mode&&!live.demo_mode.switching&&live.demo_mode.active!=='difi';
  for(const id of ['restart-traffic','restart-radios','reset'])$(id).hidden=!!noDemo||!!(live&&live.demo_mode&&live.demo_mode.switching);
  $('radio-ready').classList.toggle('idle',!!noDemo);
  if(noDemo){$('radio-ready').classList.remove('ready');$('radio-ready').lastChild.textContent=live.demo_mode.active===null?'Last switch failed':'Radios idle';$('live-status').textContent='No demo running';}
}
buildRow('tx',txStages,txSplits); buildRow('rx',rxStages,rxSplits);
for(const button of document.querySelectorAll('[data-bits]'))button.addEventListener('click',()=>{const bits=+button.dataset.bits;if(bits===state.bits)return;state.bits=bits;hold('bits');post('compression',bits===16?'plain':'trim');});
$('modcod').addEventListener('input',()=>{state.modcod=+$('modcod').value;hold('modcod');render();});
$('modcod').addEventListener('change',()=>{hold('modcod');post('modcod',modcodIds[+$('modcod').value]);});
let altitudePosted=0;
function altitudeFromSlider(){return 550*Math.pow(35786/550,+$('altitude').value/1000);}
$('altitude').addEventListener('input',()=>{state.altitude=altitudeFromSlider();hold('altitude');updateOrbit();render();const now=performance.now();if(now-altitudePosted>300){altitudePosted=now;post('altitude',state.altitude);}});
$('altitude').addEventListener('change',()=>{state.altitude=altitudeFromSlider();hold('altitude');post('altitude',state.altitude);});
for(const bound of ['min','max']){
  $('acm-'+bound).addEventListener('input',()=>{const value=+$('acm-'+bound).value;if(bound==='min')state.minModcod=Math.min(value,state.maxModcod);else state.maxModcod=Math.max(value,state.minModcod);hold('acm');render();});
  $('acm-'+bound).addEventListener('change',()=>{hold('acm');post('acm',{enabled:true,min:state.minModcod,max:state.maxModcod});});
}
$('reset').addEventListener('click',()=>{for(const key of Object.keys(pending))delete pending[key];post('reset',null);});
$('restart-traffic').addEventListener('click',()=>post('restart_traffic',null));
$('restart-radios').addEventListener('click',()=>post('restart_radios',null));
const demoMessages={
  cellular:'Start the cellular demo?\n\nThis stops the DIFI modems, loads the stock UHD image on both X410s and starts OAI, Open5GS, the LEO feeder and the Sierra UE. It takes a few minutes.',
  difi:'Start the DIFI demo?\n\nThis stops anything running, loads the LIVESPLIT image on both X410s and starts the DIFI modems. It takes a few minutes.',
  stopped:'Stop the demo?\n\nThis stops all modems (and the gNB and Sierra UE, if running) and leaves the radios idle.'};
function demoMode(target){if(window.confirm(demoMessages[target]))post('demo_mode',target);}
$('demo-switch').addEventListener('click',()=>demoMode('cellular'));
$('demo-stop').addEventListener('click',()=>demoMode('stopped'));
$('start-difi').addEventListener('click',()=>demoMode('difi'));
$('start-cellular').addEventListener('click',()=>demoMode('cellular'));
$('fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{say('Full screen unavailable; use your browser’s full-screen command.');}});
document.addEventListener('fullscreenchange',()=>{$('fullscreen').firstChild.textContent=document.fullscreenElement?'Exit full screen ':'Full screen ';});

// Orbit geometry mirrors dashboard-server.py (spherical Earth, circular orbit).
const acmThresholds=[1.80,4.43,6.92,8.11,9.17,10.41,12.93,16.05];
const acmGuard=2.5; // lib/cusat_dsp/acm_config.h
function updateOrbit() {
  const earth=6371,alt=state.altitude,radius=earth+alt;
  const horizon=Math.acos(earth/radius);
  const alpha=state.angle;
  state.visible=Math.abs(alpha)<horizon;
  const distance=Math.sqrt(earth*earth+radius*radius-2*earth*radius*Math.cos(alpha));
  state.distance=distance;
  const visualRadius=102+22*Math.log(state.altitude/550)/Math.log(35786/550);
  $('orbit-track').setAttribute('r',visualRadius);
  const x=150+visualRadius*Math.sin(alpha),y=150-visualRadius*Math.cos(alpha);
  $('satellite-position').setAttribute('transform','translate('+x.toFixed(2)+' '+y.toFixed(2)+') rotate('+(alpha*180/Math.PI+45).toFixed(2)+')');
  $('orbit-link').setAttribute('x2',x);$('orbit-link').setAttribute('y2',y);$('orbit-link').setAttribute('visibility',state.visible?'visible':'hidden');
  $('link-lost').setAttribute('visibility',state.visible?'hidden':'visible');$('link-lost').removeAttribute('hidden');
  document.querySelector('#link-lost text').textContent=state.orbit?'LINK LOST · 24× SPEED':'LINK LOST';
  document.querySelector('.orbit-scene').setAttribute('aria-label','Satellite in '+(state.orbit?'animated orbit':'paused orbit')+'; '+(state.visible?'slant distance '+distance.toFixed(0)+' km':'no line of sight, link lost')+'. Diagram not to scale.');
}
$('ccm-toggle').addEventListener('change',()=>{state.acm=!$('ccm-toggle').checked;hold('acm');render();post('acm',{enabled:state.acm,min:state.minModcod,max:state.maxModcod});});
$('animation-toggle').addEventListener('change',()=>{state.orbit=$('animation-toggle').checked;hold('animate');render();post('animate',state.orbit);});

async function poll() {
  try {
    const response=await fetch('/api/state?lite=1',{cache:'no-store'});
    live=await response.json();lastPoll=performance.now();
    const o=live.orbit,l=live.link,a=live.hosts&&live.hosts['11']||{};
    document.body.classList.toggle('difi-demo',live.demo==='difi');
    // The server serves the cellular page once that demo owns the radios.
    if(live.demo_mode&&live.demo_mode.active==='cellular'&&!live.demo_mode.switching)location.href='/';
    syncClock(live.server_time);syncEvents(live.events);
    state.angle=o.angle;
    if(settled('altitude'))state.altitude=o.altitude;
    if(settled('animate'))state.orbit=o.animate;
    if(settled('acm')){state.acm=l.acm;state.minModcod=l.min;state.maxModcod=l.max;}
    if(settled('bits')&&a.connected)state.bits=a.profile==='trim'?8:16;
    if(settled('split')&&live.split&&!live.busy){
      state.tx=Math.max(0,txSplits.findIndex(x=>x.key===live.split.tx));
      state.rx=Math.max(0,rxSplits.findIndex(x=>x.key===live.split.rx));
    }
  } catch { live=live&&{...live}; }
  setTimeout(poll,250);
}
// Piecewise angular speed matches the server: a 10-second orbit with a 24× outage.
function advanceOrbit(seconds) {
  const horizon=Math.acos(6371/(6371+state.altitude)),full=2*Math.PI;
  const normalRate=(2*horizon+(full-2*horizon)/24)/10;
  let angle=state.angle;
  if(angle < -horizon)angle+=full;
  while(seconds>1e-9) {
    const visible=angle<horizon;
    const rate=normalRate*(visible?1:24);
    const boundary=visible?horizon:full-horizon;
    const duration=(boundary-angle)/rate;
    if(seconds<duration){angle+=seconds*rate;break;}
    seconds-=duration;angle=visible?horizon:-horizon;
  }
  state.angle=angle>Math.PI?angle-full:angle;
}
// 120-s fronthaul history on the server's clock: each point sits at the middle of
// its telemetry rate window, and events at the times the server recorded, so a
// switch mark lines up with the change it caused.
const windowMs=120000, points=[];
let clockOffset=null, lastPointKey='', lastGap=0, lastFrame=performance.now(), lastRender=lastFrame;
function syncClock(serverTime) {
  if(typeof serverTime!=='number')return;
  const offset=serverTime*1000-Date.now();
  clockOffset=clockOffset===null||Math.abs(offset-clockOffset)>2000?offset:clockOffset+.1*(offset-clockOffset);
}
function local(serverTime) { return serverTime*1000-(clockOffset||0); }
function recordFronthaul(a,b,connected) {
  if(clockOffset===null)return;
  if(!connected||typeof a.sample_time!=='number'||typeof b.sample_time!=='number') {
    const now=Date.now(); if(now-lastGap>1000){points.push({t:now,tx:null,rx:null});lastGap=now;}
    return;
  }
  const key=a.sample_time+'/'+b.sample_time; if(key===lastPointKey)return; lastPointKey=key;
  points.push({t:local((a.sample_time+b.sample_time)/2),tx:mean([a.tx_mbps,b.tx_mbps]),rx:mean([a.rx_mbps,b.rx_mbps])});
  while(points.length&&points[0].t<Date.now()-windowMs-5000)points.shift();
}
function xOf(t,now) { return (t-(now-windowMs))/windowMs; }
function path(key,scale,now) {
  let open=false;
  return points.map(point=>{
    if(point[key]===null||point[key]===undefined){open=false;return '';}
    const prefix=open?'L':'M';open=true;
    return prefix+(xOf(point.t,now)*800).toFixed(2)+' '+scale(point[key]).toFixed(2);
  }).join(' ');
}
const events=new Map();
function syncEvents(list) {
  const seen=new Set();
  for(const e of list||[]) { seen.add(e.id); events.set(e.id,e); }
  for(const id of [...events.keys()])if(!seen.has(id))events.delete(id);
}
function duration(ms) { return ms<10000?(ms/1000).toFixed(2)+' s':Math.round(ms/1000)+' s'; }
// Quick operations get one mark when they took effect. Restarts also show when they
// were issued, with the outage between the two shaded.
function plotEvents(now) {
  const marks=[]; let previous=-100,row=0;
  const ordered=[...events.values()].map(e=>({...e,issued:local(e.time),done:e.applied?local(e.applied):null})).sort((x,y)=>(x.done??x.issued)-(y.done??y.issued));
  for(const e of ordered) {
    const at=e.done??e.issued, x=Math.min(100,xOf(at,now)*100), from=Math.max(0,xOf(e.issued,now)*100);
    if(x<0)continue;
    row=x-previous<16?(row+1)%3:0; previous=x;
    const span=e.done!==null&&e.done-e.issued>1500;
    if(span||e.done===null) {
      const issued=document.createElement('div'); issued.className='chart-event issued '+e.kind+(span?' with-span':'');
      issued.style.left=from+'%'; if(span)issued.style.width=(x-from)+'%';
      marks.push(issued);
    }
    if(e.done===null&&e.status==='pending'){
      const mark=document.createElement('div'); mark.className='chart-event pending '+e.kind+(x<16?' flip':'');
      mark.style.left=from+'%'; mark.style.setProperty('--row',row); mark.style.borderLeftColor='transparent';
      const label=document.createElement('span'); label.textContent=e.label+' · issued …'; mark.append(label); marks.push(mark);
      continue;
    }
    const mark=document.createElement('div'); mark.className='chart-event '+e.kind+' '+e.status+(x<16?' flip':'');
    mark.style.left=x+'%'; mark.style.setProperty('--row',row);
    const label=document.createElement('span');
    label.textContent=e.label+(e.status==='failed'?' ✕ failed':e.ms!==null&&e.ms!==undefined?' · '+duration(e.ms):'');
    label.title=span?'Issued '+duration(e.done-e.issued)+' before it took effect':'';
    mark.append(label); marks.push(mark);
  }
  $('chart-events').replaceChildren(...marks);
}
function plotHistory() {
  const now=Date.now(), fh=v=>195-Math.min(2100,Math.max(0,v))/2000*190;
  $('fh-tx').setAttribute('d',path('tx',fh,now));$('fh-rx').setAttribute('d',path('rx',fh,now));
  plotEvents(now);
}
function animate(now) {
  const elapsed=Math.min((now-lastFrame)/1000,.1);lastFrame=now;
  if(state.orbit)advanceOrbit(elapsed);
  updateOrbit();
  if(now-lastRender>=100){render();lastRender=now;}
  requestAnimationFrame(animate);
}
// A timer, not animation frames, keeps the graph moving while the tab is in the background.
setInterval(()=>{render();lastRender=performance.now();plotHistory();},250);
updateOrbit();render();plotHistory();poll();requestAnimationFrame(animate);

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
