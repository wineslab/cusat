'use strict';
// Standalone UI state and calculated animations only. No network or radio control.
const txStages = [
  ['GSE + BB framing', 'CPU'], ['BCH + LDPC encode', 'CPU / GPU'], ['Interleave + PL header', 'Host / GPU'],
  ['Constellation mapping', 'Host / GPU'], ['Pilots + PL scrambling', 'Host / GPU'], ['RRC + resampling', 'Host / GPU'], ['RF output', 'RF']
];
// Spatial ordering is host-left/radio-right; RX signal traverses right-to-left.
const rxStages = [
  ['GSE + BB recovery', 'CPU'], ['LDPC + BCH decode', 'CPU / GPU'], ['Soft demap + deinterleave', 'Host / GPU'],
  ['Carrier + frame/pilot recovery', 'Host / GPU'], ['Timing recovery', 'Host / GPU'], ['Matched filter', 'Host / GPU'], ['RF input', 'RF']
];
const txSplits = [
  {name:'Constellation indices', boundary:3, detail:'FPGA maps indices, inserts pilots and PL scrambling, then shapes the waveform.', representation:'Packed indices + PL header', type:'indices'},
  {name:'Symbol IQ', boundary:5, detail:'Host constructs PL symbols; FPGA performs pulse shaping and interpolation.', representation:'Complex PL symbols · 1 sample/symbol', type:'symbols'},
  {name:'Sample IQ', boundary:6, detail:'Host generates the shaped waveform; FPGA handles radio conversion.', representation:'Waveform IQ · 4 samples/symbol', type:'samples'}
];
const rxSplits = [
  {name:'Synchronized symbols', boundary:4, detail:'FPGA matched filtering and symbol timing recovery. Carrier recovery, frame synchronization and decoding remain on the host. Candidate route, not radio-validated.', representation:'Synchronized IQ · 1 sample/symbol', sps:1},
  {name:'Filtered IQ', boundary:5, detail:'FPGA matched filtering; planned 2 samples/symbol output. Host recovers timing and decodes.', representation:'Filtered IQ · 2 samples/symbol (planned)', sps:2},
  {name:'Sample IQ', boundary:6, detail:'Host performs matched filtering, synchronization and decoding.', representation:'Sample IQ · 4 samples/symbol', sps:4}
];
const modcods = [
  ['QPSK 1/2',2],['QPSK 3/4',2],['8PSK 2/3',3],['8PSK 3/4',3],
  ['16APSK 2/3',4],['16APSK 3/4',4],['32APSK 3/4',5],['32APSK 9/10',5]
];
const baseline = {tx:0,rx:1,timing:"feedforward",bits:16,modcod:0,snr:22,acm:true,orbit:true,angle:-.34,altitude:550,minModcod:0,maxModcod:7,visible:true};
const state = {...baseline};
const $ = id => document.getElementById(id);
const symbolRate = 15.36; // Msymbol/s. Payload budget excludes all framing/transport overhead.
const reference = symbolRate * 4 * 32;

function buildRow(direction, stages, splits) {
  for (const [label, processor] of stages) {
    const isTiming=direction==='rx' && label==='Timing recovery';
    const block = document.createElement(isTiming?'button':'div'); block.className='block';
    if(isTiming) { block.type='button';block.id='timing-block';block.addEventListener('click',()=>{state.timing=state.timing==='feedforward'?'gardner':'feedforward';render();$('action').textContent='FPGA timing: '+state.timing;}); }
    const title=document.createElement('strong'); title.textContent=label;
    const owner=document.createElement('small'); owner.textContent=processor;
    block.append(title,owner);
    if(isTiming) { const mode=document.createElement('span');mode.className='timing-mode';block.append(mode); }
    $(direction+'-blocks').append(block);
  }
  splits.forEach((split,index) => {
    const stop=document.createElement('button'); stop.className='stop'; stop.style.left=(split.boundary/7*100)+'%';
    stop.setAttribute('aria-label',direction.toUpperCase()+' split: '+split.name);
    stop.title=split.name+' — '+split.detail;
    stop.addEventListener('click',event=>{if(event.detail===0)selectSplit(direction,index);});
    $(direction+'-stops').append(stop);
  });
  for (const [name,processor,style] of [['DGX Spark','CPU/GPU','host-owner'],['X410','FPGA/RF','fpga-owner']]) {
    const group=document.createElement('div'); group.className=style;
    group.innerHTML='<svg viewBox="0 0 100 12" preserveAspectRatio="none" aria-hidden="true"><path d="M1 0 V3 Q1 6 4 6 H46 Q50 6 50 10 Q50 6 54 6 H96 Q99 6 99 3 V0"/></svg><span><b>'+name+'</b> · '+processor+'</span>';
    $(direction+'-ownership').append(group);
  }
  const rail=$(direction+'-split'); let dragging=false, pending=state[direction];
  function point(event) {
    const rect=rail.getBoundingClientRect();
    const position=Math.max(splits[0].boundary,Math.min(splits[splits.length-1].boundary,(event.clientX-rect.left)/rect.width*7));
    pending=0; splits.forEach((split,index)=>{if(Math.abs(split.boundary-position)<Math.abs(splits[pending].boundary-position))pending=index;});
    rail.style.setProperty('--drag-position',(position/7*100)+'%');
  }
  function finish(commit) {
    if(!dragging)return;
    dragging=false;rail.classList.remove('dragging');rail.style.removeProperty('--drag-position');
    if(commit)selectSplit(direction,pending);
  }
  rail.addEventListener('pointerdown',event=>{
    if(event.button!==0)return;
    dragging=true;rail.classList.add('dragging');rail.setPointerCapture(event.pointerId);rail.focus();point(event);
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
function selectSplit(direction,index) { state[direction]=index; render(); $('action').textContent=direction.toUpperCase()+' split updated in preview'; }
function updateRow(direction,stages,splits) {
  const split=splits[state[direction]];
  $(direction+'-pipeline').style.setProperty('--boundary',(split.boundary/7*100)+'%');
  const rail=$(direction+'-split'); rail.setAttribute('aria-valuenow',state[direction]); rail.setAttribute('aria-valuetext',split.name);
  $(direction+'-description').textContent=split.name;
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
function rateValue(id,value,unitName='Mb/s') { $(id).replaceChildren(document.createTextNode(value.toFixed(1)+' ')); const unit=document.createElement('small'); unit.textContent=unitName; $(id).append(unit); }
function render() {
  updateRow('tx',txStages,txSplits); updateRow('rx',rxStages,rxSplits);
  const [modulation,nBits]=modcods[state.modcod];
  const tx=state.tx===0?symbolRate*nBits:state.tx===1?symbolRate*32:reference;
  const rx=symbolRate*rxSplits[state.rx].sps*state.bits*2;
  rateValue('tx-rate',tx); rateValue('rx-rate',rx);
  $('tx-reduction').textContent=state.tx===2?'16-bit sample IQ reference':(reference/tx).toFixed(1).replace(/\.0$/,'')+'× below sample IQ';
  const timing=$('timing-block');timing.disabled=state.rx!==0;timing.classList.toggle('gardner',state.timing==='gardner');
  timing.querySelector('.timing-mode').textContent=state.rx===0?(state.timing==='gardner'?'Gardner ⇄':'Feedforward ⇄'):'';
  timing.setAttribute('aria-label',state.rx===0?'FPGA timing: '+state.timing+'. Click to switch to '+(state.timing==='gardner'?'Feedforward':'Gardner'):'Timing recovery on host; select Synchronized symbols to configure FPGA timing');
  timing.title=state.rx===0?'Click to switch timing algorithm':'Move RX boundary left to put timing recovery on FPGA';
  if(state.rx===0)$('rx-description').textContent='Synchronized symbols · '+(state.timing==='gardner'?'Gardner':'Feedforward');
  $('rx-reduction').textContent=rx===reference?'16-bit sample IQ reference':(reference/rx).toFixed(1).replace(/\.0$/,'')+'× below 16-bit sample IQ';
  for (const [name,value] of [['tx',tx],['rx',rx]]) { $(name+'-arc').style.strokeDasharray=(value/reference*100)+' 100'; }
  targets.tx=tx;targets.rx=rx;
  $('snr-metric').textContent=state.visible?state.snr.toFixed(1)+' dB':'Link lost';
  targets.snr=state.snr;
  const fraction=modulation.split(' ')[1].split('/').map(Number);
  targets.ota=state.visible?symbolRate*nBits*fraction[0]/fraction[1]:0;rateValue('ota-rate',targets.ota);
  const ping=state.visible?4*state.distance/299.792458:null;
  if(ping===null)$('ping-rate').innerHTML='— <small>ms</small>';else rateValue('ping-rate',ping,'ms');
  $('ping-arc').style.strokeDasharray=(ping===null?0:Math.min(100,ping/800*100))+' 100';
  $('modcod-control').hidden=state.acm;$('acm-thermometer').hidden=!state.acm;
  $('modcod').disabled=state.acm;
  $('ccm-toggle').checked=!state.acm;$('animation-toggle').checked=state.orbit;

  $('modcod-mode').textContent=state.acm?'ACM':'CCM';
  for(const [id,active] of [['acm',state.acm],['ccm',!state.acm],['animation',state.orbit],['pause',!state.orbit]])$(id+'-label').classList.toggle('active',active);
  $('orbit-note').textContent=!state.visible?(state.orbit?'No line of sight · outage accelerated 24×':'No line of sight · below the horizon'):state.acm&&state.snr<acmThresholds[state.minModcod]+acmGuard?'Below guarded ACM minimum · schematic':'10-s orbit · distance scale compressed';
  $('modcod').value=state.modcod;$('modcod-value').textContent=modulation;$('modcod-manual-value').textContent=modulation;$('modcod').setAttribute('aria-valuetext',modulation);
  $('altitude').value=Math.round(Math.log(state.altitude/550)/Math.log(35786/550)*1000);$('altitude-value').textContent=Math.round(state.altitude).toLocaleString('en-US')+' km';$('altitude').setAttribute('aria-valuetext',Math.round(state.altitude)+' kilometers');
  for(const [bound,index] of [['min',state.minModcod],['max',state.maxModcod]]) {
    $('acm-'+bound).value=index;$('acm-'+bound).setAttribute('aria-valuetext',modcods[index][0]);$('acm-'+bound+'-value').textContent=modcods[index][0];
  }
  $('modcod-fill').style.height=(state.modcod/7*100)+'%';$('modcod-indicator').style.bottom=(state.modcod/7*100)+'%';
  $('ccm-fill').style.height=(state.modcod/7*100)+'%';$('ccm-indicator').style.bottom=(state.modcod/7*100)+'%';
  $('modcod-meter').setAttribute('aria-valuenow',state.modcod);$('modcod-meter').setAttribute('aria-valuetext',modulation);
  $('acm-thermometer').classList.toggle('outage',!state.visible);
  for (const button of document.querySelectorAll('[data-bits]')) { const selected=+button.dataset.bits===state.bits; button.classList.toggle('selected',selected); button.setAttribute('aria-pressed',selected); }
}
let targets={tx:30.72,rx:983.04,snr:0,ota:0};
buildRow('tx',txStages,txSplits); buildRow('rx',rxStages,rxSplits);
for(const button of document.querySelectorAll('[data-bits]'))button.addEventListener('click',()=>{state.bits=+button.dataset.bits;render();$('action').textContent='RX precision updated in preview';});
$('modcod').addEventListener('input',()=>{state.modcod=+$('modcod').value;render();$('action').textContent='Manual MODCOD updated in preview';});
$('altitude').addEventListener('input',()=>{state.altitude=550*Math.pow(35786/550,+$('altitude').value/1000);updateOrbit();render();$('action').textContent='Altitude updated independently of animation';});
for(const bound of ['min','max'])$('acm-'+bound).addEventListener('input',()=>{const value=+$('acm-'+bound).value;if(bound==='min')state.minModcod=Math.min(value,state.maxModcod);else state.maxModcod=Math.max(value,state.minModcod);state.modcod=Math.max(state.minModcod,Math.min(state.maxModcod,state.modcod));updateOrbit();render();});
$('reset').addEventListener('click',()=>{Object.assign(state,baseline);updateOrbit();render();$('action').textContent='Automatic ACM baseline restored';});
$('fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{$('action').textContent='Full screen unavailable; use your browser’s full-screen command.';}});
document.addEventListener('fullscreenchange',()=>{$('fullscreen').firstChild.textContent=document.fullscreenElement?'Exit full screen ':'Full screen ';});

// Local orbit model inspired by the archived cuSAT OpenShift dashboard.
// Spherical Earth and a selectable 550-km-to-GEO circular orbit; 22 dB reference SNR.
// Only relative free-space loss is modeled. Not a calibrated RF link budget.
const acmThresholds=[1.80,4.43,6.92,8.11,9.17,10.41,12.93,16.05];
const acmGuard=2.5; // Same default guard as lib/cusat_dsp/acm_config.h.
function selectAcm() {
  if(!state.visible)return;
  let index=state.minModcod;
  for(let i=state.minModcod;i<=state.maxModcod;i++)if(state.snr>=acmThresholds[i]+acmGuard)index=i;
  state.modcod=index;
}
function updateOrbit() {
  const earth=6371,alt=state.altitude,radius=earth+alt;
  const horizon=Math.acos(earth/radius);
  const alpha=state.angle;
  state.visible=Math.abs(alpha)<horizon;
  const distance=Math.sqrt(earth*earth+radius*radius-2*earth*radius*Math.cos(alpha));
  state.distance=distance;
  const loss=20*Math.log10(distance/550);
  state.snr=state.visible?22-loss:null;
  // The circular scene is schematic; visibility uses spherical geometry above.
  const visualRadius=102+22*Math.log(state.altitude/550)/Math.log(35786/550);
  $('orbit-track').setAttribute('r',visualRadius);
  const x=150+visualRadius*Math.sin(alpha),y=150-visualRadius*Math.cos(alpha);
  $('satellite-position').setAttribute('transform','translate('+x.toFixed(2)+' '+y.toFixed(2)+') rotate('+(alpha*180/Math.PI+45).toFixed(2)+')');
  $('orbit-link').setAttribute('x2',x);$('orbit-link').setAttribute('y2',y);$('orbit-link').setAttribute('visibility',state.visible?'visible':'hidden');
  $('link-lost').setAttribute('visibility',state.visible?'hidden':'visible');$('link-lost').removeAttribute('hidden');
  document.querySelector('#link-lost text').textContent=state.orbit?'LINK LOST · 24× SPEED':'LINK LOST';
  if(state.acm)selectAcm();
  document.querySelector('.orbit-scene').setAttribute('aria-label','Satellite in '+(state.orbit?'animated orbit':'paused orbit')+'; '+(state.visible?'slant distance '+distance.toFixed(0)+' km; modeled SNR '+state.snr.toFixed(1)+' dB':'no line of sight, link lost')+'. Diagram not to scale.');
}
$('ccm-toggle').addEventListener('change',()=>{state.acm=!$('ccm-toggle').checked;updateOrbit();render();});
$('animation-toggle').addEventListener('change',()=>{state.orbit=$('animation-toggle').checked;updateOrbit();render();});
updateOrbit();render();

// Histories contain calculated values only; no invented telemetry or errors.
const history=Array.from({length:121},()=>({...targets}));
let displayed={...targets}, lastSample=performance.now(),lastFrame=lastSample,lastRender=lastSample;
const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)').matches;
function plotHistory() {
  let open=false;
  const snrPath=history.map((point,i)=>{
    if(point.snr===null){open=false;return '';}
    const prefix=open?'L':'M';open=true;
    return prefix+(i/120*800).toFixed(2)+' '+(195-(point.snr+20)/50*190).toFixed(2);
  }).join(' ');
  $('snr-trend').setAttribute('d',snrPath);
  const otaPath=history.map((point,i)=>(i?'L':'M')+(i/120*800).toFixed(2)+' '+(195-point.ota/80*190).toFixed(2)).join(' ');
  $('ota-trend').setAttribute('d',otaPath);
}
// Piecewise angular speed gives an exact 10-second orbit, including a 24x outage.
function advanceOrbit(seconds) {
  const horizon=Math.acos(6371/(6371+state.altitude)),full=2*Math.PI;
  const visibleArc=2*horizon,hiddenArc=full-visibleArc;
  const normalRate=(visibleArc+hiddenArc/24)/10;
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
function animate(now) {
  const elapsed=Math.min((now-lastFrame)/1000,.1);lastFrame=now;
  if(state.orbit)advanceOrbit(elapsed);
  updateOrbit();
  if(now-lastRender>=100){render();lastRender=now;}
  for(const name of ['snr','ota']) {
    if(targets[name]===null)displayed[name]=null;
    else if(displayed[name]===null)displayed[name]=targets[name];
    else displayed[name]+=(targets[name]-displayed[name])*(reducedMotion?1:1-Math.exp(-elapsed*7));
  }
  if(now-lastSample>=500){history.shift();history.push({...displayed});lastSample=now;plotHistory();}
  requestAnimationFrame(animate);
}
plotHistory();requestAnimationFrame(animate);
