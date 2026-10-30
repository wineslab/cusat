#!/usr/bin/env python3
"""Local DIFI demo dashboard. Telemetry and controls come from the real modems."""
import argparse, concurrent.futures, http.server, importlib.util, json, math, pathlib, re, shlex, socket, struct, subprocess, threading, time
import launcher as transport
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent/'cellular'))
import cellular as cellular_demo
BASE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('compression_demo',BASE/'compression-demo.py')
demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)
HOSTS=('11','12')
MODCODS=[(4,'QPSK 1/2'),(7,'QPSK 3/4'),(12,'8PSK 3/5'),(13,'8PSK 2/3'),(18,'16APSK 2/3'),(19,'16APSK 3/4'),(24,'32APSK 3/4'),(28,'32APSK 9/10')]
PROFILES={'plain':(0,6),'trim':(1,6),'dpcm2':(2,2),'dpcm4':(2,4),'dpcm6':(2,6),'dpcm8':(2,8)}
POOL=concurrent.futures.ThreadPoolExecutor(max_workers=4)
LOCK=threading.Lock(); CONTROL=threading.Lock()
POLL_LOCK={n:threading.Lock() for n in HOSTS}
BASELINES={}
STATE={'hosts':{},'busy':False,'last_action':'Waiting for modem telemetry','history':[],'modcods':MODCODS,'profiles':list(PROFILES)}
UI='live'
# Opt-in experiment: inject symbol-IQ noise before RF so RX timing estimators
# receive the same impaired waveform. Normal demos keep their RX-side injector.
MEASUREMENT_TX_NOISE=False
MEASUREMENT_NSYS=False
# 'difi': functional-split demo on a static GEO satellite at zenith (fixed GEO SNR and
# bent-pipe delay, automatic ACM). 'orbit': the satellite pass/altitude channel emulation.
DEMO='difi'
# Live UI: the server owns the orbit and drives the real channel. SNR follows
# the preview's relative free-space model (22 dB at 550-km zenith) through each
# modem's RX noise injector, LINK blanks RX outside line of sight, and netem on
# each DVB TAP adds the bent-pipe propagation delay so ping is measured.
EARTH_KM,LEO_KM,GEO_KM,LIGHT_KM_S,REF_SNR_DB=6371.0,550.0,35786.0,299792.458,22.0
NOIS_MAGIC,LINK_MAGIC=0x4E4F4953,0x4C494E4B
ACM_THRESHOLDS=[1.80,4.43,6.92,8.11,9.17,10.41,12.93,16.05]  # lib/cusat_dsp/acm_config.h, MODCODS order
ACM_GUARD,ACM_HYSTERESIS=2.5,0.5
TAPS={'11':'dvbs2b0','12':'dvbs2a0'};PEERS={'11':'10.30.0.1','12':'10.30.0.2'}
ORBIT={'altitude':LEO_KM,'angle':-.34,'animate':True,'visible':True,'distance_km':LEO_KM,'target_snr_db':REF_SNR_DB,'one_way_ms':0.0}
LINKCTRL={'acm':True,'min':0,'max':7,'ccm_index':0}
# Functional split of the running modems (radio_compfh tx_split/rx_split/sync).
# Default is plain DIFI both ways (TX A / RX A: sample IQ, all processing on the host).
# Filtered IQ (RX B) decimates to 2 samples/symbol in the FPGA matched filter.
BASELINE_SPLIT={'tx':'samples','rx':'samples'}
SPLIT={'tx':'samples','rx':'samples','sync':'ff','filtered_sps':2}
PROFILE={'value':'plain'}  # IQ compression to restore after a modem restart
# 8-bit trim right-shift per RX split (measured at GEO SNR): the matched-filtered and
# timing-recovered streams sit lower in the 16-bit range, so one less bit of shift
# recovers ~0.8 dB on RX B (filtered IQ at 2 sps) and ~0.5 dB on RX C.
TRIM_SHIFT={'samples':6,'filtered':5,'sync':5}
SPLIT_BINARY='/fresh/build/apps-split/dvbs2-uhd-trx'
# apps-live adds in-session RX split switching ("rxsplit" on the demo port).
# apps-live2 adds the vectorized sample-IQ TX path and 2-sps filtered IQ.
LIVE_BINARY='/fresh/build/apps-live2/dvbs2-uhd-trx'
RX_SPLIT_CODES={'sync':0,'filtered':1,'samples':2}
# Poster nomenclature: TX A sample IQ, B symbol IQ, C constellation indices;
# RX A sample IQ, B filtered IQ, C timing-recovered symbols.
TX_LETTER={'samples':'A','symbols':'B','indices':'C'};RX_LETTER={'samples':'A','filtered':'B','sync':'C'}
# Split, compression and restart events, drawn on the fronthaul throughput graph.
EVENTS=[]
# The RX noise injector adds white noise across the host sample band. In sample-IQ
# RX the host matched filter keeps only the symbol-rate band (1 of 4 samples/symbol),
# so inject 6 dB more noise there to present the same channel SNR in every split.
SAMPLE_IQ_NOISE_DB=6.02
TRAFFIC_MBPS=10
# Which demo owns the radios: 'difi' (this server's LIVESPLIT modems), 'cellular'
# (launcher.py --mode oai: stock UHD image, OAI gNB on .18 chain 1, Open5GS core on
# .12, DVB-S2 backhaul on chain 0) or 'stopped' (nothing transmits). None after a
# failed switch. Each demo's loops idle unless that demo is active.
DEMO_MODE={'active':'difi','switching':None,'log':[],'error':None}
def difi_active():return DEMO_MODE['active']=='difi' and not DEMO_MODE['switching'] and not STATE.get('starting_radios',False)
def cellular_active():return DEMO_MODE['active']=='cellular' and not DEMO_MODE['switching']
# DVB-S2 carrier: 15.36 Msym/s, roll-off 0.2, FDD (.18 sends at 2.05 GHz, .20 at 1.95 GHz).
PHY_INFO={'rolloff':0.2,'tx_ghz':{'11':2.05,'12':1.95}}
PING={'rtt_ms':None,'time':0.0,'recent':[]}
CHANNEL_SOCK=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
EPOCH={'radio':0,'ping':0,'restart_time':0.0}
RESTART=threading.Lock()
# Higher orbits assume proportionally more antenna gain/EIRP (as MEO/GEO terminals
# do), so a 550-km pass keeps its 8-22 dB swing while GEO zenith closes near
# 11 dB instead of -14 dB, which drove receivers into long deep fades.
ALTITUDE_GAIN=0.7
def link_snr(distance,altitude):
 return REF_SNR_DB-20*math.log10(distance/LEO_KM)+ALTITUDE_GAIN*20*math.log10(altitude/LEO_KM)
# Commands are generated from validated enums/numbers, never from a shell command supplied by HTTP.
UDP_CODE='''import socket,json,sys,struct
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.settimeout(4)
port=int(sys.argv[1]);message=sys.argv[2]
if port==5556:
 s.sendto(struct.pack("<If",0x4e4f4953,float(message)),("127.0.0.1",port));print('{"ok":true,"pending":true}')
else:
 s.sendto(message.encode(),("127.0.0.1",port));print(s.recv(16384).decode())
'''
def request(n,port,message):
 cmd='python3 -c '+shlex.quote(UDP_CODE)+' '+str(port)+' '+shlex.quote(str(message))
 out=subprocess.run(['ssh','-S',str(transport.SOCKETS[n]),'-o','BatchMode=yes',
     '-o','ConnectTimeout=3','-o','ServerAliveInterval=3','-o','ServerAliveCountMax=2',
     'wineslab@192.168.50.'+n,cmd],check=True,text=True,capture_output=True,timeout=10).stdout
 reply=json.loads(out)
 if not reply.get('ok'):raise RuntimeError(reply.get('error','Modem rejected request'))
 return reply

def _poll_host(n):
 now=time.time()
 try:
  value=request(n,5560,'stats');value['timestamp']=time.time();value['connected']=True
  with LOCK:previous=STATE['hosts'].get(n,{})
  dt=value['timestamp']-previous.get('timestamp',now)
  same=previous.get('connected') and value['uptime']>=previous.get('uptime',0)
  def delta(key):return max(0,value.get(key,0)-previous.get(key,0)) if same else 0
  value['sample_time']=value['timestamp']-min(dt,3)/2  # middle of the rate window
  value['rx_mbps']=delta('rx_wire_bytes')*8/max(dt,.001)/1e6
  value['tx_mbps']=delta('tx_wire_bytes')*8/max(dt,.001)/1e6
  value['goodput_mbps']=delta('pdu_bytes')*8/max(dt,.001)/1e6
  frames=delta('fec_frames');errors=delta('fec_errors')
  value['fer']=errors/frames if frames else None;value['frames_per_s']=frames/max(dt,.001)
  value['errors_per_s']=errors/max(dt,.001)
  value['clips_per_s']=delta('clips')/max(dt,.001)
  if value['compression_mode']==0 and same and .3<dt<3:
   BASELINES[n]=.2*value['rx_mbps']+.8*BASELINES.get(n,value['rx_mbps'])
  value['reference_mbps']=BASELINES.get(n)
  value['drop_delta']=sum(delta(k) for k in ('underflows','overflows','ring_drops'))
  mode=value['compression_mode'];bits=value['dbits']
  value['profile']='plain' if mode==0 else 'trim' if mode==1 else 'dpcm'+str(bits)
  # Legacy injector's two unit-variance Gaussian components add 3.0103 dB.
  p=value['noise_parameter'];value['noise_snr_db']=30-2*p-3.0103 if p>.01 else None
 except Exception as e:
  value={'connected':False,'timestamp':now,'error':str(e)[-300:]}
 with LOCK:STATE['hosts'][n]=value
 return value

def poll_host(n):
 with POLL_LOCK[n]:return _poll_host(n)

def poll_loop():
 while True:
  if not difi_active():time.sleep(1);continue
  started=time.monotonic()
  values=list(POOL.map(poll_host,HOSTS))
  with LOCK:
   STATE['history'].append({'time':time.time(),'hosts':{n:v for n,v in zip(HOSTS,values)}})
   STATE['history']=STATE['history'][-180:]
  time.sleep(max(.1,1-(time.monotonic()-started)))

def advance_orbit(angle,altitude,seconds):
 # Same piecewise rate as the UI: a 10-s orbit whose no-LoS arc runs 24x faster.
 horizon=math.acos(EARTH_KM/(EARTH_KM+altitude));full=2*math.pi
 rate=(2*horizon+(full-2*horizon)/24)/10
 if angle<-horizon:angle+=full
 while seconds>1e-9:
  visible=angle<horizon;r=rate*(1 if visible else 24);boundary=horizon if visible else full-horizon
  duration=(boundary-angle)/r
  if seconds<duration:angle+=seconds*r;break
  seconds-=duration;angle=horizon if visible else -horizon
 return angle-full if angle>math.pi else angle
def geometry(angle,altitude):
 radius=EARTH_KM+altitude;visible=abs(angle)<math.acos(EARTH_KM/radius)
 distance=math.sqrt(EARTH_KM**2+radius**2-2*EARTH_KM*radius*math.cos(angle))
 return visible,distance
def noise_parameter(snr):
 # Inverse of the injector mapping used in _poll_host; above ~27 dB the injector is off.
 return 0.0 if snr>=26.99 else (30-3.0103-max(-5.0,snr))/2
def channel(magic,value):
 packet=struct.pack('<If',magic,float(value))
 for n in HOSTS:CHANNEL_SOCK.sendto(packet,('192.168.50.'+n,5556))
def set_delay(n,one_way_ms):
 tap=TAPS[n];cmd=f'docker exec cusat-compression tc qdisc replace dev {tap} root netem delay {one_way_ms:.2f}ms limit 100000'
 subprocess.run(['ssh','-S',str(transport.SOCKETS[n]),'-o','BatchMode=yes','wineslab@192.168.50.'+n,cmd],check=True,capture_output=True,timeout=10)

def orbit_loop():
 last=time.monotonic();sent={'link':None,'noise':None,'link_time':0.0,'epoch':-1}
 while True:
  time.sleep(.1);now=time.monotonic();dt=min(now-last,.5);last=now
  if not difi_active():continue
  with LOCK:
   if sent['epoch']!=EPOCH['radio']:sent.update(link=None,noise=None,epoch=EPOCH['radio'])
   if ORBIT['animate']:ORBIT['angle']=advance_orbit(ORBIT['angle'],ORBIT['altitude'],dt)
   visible,distance=geometry(ORBIT['angle'],ORBIT['altitude'])
   snr=link_snr(distance,ORBIT['altitude'])
   ORBIT.update(visible=visible,distance_km=distance,target_snr_db=snr if visible else None,one_way_ms=2*distance/LIGHT_KM_S*1000)
   # Acquire at the normal cabled RF SNR before applying experimental TX
   # noise. This also leaves route/startup transients out of steady-state tests.
   acquiring=MEASUREMENT_TX_NOISE and STATE['busy']
   injected=40.0 if acquiring else snr-(SAMPLE_IQ_NOISE_DB if SPLIT['rx']=='samples' and not MEASUREMENT_TX_NOISE else 0.0)
  try:
   if visible!=sent['link'] or now-sent['link_time']>1:channel(LINK_MAGIC,1.0 if visible else 0.0);sent.update(link=visible,link_time=now)
   parameter=noise_parameter(injected)
   # While adopting an existing packed-index modem, wait for the experiment's
   # IQ-TX restart; packed constellation indices cannot carry additive noise.
   if not (MEASUREMENT_TX_NOISE and SPLIT['tx']=='indices') and (sent['noise'] is None or abs(parameter-sent['noise'])>.02):channel(NOIS_MAGIC,parameter);sent['noise']=parameter
  except OSError as e:print('channel command failed:',e,flush=True)

def delay_loop():
 applied={};epoch=-1
 while True:
  time.sleep(.5)
  if not difi_active():applied.clear();continue
  with LOCK:
   target=round(ORBIT['one_way_ms']*2)/2
   if epoch!=EPOCH['radio']:applied.clear();epoch=EPOCH['radio']
  for n in HOSTS:
   if applied.get(n)==target:continue
   try:set_delay(n,target);applied[n]=target
   except (subprocess.SubprocessError,OSError):applied.pop(n,None)

def acm_loop():
 # Host-side ACM: each transmitter follows the peer receiver's pilot SNR. Telemetry is
 # 1 Hz while a 10-s orbit moves fast, so the controller also uses the channel's
 # instantaneous geometric target (ephemeris-aided ACM) and takes the lower of the two.
 commanded={};epoch=-1
 while True:
  time.sleep(.25)
  if not difi_active():commanded.clear();continue
  with LOCK:
   control=dict(LINKCTRL);visible=ORBIT['visible'];orbit=dict(ORBIT);hosts=json.loads(json.dumps(STATE['hosts']))
   if epoch!=EPOCH['radio']:commanded.clear();epoch=EPOCH['radio']  # fresh modems start at the config MODCOD
  if not control['acm']:commanded.clear();continue
  if not visible:continue
  # Lowest geometric SNR over the next second of the (known) trajectory.
  angle,predicted=orbit['angle'],[]
  for _ in range(5):
   seen,distance=geometry(angle,orbit['altitude'])
   if seen:predicted.append(link_snr(distance,orbit['altitude']))
   if orbit['animate']:angle=advance_orbit(angle,orbit['altitude'],.25)
  ids=[m[0] for m in MODCODS]
  for n in HOSTS:
   peer=HOSTS[1-HOSTS.index(n)];rx=hosts.get(peer,{});tx=hosts.get(n,{})
   if not (rx.get('connected') and rx.get('locked') and tx.get('connected')) or rx.get('snr_db') is None:continue
   snr=min([rx['snr_db']]+predicted)
   last=commanded.get(n,tx.get('tx_modcod'));current=ids.index(last) if last in ids else control['min']
   index=control['min']
   for i in range(control['min'],control['max']+1):
    if snr>=ACM_THRESHOLDS[i]+ACM_GUARD+(ACM_HYSTERESIS if i>current else 0):index=i
   if ids[index]!=last:
    try:request(n,5560,'modcod '+str(ids[index]));commanded[n]=ids[index]
    except (subprocess.SubprocessError,OSError,RuntimeError,ValueError) as e:print('ACM modcod failed on',n,e,flush=True)

def ping_loop():
 # Measured RTT over the full radio path: Spark .11 TAP -> .18 -> .20 -> Spark .12 TAP and back.
 while True:
  if not difi_active():time.sleep(1);continue
  try:
   with LOCK:epoch=EPOCH['ping']
   process=subprocess.Popen(['ssh','-S',str(transport.SOCKETS['11']),'-o','BatchMode=yes','wineslab@192.168.50.11','exec ping -n -O -i 0.5 -W 2 '+PEERS['11']],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
   def restart_on_epoch():
    while process.poll() is None:
     time.sleep(.5)
     with LOCK:changed=EPOCH['ping']!=epoch or not difi_active()
     if changed:process.kill()
   threading.Thread(target=restart_on_epoch,daemon=True).start()
   # -O reports an outstanding probe at the next send (500 ms), before
   # the GEO RTT has necessarily elapsed. Count it only after the 2-s timeout,
   # and let a reply replace its pending entry instead of adding a false loss.
   pending,results={},{}
   for line in process.stdout:
    match=re.search(r'time=([0-9.]+) ms',line)
    sequence=re.search(r'icmp_seq[= ](\d+)',line)
    if not sequence:continue
    seq=int(sequence.group(1));now=time.monotonic()
    if match:
     pending.pop(seq,None);results[seq]=float(match.group(1))
    elif 'no answer yet' in line:
     pending.setdefault(seq,now-.5)
    else:continue
    for probe,sent in list(pending.items()):
     if now-sent>=2:
      results[probe]=None;pending.pop(probe)
    results={probe:results[probe] for probe in sorted(results)[-20:]}
    with LOCK:
     PING['recent']=[results[probe] for probe in sorted(results)]
     if match:PING['rtt_ms']=float(match.group(1));PING['time']=time.time()
  except OSError as e:print('ping failed:',e,flush=True)
  time.sleep(2)

def apply(action,value):
 if action=='demo_mode':
  if value not in ('difi','cellular','stopped'):raise ValueError('Unknown demo')
  if DEMO_MODE['switching']:raise ValueError('A demo switch is already running')
  if value==DEMO_MODE['active']:return {'ok':True}
  if RESTART.locked():raise ValueError('A restart is already running')
  def work():
   with LOCK:DEMO_MODE.update(switching=value,error=None,log=[])
   begin_progress(value)
   try:
    {'cellular':to_cellular,'difi':to_difi,'stopped':to_stopped}[value]()
    with LOCK:DEMO_MODE['active']=value
   except Exception as e:
    with LOCK:DEMO_MODE.update(active=None,error=str(e)[-300:])
    raise
   finally:
    with LOCK:DEMO_MODE['switching']=None
  if value=='stopped':background('the demo',work,('Stopping','Stopped','Stopping'),mark=('restart','Demo stopped'))
  else:background('to the '+('cellular' if value=='cellular' else 'DIFI')+' demo',work,('Moving radios','Moved radios','Moving radios'),
                  mark=('restart','Cellular demo' if value=='cellular' else 'DIFI demo'))
  return {'ok':True,'pending':True}
 if action in ('cell_restart_ue','cell_restart','cell_animate'):
  if not cellular_active():raise ValueError('The cellular demo is not running')
  if action=='cell_animate':
   if type(value) is not bool:raise ValueError('Animate must be true or false')
   with LOCK:CELL.orbit['animate']=value
   return {'ok':True}
  if RESTART.locked():raise ValueError('A restart is already running')
  if action=='cell_restart_ue':background('UE session',lambda:CELL.restart_ue(switch_log),mark=('restart','UE session'))
  else:
   def again():
    CELL.hold=True  # keep the satellite at zenith while the feeder modems restart
    try:run_logged(['python3','-u','launcher.py','--mode','oai','--detach'])
    finally:CELL.hold=False
    CELL.setup(switch_log)
   background('cellular stack',again,mark=('restart','Cellular restart'))
  return {'ok':True,'pending':True}
 if not difi_active():raise ValueError('The DIFI demo is not running')
 if action in ('restart_traffic','restart_radios'):
  if RESTART.locked():raise ValueError('A restart is already running')
  background('traffic' if action=='restart_traffic' else 'radios',start_traffic if action=='restart_traffic' else start_radios,
             mark=('restart','Traffic restart' if action=='restart_traffic' else 'Radio restart'))
  return {'ok':True,'pending':True}
 if action=='split':
  if type(value) is not dict:raise ValueError('Split needs tx, rx and sync')
  tx,rx,sync=value.get('tx',SPLIT['tx']),value.get('rx',SPLIT['rx']),value.get('sync',SPLIT['sync'])
  if tx not in ('indices','symbols','samples') or rx not in ('sync','filtered','samples') or sync not in ('ff','gardner'):raise ValueError('Unknown split')
  if MEASUREMENT_TX_NOISE and tx!='symbols':raise ValueError('The TX-noise measurement requires symbol-IQ TX (TX B)')
  if tx=='samples' and rx!='samples' and demo.BINARY!=LIVE_BINARY:raise ValueError('Sample-IQ TX with filtered or symbol RX needs the live build (RX starts at sample IQ, then switches)')
  if demo.BINARY not in (SPLIT_BINARY,LIVE_BINARY) and (tx,rx)!=('indices','filtered'):raise ValueError('Split switching needs the split-capable modem build (apps-split)')
  if RESTART.locked():raise ValueError('A restart is already running')
  previous=dict(SPLIT)
  if demo.BINARY==LIVE_BINARY and not MEASUREMENT_NSYS and tx==SPLIT['tx'] and (rx!='sync' or sync=='ff'):
   # In-session RX switch: both modems re-route and swap frontends; confirm
   # through their stats, else fall back to the restart path below.
   def live():
    started=time.monotonic()
    if not live_rx(rx):
     with LOCK:SPLIT.update(rx=rx,sync=sync)
     start_radios();raise RuntimeError('live RX switch not confirmed; restarted radios instead')
    applied_event()
    with LOCK:SPLIT.update(rx=rx,sync=sync);STATE['split_switch_ms']=round((time.monotonic()-started)*1000)
   background('RX to '+rx,live,('Switching','Switched','Switch'),mark=('rx','RX: '+RX_LETTER[rx]))
   return {'ok':True,'pending':True}
  def work():
   with LOCK:SPLIT.update(tx=tx,rx=rx,sync=sync)
   try:start_radios()
   except Exception:
    for n in HOSTS:
     try:print(f'--- modem {n} log before fallback ---\n'+transport.ssh(n,'docker logs --tail 40 '+demo.CONTAINER+' 2>&1'),flush=True)
     except Exception:pass
    # Fall back to the last working route so the demo keeps running.
    with LOCK:SPLIT.update(previous)
    start_radios()
    raise
  label=' · '.join(([('TX: '+TX_LETTER[tx])] if tx!=previous['tx'] else [])+([('RX: '+RX_LETTER[rx])] if rx!=previous['rx'] or sync!=previous['sync'] else []))
  background('split TX '+tx+' / RX '+rx+(' ('+sync+')' if rx=='sync' else ''),work,mark=('tx' if tx!=previous['tx'] else 'rx',label or 'Split restart'))
  return {'ok':True,'pending':True}
 if action=='acm':
  if type(value) is not dict or type(value.get('enabled')) is not bool:raise ValueError('ACM needs enabled/min/max')
  low,high=value.get('min',LINKCTRL['min']),value.get('max',LINKCTRL['max'])
  if type(low) is not int or type(high) is not int or not 0<=low<=high<len(MODCODS):raise ValueError('Invalid ACM bounds')
  with LOCK:LINKCTRL.update(acm=value['enabled'],min=low,max=high);STATE['last_action']=('ACM '+MODCODS[low][1]+' – '+MODCODS[high][1]) if value['enabled'] else 'CCM'
  if not value['enabled']:return apply('modcod',MODCODS[LINKCTRL['ccm_index']][0])
  return {'ok':True}
 if action in ('altitude','animate') and DEMO=='difi':raise ValueError('The GEO satellite is fixed in the DIFI demo')
 if action in ('acm','modcod') and DEMO=='difi':raise ValueError('MODCOD follows ACM automatically in the DIFI demo')
 if action=='altitude':
  if type(value) not in (int,float) or not math.isfinite(value) or not LEO_KM<=value<=GEO_KM:raise ValueError('Altitude must be 550–35786 km')
  with LOCK:ORBIT['altitude']=float(value)
  return {'ok':True}
 if action=='animate':
  if type(value) is not bool:raise ValueError('Animate must be true or false')
  with LOCK:ORBIT['animate']=value
  return {'ok':True}
 if action=='reset':
  with LOCK:
   if DEMO=='difi':ORBIT.update(altitude=GEO_KM,angle=0.0,animate=False)
   else:ORBIT.update(altitude=LEO_KM,angle=-.34,animate=True)
   LINKCTRL.update(acm=True,min=0,max=7,ccm_index=0)
  with LOCK:profile=STATE['hosts'].get('11',{}).get('profile');baseline=all(SPLIT[k]==v for k,v in BASELINE_SPLIT.items())
  if profile!='plain':apply('compression','plain')
  if not baseline:apply('split',dict(BASELINE_SPLIT))
  with LOCK:STATE['last_action']='Baseline restored: '+('static GEO link' if DEMO=='difi' else 'animated 550-km orbit')+', ACM, 16-bit IQ, TX A / RX A'
  return {'ok':True}
 if action=='compression':
  if value not in PROFILES:raise ValueError('Unknown compression profile')
  if demo.IMAGE=='livesplit' and value.startswith('dpcm'):raise ValueError('DPCM is compiled out of the LIVESPLIT image')
  mode,bits=PROFILES[value];port=5558;message=f'compression {mode} {TRIM_SHIFT[SPLIT["rx"]] if value=="trim" else 6} {bits}'
 elif action=='modcod':
  if type(value) is not int or value not in dict(MODCODS):raise ValueError('Unsupported MODCOD')
  with LOCK:
   if UI=='live' and LINKCTRL['acm']:raise ValueError('Switch to CCM before choosing a MODCOD')
   LINKCTRL['ccm_index']=[m[0] for m in MODCODS].index(value)
  port=5560;message='modcod '+str(value)
 elif action=='snr':
  if UI=='live':raise ValueError('SNR follows the satellite geometry in the live UI')
  if value is None:parameter=0
  elif type(value) not in (float,int) or not math.isfinite(value) or not 2<=value<=26:raise ValueError('SNR must be 2–26 dB or null (off)')
  else:parameter=(30-3.0103-value)/2
  port=5556;message=str(parameter)
 else:raise ValueError('Unknown control')
 with CONTROL:
  with LOCK:STATE['busy']=True;STATE['last_action']='Applying '+action+' on both radios…'
  errors={}
  marked=event('iq','IQ: '+{'plain':'16-bit','trim':'8-bit'}.get(value,value)) if action=='compression' else None
  try:
   futures={n:POOL.submit(request,n,port,message) for n in HOSTS}
   for n,f in futures.items():
    try:f.result()
    except Exception as e:errors[n]=str(e)
   # Verify applied state rather than presenting desired state as actual.
   deadline=time.monotonic()+6
   while not errors and time.monotonic()<deadline:
    values=list(POOL.map(poll_host,HOSTS))
    if all(v.get('connected') and ((action=='compression' and v.get('profile')==value) or
       (action=='modcod' and v.get('tx_modcod')==value) or
       (action=='snr' and (v.get('noise_snr_db') is None if value is None else abs(v.get('noise_snr_db',-99)-value)<.05))) for v in values):break
    time.sleep(.3)
   else:
    if not errors:errors={'verification':'Applied state could not be confirmed on both radios'}
   with LOCK:STATE['last_action']=('Failed: '+json.dumps(errors)) if errors else 'Applied '+action+' · '+str(value if value is not None else 'noise off')
   if marked:finish_event(marked,not errors)
   if action=='compression' and not errors:PROFILE['value']=value
   if errors:raise RuntimeError(json.dumps(errors))
   return {'ok':True}
  finally:
   with LOCK:STATE['busy']=False

def live_rx(rx,boot=False):
 # In-session RX switch on both modems, confirmed through their stats. Right after
 # a restart, wait for the modems' control ports to answer first.
 deadline=time.monotonic()+(40 if boot else 6)
 # After a boot, switch once both receivers have locked on the boot route (the
 # proven live-switch path), or after 25 s.
 locked_by=time.monotonic()+25
 while boot and time.monotonic()<min(deadline,locked_by) and not all(v.get('connected') and v.get('locked') for v in POOL.map(poll_host,HOSTS)):time.sleep(.5)
 for n in HOSTS:
  for attempt in range(6):
   try:request(n,5560,'rxsplit '+rx);break
   except (subprocess.SubprocessError,OSError,ValueError,RuntimeError):
    if attempt==5 or time.monotonic()>deadline:raise
    time.sleep(1)
 deadline=max(deadline,time.monotonic()+6)
 values=None
 while time.monotonic()<deadline:
  values=dict(zip(HOSTS,POOL.map(poll_host,HOSTS)))
  if all(v.get('rx_split')==RX_SPLIT_CODES[rx] for v in values.values()):break
  time.sleep(.1)
 else:return False
 # The switch counts once RX data flows again on both radios (a re-route can
 # leave the RX stream stalled; the caller then restarts the modems).
 before={n:values[n].get('rx_wire_bytes',0) for n in HOSTS};flow_by=time.monotonic()+3
 while time.monotonic()<flow_by:
  time.sleep(.3)
  if all(v.get('rx_wire_bytes',0)>before[n] for n,v in zip(HOSTS,POOL.map(poll_host,HOSTS))):break
 else:return False
 if PROFILE['value']=='trim':
  for n in HOSTS:request(n,5558,'compression 1 '+str(TRIM_SHIFT[rx])+' 6')
 return True
def boot_rx():
 # Sample-IQ TX needs the 4-sps host rate at startup, so the modems boot RX at
 # sample IQ and switch live to filtered IQ or symbols once they run.
 tx,rx=SPLIT['tx'],SPLIT['rx']
 return 'samples' if tx=='samples' and rx!='samples' and demo.BINARY==LIVE_BINARY else rx

def split_radio(text):
 # Rewrite compression-demo's radio section for the selected functional split.
 # Host rates: sample IQ and filtered IQ at 4 and 2 samples/symbol, synchronized
 # symbols at 1; TX symbols/indices leave the host at the symbol rate.
 tx,rx,sync=SPLIT['tx'],boot_rx(),SPLIT['sync']
 sym=demo.PHY['sym_rate'];sps={'samples':4,'filtered':SPLIT['filtered_sps'],'sync':1}[rx]
 # sym_rate is explicit: sample-IQ TX gives the driver no symbol-rate TX stream.
 args=',tx_split='+tx+',rx_split='+rx+(',sync='+sync if rx=='sync' else '')+',sym_rate='+str(int(sym))+',filtered_sps='+str(SPLIT['filtered_sps'])
 if rx=='filtered' and sps==2:text=text.replace(',p=1,q=1',',p=1,q=2')
 text=text.replace(',shift='+str(demo.PHY['shift'])+',',',shift='+str(TRIM_SHIFT[rx])+',')
 text=text.replace(',issue=streamer',',issue=streamer'+args)
 text=text.replace('  tx_modcomp: true\n',{'indices':'  tx_modcomp: true\n','symbols':'  tx_symbols: true\n','samples':''}[tx])
 text=text.replace('  rx_prefiltered: true\n',{'filtered':'  rx_prefiltered: true\n','samples':'','sync':'  rx_symbol_sync: true\n'}[rx])
 if demo.BINARY==LIVE_BINARY:text=text.replace('  rx_gain: 20\n','  rx_gain: 20\n  rx_live_split: true\n  rx_filtered_sps: '+str(SPLIT['filtered_sps'])+'\n')
 return re.sub(r'\n  rate: [0-9.e+]+\n','\n  rate: '+repr(float(sps*sym))+'\n',text)

def install_config():
 if getattr(demo.config,'dashboard',False):return
 original=demo.config
 def config(n,profile):
  return split_radio(original(n,profile)).replace('  modcod: qpsk1/2','  modcod: qpsk1/2\n  manual: true')+'''channel:
  noise: true
  side: '''+('tx' if MEASUREMENT_TX_NOISE else 'rx')+'''
  noise_port: 5556
demo:
  control_port: 5560
'''
 config.dashboard=True;demo.config=config

def start_traffic():
 # iperf3 servers/clients and the measured ping; safe to repeat on running modems.
 for n in HOSTS:
  peer=PEERS[n]
  transport.ssh(n,'tmux kill-window -t cusat-compression:traffic 2>/dev/null; tmux kill-window -t cusat-compression:iperf-server 2>/dev/null; docker exec cusat-compression pkill iperf3 || true')
  transport.ssh(n,'tmux new-window -d -t cusat-compression -n iperf-server '+shlex.quote('docker exec -it cusat-compression iperf3 -s --forceflush --idle-timeout 10 --rcv-timeout 10000'))
 for n in HOSTS:
  peer=PEERS[n]
  for _ in range(30):
   try:transport.ssh(n,'docker exec cusat-compression ping -c 2 -W 1 '+peer);break
   except subprocess.CalledProcessError:time.sleep(1)
  else:raise RuntimeError('Radio peer did not answer on Spark '+n)
  command='while true; do docker exec cusat-compression iperf3 -c '+peer+' -u -b '+str(TRAFFIC_MBPS)+'M -l 1200 -t 86400 -i 1 --forceflush --connect-timeout 3000 || true; sleep 2; done > '+demo.SOURCE+'/dashboard-iperf-client.log 2>&1'
  transport.ssh(n,'tmux new-window -d -t cusat-compression -n traffic '+shlex.quote(command))
 with LOCK:EPOCH['ping']+=1

def start_radios():
 # Keep channel/ACM/netem/watchdog loops from racing device initialization.
 with LOCK:STATE['starting_radios']=True
 try:
  demo.stop()
  profile=PROFILE['value']
  list(POOL.map(lambda n:demo.endpoint(n,profile),HOSTS))
  with LOCK:EPOCH['radio']+=1  # fresh TAPs and modems: reapply netem, LINK and noise
  if boot_rx()!=SPLIT['rx']:
   if not live_rx(SPLIT['rx'],boot=True):raise RuntimeError('modems up, but the RX switch to '+SPLIT['rx']+' was not confirmed')
  applied_event()  # the new modems carry the new route; traffic restarts next
  start_traffic()
  # Client processes start asynchronously. Keep the clean startup channel
  # until both directions actually deliver data, before ACM/noise/netem resume.
  deadline=time.monotonic()+30;flowing=0
  while time.monotonic()<deadline:
   values=list(POOL.map(poll_host,HOSTS))
   flowing=flowing+1 if all(v.get('connected') and (v.get('goodput_mbps') or 0)>1 for v in values) else 0
   if flowing>=3:break
   time.sleep(1)
  else:raise RuntimeError('DIFI modems started, but bidirectional traffic did not establish')
 finally:
  with LOCK:STATE['starting_radios']=False

# Each event records when it was issued and when it took effect on the radios
# (a restart's new modems are up; a live switch or compression is confirmed).
CURRENT_MARK={'mark':None}
def event(kind,label):
 now=time.time()
 with LOCK:
  mark={'id':'%.6f'%now,'time':now,'applied':None,'start':time.monotonic(),'kind':kind,'label':label,'status':'pending','ms':None}
  EVENTS.append(mark);del EVENTS[:-40]
 return mark
def applied_event(mark=None):
 mark=mark or CURRENT_MARK['mark']
 if mark:
  with LOCK:
   if mark['applied'] is None:mark.update(applied=time.time(),ms=round((time.monotonic()-mark['start'])*1000))
def finish_event(mark,ok):
 if ok:applied_event(mark)
 with LOCK:mark.update(status='ok' if ok else 'failed')

def background(name,work,verb=('Restarting','Restarted','Restart'),mark=None):
 # Restarts take tens of seconds; run them off the HTTP thread, one at a time.
 def run():
  if not RESTART.acquire(blocking=False):return
  marked=event(*mark) if mark else None
  CURRENT_MARK['mark']=marked
  try:
   with LOCK:STATE['busy']=True;STATE['last_action']=verb[0]+' '+name+'…'
   work()
   with LOCK:STATE['last_action']=verb[1]+' '+name+(' in %d ms'%STATE['split_switch_ms'] if verb[0]=='Switching' and STATE.get('split_switch_ms') else '')
   if marked:finish_event(marked,True)
  except Exception as e:
   with LOCK:STATE['last_action']=verb[2]+' '+name+' failed: '+str(e)[-200:]
   if marked:finish_event(marked,False)
  finally:
   with LOCK:STATE['busy']=False;EPOCH['restart_time']=time.time()
   CURRENT_MARK['mark']=None
   RESTART.release()
 threading.Thread(target=run,daemon=True).start()

def watchdog_loop():
 # A receiver driven through long deep fades can stay wedged after the channel
 # recovers; restart both modems if one stays unlocked under a good channel.
 unlocked={};stalled={};starved={}  # distinguish IQ delivery from decoded application traffic
 while True:
  time.sleep(2)
  # CUDA measurement deliberately closes each modem after its sample window
  # to flush the trace. The coordinator owns its next restart.
  if not difi_active() or MEASUREMENT_NSYS:unlocked.clear();stalled.clear();starved.clear();continue
  with LOCK:
   good=ORBIT['visible'] and (ORBIT['target_snr_db'] or -99)>=8;hosts=json.loads(json.dumps(STATE['hosts']));recent=time.time()-EPOCH['restart_time']<60;busy=STATE['busy']
  now=time.time()
  for n in HOSTS:
   if good and hosts.get(n,{}).get('connected') and not hosts[n].get('locked'):unlocked.setdefault(n,now)
   else:unlocked.pop(n,None)
   if hosts.get(n,{}).get('connected') and (hosts[n].get('rx_mbps') or 0)<1:stalled.setdefault(n,now)
   else:stalled.pop(n,None)
   if good and hosts.get(n,{}).get('connected') and (hosts[n].get('goodput_mbps') or 0)<.1:starved.setdefault(n,now)
   else:starved.pop(n,None)
  settled=time.time()-EPOCH['restart_time']>20
  app_stalled=settled and any(now-t>12 for t in starved.values())
  if not busy and (not recent and any(now-t>12 for t in unlocked.values()) or settled and any(now-t>6 for t in stalled.values()) or app_stalled):
   print('watchdog: receiver '+('application traffic stalled' if app_stalled else 'RX stream stalled' if stalled else 'unlocked under a good channel')+'; restarting radios',flush=True)
   unlocked.clear();stalled.clear();starved.clear();background('radios (watchdog)',start_radios,mark=('restart','Watchdog restart'))

def both_locked(timeout):
 deadline=time.monotonic()+timeout
 while time.monotonic()<deadline:
  if all(v.get('connected') and v.get('locked') for v in POOL.map(poll_host,HOSTS)):return True
  time.sleep(2)
 return False
def start():
 errors=demo.preflight()
 if errors:raise RuntimeError('; '.join(errors))
 demo.stop(include_previous=True)
 install_config()
 # Switch helper checks identity and skips a reload when the image is present.
 import contextlib
 with contextlib.redirect_stdout(LogTee()):demo.fpga_switch.switch(demo.IMAGE)
 switch_log('Starting the DIFI modems and traffic')
 start_radios()
 # The first start after an FPGA load can leave one receiver unlocked; restart once
 # rather than waiting for the watchdog.
 switch_log('Waiting for both receivers to lock')
 if not both_locked(45):
  print('start: a receiver did not lock; restarting the modems once',flush=True)
  start_radios()
 # Laptop console keeps radio logs accessible beside the dashboard server.
 windows=subprocess.run(['tmux','list-windows','-t','cusat-dashboard','-F','#{window_name}'],capture_output=True,text=True)
 if windows.returncode==0 and 'radios' not in windows.stdout.split():
  for index,n in enumerate(HOSTS):
   remote='docker logs --tail 15 -f cusat-compression'
   command=shlex.join(['ssh','-tt','-S',str(transport.SOCKETS[n]),'wineslab@192.168.50.'+n,remote])
   args=['tmux','new-window','-d','-t','cusat-dashboard','-n','radios',command] if not index else ['tmux','split-window','-h','-d','-t','cusat-dashboard:radios',command]
   subprocess.run(args,check=True)
  subprocess.run(['tmux','set-option','-t','cusat-dashboard','pane-border-status','top'],check=True)
 print('Bidirectional 10 Mbit/s application traffic started.',flush=True)

CELL=cellular_demo.Cellular(transport,cellular_active,LOCK)

# Steps of each demo transition with their expected seconds (measured on the bench).
# A step begins when its marker appears in the switch log; a step whose marker never
# shows (an FPGA image already in place) is skipped once a later one appears.
PHASES={
 'difi':[('Stopping the running demo',15,None),
  ('Loading the LIVESPLIT image on X410 .18',40,r'Loading verified livesplit FPGA on 192\.168\.40\.18'),
  ('Loading the LIVESPLIT image on X410 .20',40,r'Loading verified livesplit FPGA on 192\.168\.40\.20'),
  ('Starting the DIFI modems and traffic',35,r'^Starting the DIFI modems'),
  ('Waiting for both receivers to lock',15,r'^Waiting for both receivers')],
 'cellular':[('Stopping the running demo',10,None),
  ('Loading the stock UHD image on X410 .18',35,r'Loading verified stock FPGA on 192\.168\.40\.18'),
  ('Loading the stock UHD image on X410 .20',35,r'Loading verified stock FPGA on 192\.168\.40\.20'),
  ('Starting the feeder modems, core and gNB',25,r'Restarting oai demo'),
  ('Checking the feeder and the core',15,r'OAI started on'),
  ('Attaching the Sierra UE',25,r'UE iperf servers started')],
 'stopped':[('Stopping the modems, gNB and UE',15,None)]}
def begin_progress(target):
 now=time.time()
 with LOCK:DEMO_MODE['progress']={'phases':[[label,expected] for label,expected,_ in PHASES[target]],'index':0,'started':now,'phase_started':now,'target':target}
def progress_line(line):
 with LOCK:
  p=DEMO_MODE.get('progress')
  if not p or not DEMO_MODE['switching']:return
  for k,(_,_,marker) in enumerate(PHASES[p['target']]):
   if k>p['index'] and marker and re.search(marker,line):p.update(index=k,phase_started=time.time())
class LogTee:
 # Copies in-process prints (the FPGA loader) into the switch log for the progress bar.
 def __init__(self):self.buffer=''
 def write(self,text):
  sys.__stdout__.write(text);self.buffer+=text
  while '\n' in self.buffer:
   line,self.buffer=self.buffer.split('\n',1)
   if line.strip():switch_log(line.strip())
  return len(text)
 def flush(self):sys.__stdout__.flush()

def run_logged(argv):
 # Stream a launcher run into logs/demo-switch.log and the dashboard status line.
 process=subprocess.Popen(argv,cwd=BASE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 with (BASE/'logs'/'demo-switch.log').open('a') as log:
  log.write('\n=== '+time.strftime('%Y-%m-%d %H:%M:%S')+' '+' '.join(argv)+'\n')
  for line in process.stdout:
   log.write(line);log.flush();line=line.strip()
   if line:switch_log(line)
 if process.wait():raise RuntimeError(' '.join(argv[1:3])+' failed: '+(DEMO_MODE['log'][-1] if DEMO_MODE['log'] else 'no output'))
def switch_log(line):
 with LOCK:DEMO_MODE['log']=(DEMO_MODE['log']+[line[:240]])[-10:]
 progress_line(line)
def to_cellular():
 demo.stop()  # DIFI modems; the launcher flashes the stock image and starts OAI
 ensure_stopped()
 CELL.hold=True
 try:run_logged(['python3','-u','launcher.py','--mode','oai','--detach'])
 finally:CELL.hold=False
 CELL.setup(switch_log)  # UE traffic servers, Sierra session and UE agent
def to_difi():
 try:CELL.stop()  # UE agent off, Sierra session deleted and UE in airplane mode
 except Exception as e:switch_log('UE stop: '+str(e)[-120:])
 run_logged(['python3','-u','launcher.py','--stop'])
 demo.stop(include_previous=True);ensure_stopped()
 start()  # flashes LIVESPLIT and starts the DIFI modems and traffic
DEMO_CONTAINERS=('cusat-compression','cusat-comp-loopback','cusat-oai','cusat-backhaul','cusat-leo')
def ensure_stopped(timeout=20):
 # docker stop gives up silently on a modem stuck in the GPU driver (seen on .12: an
 # unkillable modem left a zombie holding GPU memory, and the next start failed). Check
 # that no demo container or modem GPU process remains before going on.
 deadline=time.monotonic()+timeout;stuck={}
 while time.monotonic()<deadline:
  stuck={}
  for n in HOSTS:
   out=transport.ssh(n,'docker ps --format {{.Names}} | grep -x -E "'+'|'.join(DEMO_CONTAINERS)+'" || true; '
                     'nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep dvbs2 || true').split('\n')
   left=[x.strip() for x in out if x.strip()]
   if left:stuck[n]=left
  if not stuck:return
  time.sleep(2)
 raise RuntimeError('Could not stop everything: '+'; '.join('Spark .'+n+': '+', '.join(v) for n,v in stuck.items())+
                    '. A modem is probably stuck in the GPU driver: reboot that Spark, then start the demo again.')

def to_stopped():
 # Leave the radios idle: no feeder modem, gNB, DIFI modem or UE transmits.
 try:CELL.stop()  # UE agent off, PDU session deleted, Sierra in airplane mode
 except Exception as e:switch_log('UE stop: '+str(e)[-120:])
 run_logged(['python3','-u','launcher.py','--stop'])
 demo.stop()
 ensure_stopped()
 switch_log('All modems stopped; the Sierra is in airplane mode')
def difi_running():
 try:return 'true' in transport.ssh('11','docker inspect -f {{.State.Running}} '+demo.CONTAINER+' 2>/dev/null || true')
 except subprocess.CalledProcessError:return False
def cellular_running():
 # The cellular demo owns the radios while its feeder runs (OAI may be stopped for rewiring).
 try:return 'true' in transport.ssh('11','docker inspect -f {{.State.Running}} cusat-oai cusat-backhaul 2>/dev/null || true')
 except subprocess.CalledProcessError:return False

# Each demo's page lives under its own prefix, so a page's files can never come from the
# other demo mid-switch; / redirects to the demo that owns the radios (idle: the DIFI
# page, which offers both start buttons).
PAGES={'/difi/':'dashboard-live','/cellular/':'dashboard-cellular'}
class Handler(http.server.SimpleHTTPRequestHandler):
 def __init__(self,*args,**kw):
  super().__init__(*args,directory=str(BASE/('dashboard-live' if UI=='live' else 'dashboard')),**kw)
 def end_headers(self):
  if not self.path.startswith('/api/'):self.send_header('Cache-Control','no-cache')  # revalidate page files
  super().end_headers()
 def redirect(self,where):
  self.send_response(302);self.send_header('Location',where);self.send_header('Cache-Control','no-store');self.send_header('Content-Length','0');self.end_headers()
 def log_message(self,*args):pass
 def answer(self,status,value):
  payload=json.dumps(value,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
 def do_GET(self):
  if self.path in ('/api/state','/api/state?lite=1'):
   with LOCK:
    payload=json.loads(json.dumps({k:v for k,v in STATE.items() if k!='history' or self.path=='/api/state'}))
    recent=[r for r in PING['recent'] if r is not None]
    payload.update(orbit=dict(ORBIT),link=dict(LINKCTRL),split=dict(SPLIT),split_capable=demo.BINARY in (SPLIT_BINARY,LIVE_BINARY),live_rx_split=demo.BINARY==LIVE_BINARY,split_switch_ms=STATE.get('split_switch_ms'),demo=DEMO,traffic_mbps=TRAFFIC_MBPS,
     server_time=time.time(),measurement_tx_noise=MEASUREMENT_TX_NOISE,measurement_nsys=MEASUREMENT_NSYS,demo_mode=dict(DEMO_MODE),phy={'sym_rate':demo.PHY['sym_rate'],**PHY_INFO},events=[{k:e[k] for k in ('id','kind','label','status','ms','time','applied')} for e in EVENTS if time.time()-e['time']<180],ping={'rtt_ms':PING['rtt_ms'],'age_s':time.time()-PING['time'] if PING['time'] else None,'loss':(len(PING['recent'])-len(recent))/len(PING['recent']) if PING['recent'] else None},ui=UI,image=demo.IMAGE)
   # CELL.payload() takes LOCK itself, so build it after the block above.
   payload['cellular']=CELL.payload() if DEMO_MODE['active']=='cellular' or DEMO_MODE['switching']=='cellular' else None
   payload['server_time']=time.time();self.answer(200,payload)
  elif UI!='live':super().do_GET()
  else:
   path=self.path.split('?')[0]
   if path in ('/','/index.html'):return self.redirect('/cellular/' if DEMO_MODE['active']=='cellular' else '/difi/')
   for prefix,folder in PAGES.items():
    if path==prefix[:-1]:return self.redirect(prefix)
    if path.startswith(prefix):
     self.directory=str(BASE/folder);self.path=self.path[len(prefix)-1:]
     return super().do_GET()
   self.answer(404,{'error':'Not found'})
 def do_POST(self):
  if self.path!='/api/control':return self.answer(404,{'error':'Not found'})
  origin=self.headers.get('Origin')
  if origin and origin!='http://'+self.headers.get('Host',''):return self.answer(403,{'error':'Invalid origin'})
  try:
   length=int(self.headers.get('Content-Length','0'))
   if length<1 or length>1024:raise ValueError('Invalid request size')
   data=json.loads(self.rfile.read(length));reply=apply(data.get('action'),data.get('value'))
   self.answer(200,reply)
  except (ValueError,TypeError) as e:self.answer(400,{'ok':False,'error':str(e)})
  except Exception as e:self.answer(502,{'ok':False,'error':str(e)})

def adopt_running_split():
 # A dashboard restart must not assume the default route: take TX (and sync) from the
 # config the running modems were started with, and RX from their telemetry, since RX
 # may have switched in-session since.
 found={}
 for n in HOSTS:
  try:text=transport.ssh(n,'docker inspect -f {{.State.Running}} '+demo.CONTAINER+' && cat '+demo.SOURCE+'/compression.yaml')
  except subprocess.CalledProcessError:return
  if not text.startswith('true'):return
  route={k:(re.search(k+r'=([a-z]+)',text) or [None,None])[1] for k in ('tx_split','rx_split','sync')}
  stats=poll_host(n)
  rx=next((k for k,v in RX_SPLIT_CODES.items() if v==stats.get('rx_split')),route['rx_split'])
  if stats.get('profile') in PROFILES:PROFILE['value']=stats['profile']
  found[n]=(route['tx_split'] or 'indices',rx or 'filtered',route['sync'] or 'ff')
 if len(set(found.values()))==1:
  tx,rx,sync=found[HOSTS[0]]
  with LOCK:SPLIT.update(tx=tx,rx=rx,sync=sync)
  print('Running split: TX',tx,'RX',rx,flush=True)
 else:print('Modems disagree on the split:',found,flush=True)

def main():
 global UI,MEASUREMENT_TX_NOISE,MEASUREMENT_NSYS
 parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');parser.add_argument('--port',type=int,default=8765)
 parser.add_argument('--ui',choices=('live','classic'),default='live',help='live: split/orbit dashboard driving the channel; classic: compression/SNR sliders')
 parser.add_argument('--no-split',action='store_true',help='run the original modem build; split controls stay on the default route')
 parser.add_argument('--measurement-tx-noise',action='store_true',help='experiment only: noise on symbol-IQ TX before RF; restart modems in TX B before measuring')
 parser.add_argument('--measurement-nsys',action='store_true',help='experiment only: launch modems under Nsight Systems for controlled CUDA captures')
 parser.add_argument('--demo',choices=('difi','orbit'),default='difi',help='difi: functional splits on a static GEO link; orbit: satellite pass/altitude emulation');args=parser.parse_args()
 global DEMO
 UI=args.ui;DEMO=args.demo;MEASUREMENT_TX_NOISE=args.measurement_tx_noise
 MEASUREMENT_NSYS=args.measurement_nsys;demo.MEASUREMENT_NSYS=MEASUREMENT_NSYS
 if MEASUREMENT_NSYS and (args.start or UI!='live' or DEMO!='difi'):parser.error('CUDA measurements require --demo difi --ui live without --start; restart modems through the dashboard')
 if MEASUREMENT_TX_NOISE and (args.start or UI!='live' or DEMO!='difi'):parser.error('TX-noise measurements require --demo difi --ui live without --start; select TX B through the dashboard')
 if DEMO=='difi':ORBIT.update(altitude=GEO_KM,angle=0.0,animate=False)
 for n in HOSTS:transport.connect(n)
 def present(binary):return all(subprocess.run(['ssh','-S',str(transport.SOCKETS[n]),'wineslab@192.168.50.'+n,'test -x '+demo.SOURCE+binary[len('/fresh'):]]).returncode==0 for n in HOSTS)
 if UI=='live' and not args.no_split:
  demo.BINARY=LIVE_BINARY if present(LIVE_BINARY) else SPLIT_BINARY if present(SPLIT_BINARY) else demo.BINARY
 print('Modem build:',demo.BINARY,flush=True)
 install_config()
 if args.start:start()
 elif cellular_running():
  DEMO_MODE['active']='cellular';CELL.status.update(setup='running',last_restart=time.time())
  print('Cellular demo is running; DIFI loops idle',flush=True)
 elif difi_running():adopt_running_split()
 else:DEMO_MODE['active']='stopped';print('No demo running: radios idle',flush=True)
 threading.Thread(target=poll_loop,daemon=True).start()
 if UI=='live':
  for loop in (orbit_loop,acm_loop,ping_loop,watchdog_loop,delay_loop):threading.Thread(target=loop,daemon=True).start()
  CELL.start()  # idles unless the cellular demo owns the radios
 print(f'Dashboard: http://localhost:{args.port}',flush=True)
 http.server.ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
if __name__=='__main__':main()
