"""Regression checks for interruption/recovery and independently timed UE samples."""
import importlib.util,sys,types,time,threading,subprocess,json
from pathlib import Path
from unittest.mock import Mock,patch
BASE=Path(__file__).resolve().parents[1]
sys.modules['serial']=types.SimpleNamespace()
s=importlib.util.spec_from_file_location('agent',BASE/'cellular/ue-agent.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
p=Mock();p.poll.return_value=None;p.wait.return_value=0;a.PROCESSES['dl']=p
rates={'dl':'27M','ul':'2.5M'}
a.traffic_control(True,'pass1',rates);revision=a.TRAFFIC['revision'];a.traffic_control(True,'pass1',rates)
assert a.TRAFFIC['revision']==revision
p.reset_mock();a.traffic_control(False,'pass1',rates,'Out of view');assert p.terminate.called
assert a.STATE['iperf']['dl']['mbps']==0 and a.STATE['iperf']['dl']['paused']
a.traffic_control(True,'pass2',rates);assert a.TRAFFIC['generation']=='pass2'
assert a.RATES==rates
sys.path.insert(0,str(BASE/'cellular'));import cellular as c
cell=c.Cellular(None,lambda:True,threading.Lock());cell.status['setup']='running';cell.orbit['phase']='gap'
calls=[]
cell.ssh=lambda n,cmd,**kw:(calls.append((n,cmd)) or subprocess.CompletedProcess([],0,'',''))
class Done(Exception):pass
count=[0]
def tick(t):
 count[0]+=1
 if count[0]==2:
  cell.orbit.update(phase='pass',pass_id=1)
  for f in cell.feeder.values():f.update(locked=True,time=time.time())
 if count[0]==4:cell.orbit['phase']='gap'
 if count[0]==5:raise Done()
with patch.object(c.time,'sleep',tick),patch.object(c.socket,'socket') as sock:
 try:cell.traffic_loop()
 except Done:pass
 packets=[json.loads(call.args[0]) for call in sock.return_value.sendto.call_args_list]
 assert [x['enabled'] for x in packets]==[False,False,True,False],packets
 assert sum('tc qdisc del' in cmd for _,cmd in calls)==4
 assert sum('for p in 5202 5203' in cmd for _,cmd in calls)==1
 assert all(x['rates']==rates for x in packets)
# The agent's 2-second byte interval must not become a 72-Mbps spike in a
# 1-second dashboard poll; duplicate agent snapshots must not become zeroes.
cell=c.Cellular(None,lambda:True,threading.Lock());step=[-1];observed=[]
samples=[(100,0),(102,9000000),(102,9000000),(103,13500000)]
def clock_tick(t):
 if step[0]>=0:observed.append(dict(cell.rates))
 step[0]+=1
 if step[0]>=len(samples):raise Done()
def ssh(n,cmd,*args):
 ts,rx=samples[step[0]]
 data=json.dumps({'wwan':{'time':ts,'rx_bytes':rx,'tx_bytes':rx}})+'\n100\n100\n' if n=='11' else '100 100 100'
 return subprocess.CompletedProcess([],0,data,'')
cell.ssh=ssh
with patch.object(c.time,'sleep',clock_tick),patch.object(c.time,'time',lambda:1000+step[0]):
 try:cell.counter_loop()
 except Done:pass
assert all(abs(r['wwan_rx']-36)<.001 for r in observed[1:]),observed
print('PASS: gap cancels senders; repeated heartbeat keeps sessions; new pass resets generation; core servers stop and queues flush; flows resume after lock; UL stays 2.5M; delayed/duplicate UE snapshots produce 36 Mbps consistently.')
