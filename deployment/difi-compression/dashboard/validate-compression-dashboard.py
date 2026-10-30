#!/usr/bin/env python3
"""Exercise live compression, manual CCM MODCOD, noise, and baseline recovery."""
import argparse,json,pathlib,time,urllib.request

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--url',default='http://localhost:8765')
 p.add_argument('--settle',type=float,default=8)
 p.add_argument('--output',default='logs/dashboard-validation.jsonl')
 a=p.parse_args()
 if a.settle<5:p.error('settle must be at least 5 seconds')
 path=pathlib.Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
 def state():return json.load(urllib.request.urlopen(a.url+'/api/state',timeout=10))['hosts']
 def control(action,value):
  body=json.dumps({'action':action,'value':value}).encode()
  return json.load(urllib.request.urlopen(urllib.request.Request(a.url+'/api/control',body,{'Content-Type':'application/json'}),timeout=20))
 control('snr',None);control('compression','plain');control('modcod',4)
 time.sleep(a.settle);previous=state()
 actions=[('compression',['trim','dpcm2','dpcm4','dpcm6','dpcm8','plain']),
          ('modcod',[7,12,13,18,19,24,28,4]),('snr',[20,12,6,2,None])]
 try:
  for action,values in actions:
   for value in values:
    reply=control(action,value);time.sleep(a.settle);current=state()
    assert all(v.get('connected') for v in current.values()),'Telemetry lost'
    assert all(v['uptime']>=previous[n]['uptime'] for n,v in current.items()),'Modem restarted'
    if action=='compression':assert all(v['profile']==value for v in current.values()),'Wrong codec'
    if action=='modcod':assert all(v['tx_modcod']==value and v['rx_modcod']==value and v['locked'] for v in current.values()),'MODCOD did not arrive at RX'
    if action=='snr' and value==2:assert all(v['fer'] is not None and v['fer']>.8 and v['goodput_mbps']<1 for v in current.values()),'Low-SNR FEC degradation missing'
    if action=='snr' and value is None:assert all(v['snr_db']>20 and v['goodput_mbps']>9 for v in current.values()),'Baseline did not recover'
    record={'action':action,'value':value,'reply':reply,'hosts':current,'time':time.time()}
    with path.open('a') as f:f.write(json.dumps(record)+'\n')
    print('PASS',action,value,{n:{k:v.get(k) for k in ('uptime','locked','profile','tx_modcod','rx_modcod','snr_db','fer','rx_mbps','goodput_mbps')} for n,v in current.items()},flush=True)
    previous=current
 finally:
  # A failed test should also try to restore the usable presentation baseline.
  for action,value in [('snr',None),('modcod',4),('compression','plain')]:
   try:control(action,value)
   except Exception as e:print('Baseline restore failed:',e,flush=True)
 print('Live hardware validation complete:',path)
if __name__=='__main__':main()
