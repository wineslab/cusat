#!/usr/bin/env python3
import re, sys
for line in sys.stdin:
 if 'Lock=' in line:
  def value(pattern,default='?'):
   m=re.search(pattern,line);return m.group(1) if m else default
  print('t='+value(r'(\d+)s Lock')+' L='+value(r'Lock=(\w+)')+' SNR='+value(r'pilotSNR=([\d.]+)')+' MC='+value(r'TxMC=.*?\((\d+)\)')+' UF='+value(r'UF=(\d+)')+' OF='+value(r'UHD_OF=(\d+)'),flush=True)
 elif any(s in line for s in ('ERROR','UNDERFLOW','OVERFLOW','CRASH','Radio stopped','RuntimeError')):
  print(line.rstrip()[:150],flush=True)
