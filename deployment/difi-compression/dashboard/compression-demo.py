#!/usr/bin/env python3
"""Compare DIFI RX compression modes on the two directly wired X410s."""
import argparse,concurrent.futures,json,os,pathlib,re,shlex,subprocess,time
import launcher as transport
import fpga_switch
BASE=pathlib.Path(__file__).resolve().parent
SOURCE='/home/wineslab/cusat-compression-release'
DEPS='/home/wineslab/cusat-compression-deps/uhd-4.8'
CONTAINER='cusat-compression'
SESSION='cusat-compression'
HOSTS=('11','12')
PHY={'sym_rate':15.36e6,'modcod':'qpsk1/2','bitrate':'10M','shift':6,'protocol':'udp','warmup':10}
# FPGA image loaded for live runs (see fpga_switch.IMAGES). LIVESPLIT drives
# daughterboard A only; DVB uses ch0 (TX/RX0 out, RX1 in), ch1 is left for OAI.
IMAGE='livesplit'
CHANNELS='tx_radio=0,tx_chan=0,rx_radio=0,rx_chan=0'
BINARY='/fresh/build/apps/dvbs2-uhd-trx'  # inside the container; the dashboard may select apps-split/
MEASUREMENT_NSYS=False  # opt-in CUDA profiling; ordinary launches are unchanged
PROFILES={'plain':('plain',0,32),'trim':('trim',0,16),
          'dpcm2':('dpcm',2,4),'dpcm4':('dpcm',4,8),
          'dpcm6':('dpcm',6,12),'dpcm8':('dpcm',8,16)}
q=shlex.quote
ssh=transport.ssh

def write_remote(n,path,text):
    ssh(n,'cat > '+q(path)+" <<'CUSAT_FILE'\n"+text+'\nCUSAT_FILE')

def preflight():
    errors=[]
    for n in HOSTS:
        transport.connect(n)
        ssh(n,'set -e; test -x '+SOURCE+'/build/apps/dvbs2-uhd-trx; test -f '+SOURCE+'/build/apps/libradio_compfh.so; command -v tmux; docker image inspect cusat:local >/dev/null')
        code=(BASE/'mpm-info.py').read_text()
        address='192.168.40.'+('18' if n=='11' else '20')
        device=json.loads(ssh(n,'python3 -c '+q(code)+' '+address))
        print('Spark '+n+': '+json.dumps(device),flush=True)
        fpga_switch.verify(IMAGE,n)
        fpga_switch.verify('stock',n)
        if device['fpga_version_hash']=='c37b318.clean':
            print('  Stock image active; verified '+IMAGE+' image will be loaded for a live run.',flush=True)
        output=ssh(n,'LD_LIBRARY_PATH='+DEPS+'/lib '+DEPS+'/bin/uhd_config_info --version 2>/dev/null || true').strip()
        if not re.search(r'\b4\.8\.',output):errors.append('Spark '+n+' needs the isolated UHD 4.8 install at '+DEPS)
    return errors

def stop(include_previous=False):
    names=[CONTAINER,'cusat-comp-loopback']+(['cusat-oai','cusat-backhaul','cusat-leo'] if include_previous else [])
    for n in HOSTS:
        ssh(n,'tmux kill-session -t '+SESSION+' 2>/dev/null || true; '+
            '; '.join('docker stop -t 20 '+c+' >/dev/null 2>&1 || true; docker rm '+c+' >/dev/null 2>&1 || true' for c in names))
    subprocess.run(['tmux','kill-session','-t',SESSION],capture_output=True)
    time.sleep(3)

def config(n,profile):
    mode,bits,_=PROFILES[profile]
    radio='18' if n=='11' else '20'; dest='111' if n=='11' else '112'
    return f'''radio:
  driver: compfh
  usrp: "compfh:addr=192.168.40.{radio},dest_addr=192.168.40.{dest},dest_port=7777,adapter=sfp1,tx_ant=TX/RX0,rx_ant=RX1,{CHANNELS},rx={mode},shift={PHY["shift"]},dbits={bits or 6},rolloff=0.2,p=1,q=1,issue=streamer"
  tx_modcomp: true
  rx_prefiltered: true
  tx_freq: {2.05e9 if n=='11' else 1.95e9}
  rx_freq: {1.95e9 if n=='11' else 2.05e9}
  tx_gain: 0
  rx_gain: 20
  rate: {4*PHY["sym_rate"]}
  sym_rate: {PHY["sym_rate"]}
  send_frame_size: 1472
  recv_frame_size: 1472
modulation:
  modcod: {PHY["modcod"]}
  rolloff: 0.2
acm:
  enabled: false
gpu:
  pin_threads: true
  realtime_threads: false
  cuda_sync: block
io:
  tap: {'dvbs2b0' if n=='11' else 'dvbs2a0'}
  log: true
'''

def endpoint(n,profile):
    write_remote(n,SOURCE+'/compression.yaml',config(n,profile))
    gpu='--runtime=nvidia' if n=='11' else '--gpus all'
    profiler_mount=' -v /opt/nvidia/nsight-systems/2025.3.2:/opt/cusat-nsys:ro' if MEASUREMENT_NSYS else ''
    command=BINARY+' --config /fresh/compression.yaml'
    if MEASUREMENT_NSYS:
        # Keep the container alive after the modem exits, so the independent
        # sampler can export CUDA buffers after graceful application cleanup.
        command='/bin/sh -c '+q('/opt/cusat-nsys/bin/nsys launch --trace=cuda --cuda-flush-interval=1000 --session-new=cusat-compute --wait=all '+command+'; exec sleep infinity')
    ssh(n,f'docker run -d --name {CONTAINER} --cpuset-cpus 4-19 {gpu} --network host --ipc host --privileged -v /dev:/dev -v {SOURCE}:/fresh -v {DEPS}:/deps:ro{profiler_mount} -e LD_LIBRARY_PATH=/deps/lib:/fresh/build/lib cusat:local {command}')
    ssh(n,'tmux new-session -d -s '+SESSION+' -n modem '+q('docker logs -f '+CONTAINER))
    ssh(n,'tmux set-option -t '+SESSION+' prefix C-a; tmux set-option -t '+SESSION+' remain-on-exit on')
    tap='dvbs2b0' if n=='11' else 'dvbs2a0'
    for _ in range(60):
        if ssh(n,'test -e /sys/class/net/'+tap+' && echo ready || true').strip()=='ready':break
        if ssh(n,'docker inspect --format {{.State.Status}} '+CONTAINER).strip()=='exited':raise RuntimeError('Spark '+n+' modem exited: '+ssh(n,'docker logs --tail 30 '+CONTAINER+' 2>&1'))
        time.sleep(1)
    else:raise RuntimeError('No TAP on Spark '+n)
    ssh(n,'docker exec '+CONTAINER+' ip link set '+tap+' address '+('02:00:00:00:00:02' if n=='11' else '02:00:00:00:00:01'))
    ssh(n,'docker exec '+CONTAINER+' /fresh/scripts/setup-network.sh '+('node1' if n=='11' else 'node0'))
    ssh(n,'tmux new-window -d -t '+SESSION+' -n iperf-server '+q('docker exec -it '+CONTAINER+' iperf3 -s --forceflush'))

def laptop(log):
    subprocess.run(['tmux','kill-session','-t',SESSION],capture_output=True)
    for i,n in enumerate(HOSTS):
        command=shlex.join(['ssh','-tt','-S',str(transport.SOCKETS[n]),'wineslab@192.168.50.'+n,'tmux attach-session -t '+SESSION])
        if not i:subprocess.run(['tmux','new-session','-d','-s',SESSION,'-n','radios',command],check=True)
        else:subprocess.run(['tmux','split-window','-h','-t',SESSION+':radios',command],check=True)
    subprocess.run(['tmux','new-window','-d','-t',SESSION,'-n','results','tail -F '+q(str(log))],check=True)

def measure(profile,seconds,log):
    stop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda n:endpoint(n,profile),HOSTS))
    result={'profile':profile,'status':'passed','rx_payload_bits_per_complex_sample':PROFILES[profile][2],
            'tx':'modulation-compressed DIFI (constant across RX comparisons)','rx_msample_s':4*PHY['sym_rate']/1e6,'sym_rate':PHY['sym_rate'],'modcod':PHY['modcod'],'offered_bitrate':PHY['bitrate'],'protocol':PHY['protocol'],'shift':PHY['shift'],'traffic':{}}
    for n in HOSTS:
        peer='10.30.0.1' if n=='11' else '10.30.0.2'
        for _ in range(15):
            try:ssh(n,'docker exec '+CONTAINER+' ping -c 3 -W 1 '+peer);break
            except subprocess.CalledProcessError:time.sleep(1)
        else:raise RuntimeError('Radio ping failed: Spark '+n+' / '+profile)
    time.sleep(PHY['warmup'])
    for n in HOSTS:
        peer='10.30.0.1' if n=='11' else '10.30.0.2'
        path=SOURCE+'/iperf-'+profile+'.json'
        cmd='docker exec '+CONTAINER+' iperf3 -c '+peer+(' -u' if PHY['protocol']=='udp' else '')+' -b '+q(PHY['bitrate'])+' -t '+str(seconds)+' -J > '+path
        ssh(n,'tmux new-window -d -t '+SESSION+' -n iperf-client '+q(cmd))
    laptop(log)
    time.sleep(seconds+3)
    for n in HOSTS:
        deadline=time.monotonic()+60
        while True:
            try:
                stats=json.loads(ssh(n,'cat '+SOURCE+'/iperf-'+profile+'.json'))
                break
            except json.JSONDecodeError:
                if time.monotonic()>=deadline:raise RuntimeError('iperf did not finish on Spark '+n)
                time.sleep(1)
        if 'error' in stats:raise RuntimeError(stats['error'])
        received=stats['end'].get('sum_received') or stats['end'].get('sum')
        if not received or received.get('bytes',0)==0:raise RuntimeError('No received iperf traffic on '+n)
        result['traffic'][n]=received
        result.setdefault('quality',{})[n]={'loss_percent':received.get('lost_percent'),'jitter_ms':received.get('jitter_ms'),'bits_per_second':received.get('bits_per_second')}
        modem=ssh(n,'docker logs --tail 80 '+CONTAINER+' 2>&1')
        (log.parent/(profile+'-spark'+n+'.log')).write_text(modem)
        status=[line for line in modem.splitlines() if 'Lock=' in line]
        if not status or 'Lock=True' not in status[-1]:raise RuntimeError('RX did not retain lock on '+n)
        result.setdefault('modem',{})[n]=status[-1]
    with log.open('a') as f:f.write(json.dumps(result)+'\n')
    summary='; '.join(n+': '+format(result['quality'][n]['bits_per_second']/1e6,'.2f')+' Mbit/s, loss '+str(result['quality'][n]['loss_percent'])+'%' for n in HOSTS)
    print('PASS flow '+profile+'; '+summary+'; results: '+str(log),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=list(PROFILES)+['all'],default='all')
    parser.add_argument('--seconds',type=int,default=20)
    parser.add_argument('--sym-rate',type=float,default=PHY['sym_rate'],help='symbols/s; RX sample rate is four times this')
    parser.add_argument('--modcod',default=PHY['modcod'])
    parser.add_argument('--bitrate',default=PHY['bitrate'],help='offered UDP rate per direction; 0 means uncapped')
    parser.add_argument('--shift',type=int,default=PHY['shift'])
    parser.add_argument('--protocol',choices=['udp','tcp'],default='udp')
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--check',action='store_true',help='read prerequisites; leaves running demos intact')
    parser.add_argument('--self-test',action='store_true',help='run compression codec QA without claiming radios')
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--stop',action='store_true')
    args=parser.parse_args()
    if not 0<args.sym_rate<=61.44e6:parser.error('--sym-rate must be in (0, 61.44e6]')
    if not 0<=args.shift<=8:parser.error('--shift must be 0..8')
    if not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?[KMG]?',args.bitrate):parser.error('invalid --bitrate')
    PHY.update(sym_rate=args.sym_rate,modcod=args.modcod,bitrate=args.bitrate,shift=args.shift,protocol=args.protocol,warmup=max(0,args.warmup))
    if args.seconds<5:parser.error('--seconds must be at least 5')
    transport.LOG.mkdir(exist_ok=True)
    if args.stop:
        for n in HOSTS:transport.connect(n)
        stop();return
    if args.self_test:
        for n in HOSTS:
            transport.connect(n)
            print(ssh(n,SOURCE+'/build/apps/qa_difi_compression --log_level=test_suite'),flush=True)
        return
    errors=preflight()
    if errors:raise RuntimeError('\n'.join(errors))
    if args.check:
        print('Software prerequisites passed; live start also verifies RFNoC block availability.');return
    stamp=time.strftime('%Y%m%d-%H%M%S');directory=BASE/'logs'/('compression-'+stamp);directory.mkdir()
    log=directory/'results.jsonl';log.touch()
    # All read-only prerequisite checks precede the switch from the current demo.
    stop(include_previous=True)
    fpga_switch.switch(IMAGE)
    failures=[]
    for profile in (list(PROFILES) if args.profile=='all' else [args.profile]):
        try:measure(profile,args.seconds,log)
        except (RuntimeError,OSError,subprocess.CalledProcessError,json.JSONDecodeError) as error:
            failures.append(profile)
            record={'profile':profile,'status':'failed','error':str(error),'phy':dict(PHY)}
            for n in HOSTS:
                try:
                    modem=ssh(n,'docker logs --tail 80 '+CONTAINER+' 2>&1')
                    (directory/(profile+'-spark'+n+'.log')).write_text(modem)
                    status=[line for line in modem.splitlines() if 'Lock=' in line]
                    if status:record.setdefault('modem',{})[n]=status[-1]
                except (OSError,subprocess.CalledProcessError):pass
            with log.open('a') as f:f.write(json.dumps(record)+'\n')
            print('FAIL '+profile+': '+str(error),flush=True)
            if args.profile!='all':raise
    if failures:raise RuntimeError('Comparison finished; failed profiles: '+', '.join(failures)+'; results: '+str(log))
    print('Complete. Last profile remains running; laptop tmux '+SESSION)
    if not args.detach:os.execvp('tmux',['tmux','attach-session','-t',SESSION])

if __name__=='__main__':
    try:main()
    except (RuntimeError,OSError,subprocess.CalledProcessError,json.JSONDecodeError) as e:
        print('FAILED: '+str(e),flush=True)
        if isinstance(e,subprocess.CalledProcessError):print(e.stderr or '',flush=True)
        raise SystemExit(1)
