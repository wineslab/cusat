#!/usr/bin/env python3
"""Start the two direct-radio endpoints and their laptop/remote tmux consoles."""
import argparse, concurrent.futures, os, pathlib, shlex, socket, subprocess, sys, threading, time
BASE = pathlib.Path(__file__).resolve().parent
LOG = BASE / 'logs'
RELEASES = {
    'old': '/home/wineslab/cusat-release-v1',
    'new': os.environ.get('CUSAT_CELLULAR_SOURCE', '/home/wineslab/cusat-radio'),
}
DEFAULT_RELEASE = os.environ.get('CUSAT_CELLULAR_RELEASE', 'new')
SOURCE = RELEASES[DEFAULT_RELEASE]
HOSTS = ('11', '12')
SOCKETS = {}
MODE = "leo"
CONTAINER = "cusat-leo"
q = shlex.quote

def run(argv, **kw):
    return subprocess.run(argv, check=True, text=True, **kw)

def ssh(n, command, capture=True):
    return run(['ssh', '-S', str(SOCKETS[n]), '-o', 'ServerAliveInterval=10',
                'wineslab@192.168.50.'+n, command], capture_output=capture).stdout

def say(s):
    print(s, flush=True)
    with (LOG/'setup.log').open('a') as f: f.write(s+'\n')

def connect(n):
    candidates = [BASE/('.ssh'+n+'.sock'), BASE.parent/'cusat-release'/('.spark'+n+'.sock')]
    for path in candidates:
        if path.exists() and subprocess.run(['ssh','-S',str(path),'-O','check',
              'wineslab@192.168.50.'+n],capture_output=True).returncode == 0:
            SOCKETS[n] = path; return
    path = candidates[0]
    run(['ssh','-M','-N','-f','-S',str(path),'-o','ControlPersist=8h',
         '-o','ServerAliveInterval=10','wineslab@192.168.50.'+n])
    SOCKETS[n] = path

def window(n, name, command):
    ssh(n, ': > '+SOURCE+'/console-'+name+'.log')
    ssh(n, 'tmux new-window -d -t cusat -n '+q(name)+' '+q(command))
    ssh(n, 'tmux pipe-pane -o -t '+q('cusat:'+name)+' '+q('cat >> '+SOURCE+'/console-'+name+'.log'))

def clocks():
    # Short RTT midpoint exchange avoids dependency on an unavailable LAN NTP server.
    listener=socket.socket(); listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    listener.bind(('192.168.50.10',15557)); listener.listen(); listener.settimeout(20)
    def serve():
        try:
            for _ in range(12):
                c,_=listener.accept()
                with c: c.recv(1); c.sendall(repr(time.time()).encode())
        finally: listener.close()
    thread=threading.Thread(target=serve,daemon=True); thread.start()
    client='''import socket,time
samples=[]
for _ in range(6):
 s=socket.create_connection(("192.168.50.10",15557)); t0=time.monotonic(); s.sendall(b"t"); ts=float(s.recv(100)); t1=time.monotonic(); s.close(); samples.append((t1-t0,ts+(t1-t0)/2,t1))
rtt,ts,m=min(samples)
time.clock_settime(time.CLOCK_REALTIME,ts+time.monotonic()-m)
print("Clock synchronized; RTT %.3f ms"%(rtt*1000))
'''
    for n in HOSTS: say(n+': '+ssh(n,'docker exec '+CONTAINER+' python3 -c '+q(client)).strip())
    thread.join(1)

def endpoint(n):
    ssh(n, 'test -x '+SOURCE+'/build/release/apps/dvbs2-uhd-trx; command -v tmux; docker image inspect cusat:local >/dev/null')
    config=(BASE/'configs'/(MODE+'-spark'+n+'.yaml')).read_text()
    ssh(n,'cat > '+SOURCE+'/node-live.yaml <<\'CUSAT_CONFIG\'\n'+config+'\nCUSAT_CONFIG')
    ssh(n,'tmux kill-session -t cusat 2>/dev/null || true; for c in cusat-oai cusat-backhaul cusat-leo; do docker stop -t 20 "$c" >/dev/null 2>&1 || true; docker rm "$c" >/dev/null 2>&1 || true; done; sleep 3')
    nic='enp1s0f0np0' if n=='11' else 'aerial00'
    ssh(n,'docker run --rm --pid host --network host --privileged --entrypoint nsenter cusat:local -t 1 -m -u -i -n -p bash '+SOURCE+'/deployment/cusat/host/spark/tune.sh '+nic+' 0-3')
    gpu='--runtime=nvidia' if n=='11' else '--gpus all'
    command=f'docker run -d --name {CONTAINER} --cpuset-cpus '+('4-19' if MODE=='leo' else '4-11')+f' {gpu} --network host --ipc host --privileged -v /dev:/dev -v {SOURCE}:/fresh -e LD_LIBRARY_PATH=/fresh/build/release/lib cusat:local /fresh/build/release/apps/dvbs2-uhd-trx --config /fresh/node-live.yaml'
    ssh(n,command)
    ssh(n,'tmux new-session -d -s cusat -n modem '+q('docker logs --since 2m -f '+CONTAINER))
    ssh(n,'tmux set-option -t cusat prefix C-a; tmux bind-key -T prefix C-a send-prefix; tmux set-option -t cusat remain-on-exit on')
    for _ in range(90):
        if ssh(n,'docker exec '+CONTAINER+' test -e /sys/class/net/'+('dvbs2b0' if (n=='11')==(MODE=='leo') else 'dvbs2a0')+' && echo ready || true').strip()=='ready':break
        state=ssh(n,'docker inspect --format {{.State.Status}} '+CONTAINER).strip()
        if state=='exited':raise RuntimeError('Modem exited on Spark '+n+': '+ssh(n,'docker logs --tail 15 '+CONTAINER))
        time.sleep(1)
    else: raise RuntimeError('TAP did not appear on Spark '+n)
    if MODE=='leo':
        ssh(n,'docker exec '+CONTAINER+' /fresh/scripts/setup-network.sh '+('node1' if n=='11' else 'node0'))
    else:
        tap='dvbs2a0' if n=='11' else 'dvbs2b0'
        addr='10.200.0.1' if n=='11' else '10.200.0.2'
        peer='10.200.0.2' if n=='11' else '10.200.0.1'
        mac='02:00:00:00:00:02' if n=='11' else '02:00:00:00:00:01'
        ssh(n,'docker exec '+CONTAINER+' sh -c '+q('ip addr replace '+addr+'/30 dev '+tap+'; ip link set '+tap+' up; ip neigh replace '+peer+' lladdr '+mac+' nud permanent dev '+tap))
    window(n,'iperf-server','docker exec -it '+CONTAINER+' iperf3 -s --forceflush')
    say('Spark '+n+': modem and iperf server started')

def laptop():
    subprocess.run(['tmux','kill-session','-t','cusat-demo'],capture_output=True)
    for i,n in enumerate(HOSTS):
        command=shlex.join(['ssh','-tt','-S',str(SOCKETS[n]),'wineslab@192.168.50.'+n,'docker logs --tail 20 -f '+CONTAINER+' 2>&1 | python3 -u '+SOURCE+'/telemetry.py'])
        if i==0:
            run(['tmux','new-session','-d','-s','cusat-demo','-n','radios',command])
        else: run(['tmux','split-window','-h','-t','cusat-demo:radios',command])
        run(['tmux','select-pane','-t','cusat-demo:radios.'+str(i),'-T','Spark '+n+' / USRP .'+('18' if n=='11' else '20')])
    run(['tmux','set-option','-t','cusat-demo','pane-border-status','top'])
    run(['tmux','new-window','-d','-t','cusat-demo','-n','validation','tail -F '+q(str(LOG/'validation.txt'))])
    for i,n in enumerate(HOSTS):
        cmd=shlex.join(['ssh','-tt','-S',str(SOCKETS[n]),'wineslab@192.168.50.'+n,'tail -n 20 -F '+SOURCE+'/console-iperf-server.log'])
        if i==0:run(['tmux','new-window','-d','-t','cusat-demo','-n','traffic',cmd])
        else:run(['tmux','split-window','-v','-t','cusat-demo:traffic',cmd])
    if MODE=='oai':
        for name,n,cmd in [('oai','11','docker logs --tail 50 -f cusat-oai'),('core','12','tail -n 20 -F /var/log/open5gs/amf.log /var/log/open5gs/upf.log')]:
            run(['tmux','new-window','-d','-t','cusat-demo','-n',name,shlex.join(['ssh','-tt','-S',str(SOCKETS[n]),'wineslab@192.168.50.'+n,cmd])])
    run(['tmux','select-window','-t','cusat-demo:radios'])

def start_oai():
    config=(BASE/'configs'/'oai-gnb.conf').read_text()
    ssh('11','cat > '+SOURCE+'/oai-gnb.conf <<\'OAI_CONFIG\'\n'+config+'\nOAI_CONFIG')
    # The Docker account already has root privileges; enter the host namespace
    # to restart native core services after their DVB address exists.
    services='open5gs-nrfd open5gs-scpd open5gs-amfd open5gs-smfd open5gs-upfd open5gs-ausfd open5gs-udmd open5gs-udrd open5gs-pcfd open5gs-nssfd open5gs-bsfd'
    ssh('12','docker run --rm --pid host --network host --privileged --entrypoint nsenter cusat:local -t 1 -m -u -i -n -p systemctl restart '+services)
    binary='/opt/openairinterface5g/cmake_targets/ran_build/build/nr-softmodem'
    ssh('11','docker run -d --name cusat-oai --cpuset-cpus 12-19 --network host --ipc host --privileged -v /dev:/dev -v '+SOURCE+':/fresh -w /opt/openairinterface5g/cmake_targets/ran_build/build cusat-mobile-backhaul-oai:develop '+binary+' -O /fresh/oai-gnb.conf')
    window('11','oai','docker logs --tail 100 -f cusat-oai')
    window('12','core','tail -n 20 -F /var/log/open5gs/amf.log /var/log/open5gs/smf.log /var/log/open5gs/upf.log')
    say('OAI started on .18 chain1; Open5GS core on .20; PLMN001/01, DNN oai')

def validate_oai():
    for _ in range(90):
        output=ssh('11','docker logs cusat-oai 2>&1')
        if 'Received NGSetupResponse' in output and 'RU 0 RF started' in output:break
        state=ssh('11','docker inspect --format {{.State.Status}} cusat-oai').strip()
        if state=='exited':raise RuntimeError('OAI exited: '+output[-4000:])
        time.sleep(1)
    else:raise RuntimeError('OAI did not register with AMF')
    if 'MCC=001 MNC=01' not in output:raise RuntimeError('OAI PLMN is not 001/01')
    if 'RX stalled' in output or 'PRACH job is in the past' in output:
        raise RuntimeError('OAI radio timing errors: '+output[-3000:])
    services=ssh('12','systemctl is-active mongod open5gs-amfd open5gs-smfd open5gs-upfd open5gs-ausfd open5gs-udmd open5gs-udrd open5gs-nrfd open5gs-scpd open5gs-pcfd open5gs-nssfd open5gs-bsfd cusat-dns cusat-dn')
    route=ssh('11','ip route get 10.200.0.2')
    if 'dev dvbs2a0' not in route:raise RuntimeError('Core path bypasses DVB')
    page=ssh('11','curl --fail --silent --max-time 15 --interface dvbs2a0 http://10.200.0.2:8080/')
    if 'cuSAT Open5GS data network' not in page:raise RuntimeError('Data-network HTTP test failed')
    echo_code='import socket,struct,time;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.bind(("10.200.0.1",0));s.settimeout(5);p=struct.pack("!BBHIHBB",0x32,1,4,0,4242,0,0);t=time.monotonic();s.sendto(p,("10.200.0.2",2152));r,a=s.recvfrom(1024);assert a[0]=="10.200.0.2" and r[1]==2;print("N3 GTP-U echo passed, RTT_ms=",round((time.monotonic()-t)*1000,1))'
    echo=ssh('11','python3 -c '+q(echo_code))
    return 'PASS: OAI RF started on chain0; NGSetupResponse received; native core services active; HTTP over DVB passed.\n'+echo+route+'\n'+services


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['leo','oai'],default='leo')
    parser.add_argument('--release',choices=RELEASES,default=DEFAULT_RELEASE,
                        help='cellular/LEO modem deployment; DIFI has its own validated build')
    parser.add_argument('--detach',action='store_true',help='start and verify without attaching the laptop console')
    parser.add_argument('--stop',action='store_true',help='stop cuSAT/OAI radio containers and their tmux sessions')
    args=parser.parse_args()
    global MODE,CONTAINER,SOURCE
    MODE=args.mode;CONTAINER='cusat-leo' if MODE=='leo' else 'cusat-backhaul'
    SOURCE=RELEASES[args.release]
    LOG.mkdir(exist_ok=True)
    for n in HOSTS:
        connect(n)
    if args.stop:
        for n in HOSTS:ssh(n,'tmux kill-session -t cusat 2>/dev/null || true; for c in cusat-oai cusat-backhaul cusat-leo; do docker stop -t 20 "$c" >/dev/null 2>&1 || true; done')
        subprocess.run(['tmux','kill-session','-t','cusat-demo'],capture_output=True);return
    (LOG/'setup.log').write_text('Starting '+MODE+' demo\n')
    (LOG/'validation.txt').write_text('Starting validation...\n')
    say('Selected '+args.release+' modem deployment: '+SOURCE)
    for n in HOSTS:
        ssh(n,'test -d '+q(SOURCE))
        ssh(n,'cat > '+SOURCE+'/telemetry.py <<\'CUSAT_PY\'\n'+(BASE/'telemetry.py').read_text()+'\nCUSAT_PY')
    if MODE=='oai':
        ssh('11','docker image inspect cusat-mobile-backhaul-oai:develop >/dev/null; test -f '+SOURCE+'/build/oai-plugin/libusrpproxy.so')
        ssh('12','command -v open5gs-amfd; command -v mongod')
    from fpga_switch import restore_for_uhd
    restore_for_uhd()
    say('Restarting '+MODE+' demo on both Sparks')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(endpoint,HOSTS))
    if MODE=='leo':clocks()
    else:start_oai()
    for n in HOSTS:
        peer=('10.30.0.1' if n=='11' else '10.30.0.2') if MODE=='leo' else ('10.200.0.2' if n=='11' else '10.200.0.1')
        for _ in range(90):
            try:output=ssh(n,'docker exec '+CONTAINER+' ping -c 3 -W 2 '+peer);break
            except subprocess.CalledProcessError:time.sleep(1)
        else:raise RuntimeError('Radio ping failed on Spark '+n)
        with (LOG/'validation.txt').open('a') as f:f.write('Spark '+n+' radio ping:\n'+output+'\n')
        say('Spark '+n+': radio ping passed')
        window(n,'iperf-client','docker exec -it '+CONTAINER+' iperf3 -c '+peer+' -u -b '+('5M' if n=='11' else '1M')+' -t 0 -i 1 --forceflush')
    time.sleep(5)
    for n in HOSTS:
        for _ in range(60):
            output=ssh(n,'tmux capture-pane -p -S -30 -t cusat:iperf-server')
            if 'Mbits/sec' in output or 'Kbits/sec' in output:break
            time.sleep(1)
        else:raise RuntimeError('No received iperf traffic on Spark '+n+'\n'+output)
        with (LOG/'validation.txt').open('a') as f:f.write('Spark '+n+' received traffic:\n'+output+'\n')
    if MODE=='leo':
        window('11','leo','docker exec -it '+CONTAINER+' bash /fresh/scripts/leo-acm-demo.sh --cycle --start-in 10 192.168.50.11:5556 192.168.50.12:5556')
        for _ in range(20):
            output=ssh('11','tmux capture-pane -p -S -20 -t cusat:leo')
            if 'uploaded cycle=0' in output:break
            time.sleep(1)
        else:raise RuntimeError('LEO schedule upload failed: '+output)
    else:
        output=validate_oai()
    laptop()
    with (LOG/'validation.txt').open('a') as f:f.write('PASS: both radio pings, both iperf receivers, '+MODE+'\n'+output+'\n')
    say('PASS. Laptop tmux: cusat-demo; remote sessions: cusat (prefix Ctrl-a).')
    say('Ctrl-b d detaches the laptop. ./run-demo.sh --stop stops the demo.')
    if not args.detach:os.execvp('tmux',['tmux','attach-session','-t','cusat-demo'])

if __name__=='__main__':
    try: main()
    except (subprocess.CalledProcessError,RuntimeError,OSError) as e:
        say('FAILED: '+str(e))
        if isinstance(e,subprocess.CalledProcessError):say(e.stderr or '')
        sys.exit(1)
