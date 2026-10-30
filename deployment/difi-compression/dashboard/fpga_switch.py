#!/usr/bin/env python3
"""Switch only the FPGA component, using verified local image pairs and backups."""
import hashlib,json,pathlib,shlex,subprocess,time
import launcher as transport
BASE=pathlib.Path(__file__).resolve().parent
SOURCE='/home/wineslab/cusat-compression-release'
DEPS='/home/wineslab/cusat-compression-deps/uhd-4.8'
q=shlex.quote
# Timing-verified custom images: (bitstream, device tree, post-route timing report).
IMAGES={'flex1':('flex1/x4xx.bit','flex1/x4xx.dts','flex1/post_route_timing_summary.rpt'),
        'livesplit':('livesplit/usrp_x410_fpga_UC_200_LIVESPLIT.bit','livesplit/usrp_x410_fpga_UC_200_LIVESPLIT.dts','livesplit/post_route_timing_summary.rpt')}

def device(n):
    code=(BASE/'mpm-info.py').read_text()
    return json.loads(transport.ssh(n,'python3 -c '+q(code)+' 192.168.40.'+('18' if n=='11' else '20')))

def verify(image,n):
    root=BASE/'fpga'; manifest=json.loads((root/'manifest.json').read_text())
    paths=list(IMAGES[image]) if image in IMAGES else ['stock-backup/usrp'+('18' if n=='11' else '20')+'/x410.bin','stock-backup/usrp'+('18' if n=='11' else '20')+'/x410.dts']
    for relative in paths:
        path=root/relative
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256'][relative]:raise RuntimeError('FPGA checksum mismatch: '+str(path))
    if image in IMAGES and manifest['timing'] not in (root/paths[2]).read_text():raise RuntimeError(image+' timing report failed')
    return root/paths[0],root/paths[1]

def switch(image):
    for n in ('11','12'):
        bit,dts=verify(image,n)
        current=device(n)
        state_path=SOURCE+'/fpga-state.json'
        state=json.loads(transport.ssh(n,'cat '+state_path+' 2>/dev/null || echo "{}"'))
        if image=='stock' and current['fpga_version_hash']=='c37b318.clean':continue
        if image in IMAGES and state.get('image')==image and state.get('hash')==current['fpga_version_hash']:continue
        directory=SOURCE+'/fpga-'+image
        transport.ssh(n,'mkdir -p '+directory)
        for file in (bit,dts):
            subprocess.run(['scp','-o','ControlPath='+str(transport.SOCKETS[n]),str(file),'wineslab@192.168.50.'+n+':'+directory+'/'+file.name],check=True)
        addr='192.168.40.'+('18' if n=='11' else '20')
        print('Loading verified '+image+' FPGA on '+addr,flush=True)
        output=transport.ssh(n,'LD_LIBRARY_PATH='+DEPS+'/lib '+DEPS+'/bin/uhd_image_loader --args '+q('addr='+addr+',type=x4xx')+' --no-fw --fpga-path '+directory+'/'+bit.name)
        print(output,flush=True)
        for _ in range(60):
            try:
                current=device(n)
                if current['fpga_version']!='8.3':raise RuntimeError('Unexpected FPGA compatibility '+str(current))
                break
            except (subprocess.CalledProcessError,OSError,json.JSONDecodeError):time.sleep(1)
        else:raise RuntimeError('Radio did not return after FPGA reload: '+addr)
        record=json.dumps({'image':image,'hash':current['fpga_version_hash']})
        transport.ssh(n,'printf %s '+q(record)+' > '+state_path)
        print('Radio '+addr+' returned; '+current['fpga_version_hash'],flush=True)

def restore_for_uhd():
    # A stock UHD/tenant demo cannot use the single-chain FLEX1/LIVESPLIT switchboard graphs.
    for n in ('11','12'):transport.connect(n)
    if all(device(n)['fpga_version_hash']=='c37b318.clean' for n in ('11','12')):return
    for n in ('11','12'):
        transport.ssh(n,'for c in cusat-compression cusat-comp-loopback cusat-oai cusat-backhaul cusat-leo; do docker stop -t 20 "$c" >/dev/null 2>&1 || true; docker rm "$c" >/dev/null 2>&1 || true; done')
    time.sleep(3)
    switch('stock')
