#!/usr/bin/env python3
"""Read MPM device information without claiming or initializing a USRP."""
import json,socket,struct,sys
class Incomplete(Exception): pass

def unpack(data,offset=0):
    def take(n):
        nonlocal offset
        if offset+n>len(data):raise Incomplete()
        out=data[offset:offset+n];offset+=n;return out
    def number(n):return int.from_bytes(take(n),'big')
    def value():
        b=number(1)
        if b<0x80:return b
        if 0xA0<=b<=0xBF:return take(b&31).decode()
        if 0x90<=b<=0x9F:return [value() for _ in range(b&15)]
        if 0x80<=b<=0x8F:return {value():value() for _ in range(b&15)}
        if b==0xC0:return None
        if b in (0xC2,0xC3):return b==0xC3
        if b in (0xCC,0xCD,0xCE,0xCF):return number(1<<(b-0xCC))
        if b in (0xD9,0xDA,0xDB):return take(number(1<<(b-0xD9))).decode()
        if b in (0xC4,0xC5,0xC6):return take(number(1<<(b-0xC4))).decode()
        if b in (0xDC,0xDD):return [value() for _ in range(number(2 if b==0xDC else 4))]
        if b in (0xDE,0xDF):return {value():value() for _ in range(number(2 if b==0xDE else 4))}
        raise ValueError('Unsupported MPM MessagePack tag '+hex(b))
    return value()

def info(addr):
    # MPM occasionally returns an unparseable reply (seen while a modem holds the
    # radio); a fresh connection answers normally.
    for attempt in range(3):
        try:return query(addr)
        except (RuntimeError,ValueError,OSError):
            if attempt==2:raise
            import time;time.sleep(1)

def query(addr):
    with socket.create_connection((addr,49601),timeout=5) as sock:
        sock.sendall(bytes([0x94,0,1,0xAF])+b'get_device_info'+bytes([0x90]))
        data=b''
        while len(data)<65536:
            chunk=sock.recv(8192)
            if not chunk:raise RuntimeError('MPM closed connection')
            data+=chunk
            try:response=unpack(data)
            except Incomplete:continue
            if response[2]:raise RuntimeError(str(response[2]))
            return {key:response[3].get(key) for key in ('serial','mpm_sw_version','fpga','fpga_version','fpga_version_hash','claimed')}
        raise RuntimeError('MPM response too large')
if __name__=='__main__':print(json.dumps(info(sys.argv[1])))
