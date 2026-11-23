"""Read completed frozen phases and atomic LoRA checkpoints while one owned L4 continues."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from time import monotonic,sleep

PHASES=('narrative-serving','writer','recognition','ats-comprehension')


def collect(job):
    target=job/'intermediate';target.mkdir(exist_ok=True);base='/content/'+job.name
    until=monotonic()+10800
    def cli(*args,timeout=90):
        with (job/'progress-transfer.log').open('ab') as log:
            try:return subprocess.run(['colab','--auth','adc',*map(str,args)],stdout=log,stderr=subprocess.STDOUT,timeout=timeout,check=False).returncode==0
            except subprocess.TimeoutExpired:return False
    def pull(remote,local):
        partial=local.with_suffix(local.suffix+'.partial')
        if not cli('download','-s',job.name,base+'/'+remote,partial):return False
        partial.replace(local);return True
    done=set();epoch=0
    while monotonic()<until:
        plan=json.loads((job/'manifest.json').read_text());phase=plan.get('phase')
        if phase in {'finished','incomplete'}:break
        if phase!='remote-extraction':sleep(20);continue
        log=target/'worker.log'
        if not pull('worker.log',log):sleep(25);continue
        published={}
        for line in log.read_text(errors='replace').splitlines():
            if not line.startswith('{'):continue
            try:record=json.loads(line)
            except json.JSONDecodeError:continue
            for key,v in record.items():
                if key in PHASES:published[key]=v
        for name,receipt in published.items():
            if name in done:continue
            if not receipt['completed']:
                print(name,'failed',receipt.get('exception'),flush=True);done.add(name);continue
            packet=name+'-forecast';folder=target/packet
            if folder.is_dir():done.add(name);continue
            with TemporaryDirectory(dir=target) as tmp:
                tmp=Path(tmp);archive=packet+'.tar.gz';manifest=packet+'.sha256'
                code=tmp/'pack.py';code.write_text(f'''from pathlib import Path
from hashlib import sha256
import tarfile
base=Path({base!r})
source=base/'extraction'/{packet!r}
archive=base/{archive!r}
with tarfile.open(archive,'w:gz',compresslevel=1) as package:
    package.add(source,arcname={packet!r})
(base/{manifest!r}).write_text(sha256(archive.read_bytes()).hexdigest())
''')
                if not cli('exec','-s',job.name,'-f',code,'--timeout','120',timeout=140):continue
                if not pull(manifest,tmp/manifest) or not pull(archive,tmp/archive):continue
                sha=(tmp/manifest).read_text();data=(tmp/archive).read_bytes()
                if sha256(data).hexdigest()!=sha:raise ValueError('Intermediate archive checksum mismatch')
                with tarfile.open(tmp/archive) as z:
                    if not all(Path(m.name).parts[0]==packet and not any(x=='..' for x in Path(m.name).parts) and (m.isfile() or m.isdir()) for m in z):raise ValueError('Unexpected intermediate member')
                    z.extractall(tmp,filter='data')
                (tmp/packet).rename(folder);(target/(packet+'-seal.json')).write_text(json.dumps({**receipt,'archive_sha256':sha},indent=2)+'\n')
                done.add(name);print('collected',name,sha,flush=True)
        if all(name in done for name in PHASES):
            h=target/'history.json'
            if pull('extraction/memory-forecast/history.json',h):
                histories=json.loads(h.read_text());last=histories[-1]['epoch'] if histories else 0
                if last>epoch and pull('extraction/memory-forecast/training-state.pt',target/'training-state.pt'):
                    epoch=last;print('saved remote LoRA checkpoint',epoch,flush=True)
        sleep(35)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('job',type=Path);collect(p.parse_args().job)
