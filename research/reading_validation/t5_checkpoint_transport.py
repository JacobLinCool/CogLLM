"""Rotate exact optimizer snapshots in content-addressed chunks small enough for Colab transport."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event,Thread
from time import sleep
import subprocess

from research.backbone_transfer.extract import digest,read_json

CHUNK=7*1024*1024


def replace_json(path,value):
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(value)+'\n');tmp.replace(path)


def split_verified(path,manifest,storage):
    storage.mkdir(exist_ok=True,parents=True);sha=manifest['sha256']
    with TemporaryDirectory(dir=storage) as temporary:
        work=Path(temporary);parts=[];total=0;actual=sha256()
        try:f=path.open('rb')
        except FileNotFoundError:return None
        with f:
            while data:=f.read(CHUNK):
                key=sha256(data).hexdigest();p=work/(key+'.bin');p.write_bytes(data);actual.update(data);total+=len(data);parts.append({'sha256':key,'bytes':len(data)})
        if actual.hexdigest()!=sha or total!=manifest['bytes']:return None
        for p in work.iterdir():p.replace(storage/p.name)
        pointer={**manifest,'parts':parts};replace_json(storage/'latest.json',pointer);return pointer


def start_transport(run_dir,recovery):
    stop=Event();failures=[]
    def watch():
        previous=None
        try:
            while not stop.wait(2):
                f=run_dir/'resume-manifest.json'
                if not f.exists():continue
                m=read_json(f)
                if m['sha256']==previous:continue
                if split_verified(run_dir/m['name'],m,recovery):previous=m['sha256'];print('Durable T5 remote checkpoint',m['step'],flush=True)
        except Exception as error:failures.append(error)
    worker=Thread(target=watch,daemon=True);worker.start();return stop,worker,failures


def download(job,rel,path):
    remote='/content/'+job.name+'/checkpoint-transport/'+rel
    try:r=subprocess.run(['colab','--auth','adc','download','-s',job.name,remote,str(path)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=65)
    except subprocess.TimeoutExpired:return False
    return r.returncode==0


def sync(job,store):
    store.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=store) as temporary:
        tmp=Path(temporary);receipt=tmp/'pointer.json'
        if not download(job,'latest.json',receipt):return None
        r=read_json(receipt);sha=r['sha256'];step=r['step'];target=store/(f'step-{step:04d}-'+sha[:12]);old=store/'latest.json'
        if target.exists():return target
        plan=read_json(job/'plan.json')
        if not isinstance(step,int) or not 0<=step<=1840 or r['protocol_sha256']!=plan['protocol_sha256'] or r['bytes']!=sum(p['bytes'] for p in r['parts']):raise ValueError('Invalid snapshot receipt')
        for part in r['parts']:
            h=part['sha256']
            if len(h)!=64 or not set(h)<=set('abcdef0123456789') or not 0<part['bytes']<=CHUNK:raise ValueError('Invalid optimizer-state transport chunk')
        def pull(h):
            dest=tmp/(h+'.bin');return download(job,dest.name,dest) and digest(dest)==h
        with ThreadPoolExecutor(max_workers=3) as pool:
            if not all(pool.map(pull,sorted({p['sha256'] for p in r['parts']}))):return None
        full=tmp/'snapshot.pt';hashed=sha256()
        with full.open('wb') as w:
            for p in r['parts']:
                data=(tmp/(p['sha256']+'.bin')).read_bytes();hashed.update(data);w.write(data)
        if hashed.hexdigest()!=sha or full.stat().st_size!=r['bytes']:raise ValueError('Optimizer-state download mismatch')
        new=tmp/'verified';new.mkdir();full.replace(new/'snapshot.pt');replace_json(new/'manifest.json',r);new.replace(target);replace_json(old,{'directory':target.name,'checkpoint':sha,'step':step});print('Local T5 resume verified',step,sha[:12],flush=True)
        # Keep the newest two verified exact snapshots of this single training trajectory.
        for prior in sorted((p for p in store.glob('step-*') if p.is_dir()),reverse=True)[2:]:
            if set(k.name for k in prior.iterdir())!={'snapshot.pt','manifest.json'}:raise ValueError('Unrecognized files inside an explicitly managed restart directory')
            (prior/'snapshot.pt').unlink();(prior/'manifest.json').unlink();prior.rmdir()
        return target


def monitor(job):
    store=job.parent/'t5-adapter-resume';ever=False
    while True:
        m=read_json(job/'manifest.json');phase=m['phase']
        if phase in ('finished','failed','release-failed') or m.get('release_verified'):return
        if 'allocated' in m['timeline']:
            sync(job,store)
        sleep(20)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('job',type=Path);monitor(p.parse_args().job)
