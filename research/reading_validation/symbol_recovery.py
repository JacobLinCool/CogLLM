"""Transfer immutable scientific feature shards as they finish, with verified resume receipts."""
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,subprocess
from research.backbone_transfer.extract import digest,read_json
from research.reading_validation.t5_checkpoint_transport import split_verified,replace_json


def publish(path,details):
    folder=Path('symbol-checkpoint-transport');meta={**details,'sha256':digest(path),'bytes':path.stat().st_size,'name':path.name}
    pointer=split_verified(path,meta,folder)
    if not pointer:raise ArithmeticError('Completed symbol checkpoint was modified')
    replace_json(folder/path.with_suffix('.json').name,pointer)


def copy(job,store,j):
    name=f'shard-{j:03}';out=store/(name+'.npz');receipt=store/(name+'.json')
    if receipt.is_file() and out.is_file():return True
    def fetch(leaf,dest):
        try:r=subprocess.run(['colab','--auth','adc','download','-s',job.name,'/content/'+job.name+'/symbol-checkpoint-transport/'+leaf,str(dest)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)
        except subprocess.TimeoutExpired:return False
        return r.returncode==0
    with TemporaryDirectory(dir=store) as td:
        stage=Path(td);rfile=stage/'receipt.json'
        if not fetch(name+'.json',rfile):return False
        v=read_json(rfile)
        if v['name']!=out.name or not 1<=len(v['ids'])<=100 or v['stimuli']!=read_json(job/'plan.json')['payload_members_sha256']['stimuli.jsonl']:raise ValueError('Checkpoint disagrees with registered symbol packet')
        def pull(p):
            sha=p['sha256'];fp=stage/(sha+'.bin')
            if len(sha)!=64 or set(sha)-set('0123456789abcdef'):raise ValueError('Invalid shard content ID')
            return fetch(fp.name,fp) and digest(fp)==sha
        with ThreadPoolExecutor(max_workers=3) as pool:
            if not all(pool.map(pull,v['parts'])):return False
        archive=stage/'joined.npz';h=hashlib.sha256()
        with archive.open('wb') as w:
            for q in v['parts']:
                data=(stage/(q['sha256']+'.bin')).read_bytes();h.update(data);w.write(data)
        if h.hexdigest()!=v['sha256'] or archive.stat().st_size!=v['bytes']:raise ArithmeticError('Incorrect reassembled symbol shard')
        archive.replace(out);rfile.replace(receipt);print('Durable symbol features',name,len(v['ids']),flush=True);return True


def monitor(job):
    store=job.parent/'symbol-progress';store.mkdir(exist_ok=True);j=0
    while True:
        m=read_json(job/'manifest.json')
        if m.get('release_verified') or m['phase'] in ('finished','failed','release-failed'):return
        if 'allocated' in m['timeline']:
            while j<50 and copy(job,store,j):j+=1
        sleep(12)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('job',type=Path);monitor(p.parse_args().job)
