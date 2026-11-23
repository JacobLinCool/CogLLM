"""Reduce public time-locked Spanish EEG on the host to seven-electrode word-level observations."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request,urlopen
import re,time
import numpy as np
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.spanish_clue_data import ROOT,OUT,build,dictionary

SITE=('Pz','P3','P4','CP1','CP2','C3','C4');CON=('High','Low','Real');PROTOCOL=Path(__file__).with_name('spanish-eeg-protocol.md')


def header(p):
    s=p.read_bytes().decode('utf-8-sig',errors='replace');f=dict(re.findall(r'(?m)^(\w+)=([^\r\n]+)',s));
    if f.get('DataFormat')!='BINARY' or f['DataOrientation']!='MULTIPLEXED' or f['BinaryFormat']!='INT_16' or float(f['SamplingInterval'])!=4000 or int(f['NumberOfChannels'])!=32:raise ValueError('Unexpected original EEG header')
    chans=[];scale=[]
    for i in range(1,33):
        ch=f['Ch'+str(i)].split(',');chans.append(ch[0]);scale.append(float(ch[2]));
        if 'V' not in ch[3] or not 0<float(ch[2])<1:raise ValueError('Unverified voltage unit or gain')
    if len(set(chans))!=32 or not set(SITE)|{'LM','RM','veye'}<=set(chans):raise ValueError('Missing original electrodes')
    return chans,np.array(scale,dtype=np.float32)


def markers(p):
    s=p.read_bytes().decode('utf-8-sig',errors='replace');stims=[(int(k),int(t)-1) for k,t in re.findall(r'(?m)^Mk\d+=Stimulus,S\s*(\d+),(\d+),',s)];codes={5,7,9,11,12,13,21,22,23,31,32,33};events=[(k,t) for k,t in stims if k in codes];out=[];last=-1
    for j,(k,t) in enumerate(events):
        if k not in (11,12,13):continue
        seq=events[j:j+21];want=[k]+[5]*5+[7,k+10]+[5]*6+[k+20]+[5]*2+[9]+[5]*3
        if [s[0] for s in seq]!=want or seq[0][1]<=last:raise ValueError('A 21-word EEG marker sequence is missing a word or condition')
        ts=[s[1] for s in seq]
        if min(np.diff(ts))<100 or max(np.diff(ts))>1400:raise ValueError('Unusual timing inside a complete recorded trial')
        last=ts[-1];out.append((CON[k-11],ts))
    if len(out)!=144:raise ValueError('A recording lacks a complete set of primary exposures')
    return out


def geometry():
    tech=ROOT/'technical';lex=dictionary();rows=[];errors=[];available=read_json(ROOT/'eeg-directory.json');
    for fname in read_json(ROOT/'eligible-logs.json'):
        person,design,_,_=build(ROOT/'logs'/fname,lex);stem=f'sub-{person:03d}_ses-001_task-wseek_eeg';af=stem+'.eeg';vp=tech/(stem+'.vmrk');hp=tech/(stem+'.vhdr')
        try:
            if af not in available or not hp.exists() or not vp.exists():raise ValueError('No corresponding raw recording, header and markers')
            ch,sc=header(hp);events=markers(vp)
            if [r['condition'] for r in design]!=[c for c,t in events]:raise ValueError('Original trial-level stimulus order and voltage triggers disagree')
            ids=[r['id'] for r in design];ts=[t for c,t in events];lookup=sha256(('CogLLM Spanish subjects '+str(person)).encode()).hexdigest()[:20]
            rows.append(dict(id=lookup,filename=af,ids=ids,timestamps=ts,channels=ch,scale=sc.tolist(),source_bytes=available[af]['size'],url=available[af]['download'],header=digest(hp),markers=digest(vp)))
        except ValueError as error:errors.append({'reason':str(error),'source_session':sha256(fname.encode()).hexdigest()})
    folder=ROOT/'wave';folder.mkdir(exist_ok=True);from research.reading_validation.identity_prepare import sealed_rows
    sealed_rows(folder/'geometry.jsonl',rows);write_json(OUT/'eeg-input-seal.json',{'plan':digest(PROTOCOL),'geometry':digest(folder/'geometry.jsonl'),'original_trials':144,'words_per_trial':21,'eeg_people':len(rows),'unmatched':errors,'source_bytes':sum(x['source_bytes'] for x in rows),'technical_only_before_registration':True});print('EEG people',len(rows),'exclusions',len(errors),{e['reason'] for e in errors},flush=True)


def fetch(row,dst):
    err=None
    for attempt in range(3):
        try:
            h=sha256();n=0
            with urlopen(Request(row['url'],headers={'User-Agent':'CogLLM public EEG verification'}),timeout=90) as src, dst.open('wb') as out:
                while data:=src.read(2**20):out.write(data);h.update(data);n+=len(data)
            if n!=row['source_bytes']:raise ValueError('Partial original EEG recording')
            return h.hexdigest()
        except (OSError,ValueError) as e:err=e;time.sleep(2+attempt)
    raise RuntimeError('Public EEG recording could not be verified after retry') from err


def reduce(row,folder):
    dst=folder/(row['id']+'.npz');receipt=folder/(row['id']+'.json')
    if receipt.exists():
        rec=read_json(receipt)
        if digest(dst)!=rec['output'] or rec['protocol']!=digest(PROTOCOL):raise ValueError('Cached electrode data changed')
        return rec
    with TemporaryDirectory(dir=folder) as tmp:
        p=Path(tmp)/'source.eeg';sha=fetch(row,p);ch=row['channels'];scale=np.asarray(row['scale'],np.float32);d=np.memmap(p,mode='r',dtype='<i2').reshape(-1,len(ch));epoch=np.asarray(row['timestamps'],np.int64)[:,:,None]+np.arange(-25,125)[None,None,:];
        if epoch.min()<0 or epoch.max()>=len(d):raise ValueError('Marker outside actual original recording')
        idx=[ch.index(k) for k in SITE];mastoids=[ch.index(k) for k in ('LM','RM')];eye=ch.index('veye');scalp=[i for i,c in enumerate(ch) if c not in ('LM','RM','veye')];values=[];ok=[];strict=[]
        for ts in epoch:
            z=np.asarray(d[ts,:],np.float32)*scale[None,None,:];z-=z[:,:,mastoids].mean(2)[:,:,None];z-=z[:,:25].mean(1)[:,None,:];c=z[:,:,idx].mean(2);m=-c[:,100:150].mean(1);early=-c[:,50:75].mean(1);sc=np.abs(z[:,:,scalp]).max((1,2));oc=np.abs(z[:,:,eye]).max(1);flat=(np.ptp(c,axis=1)<.25);g=(sc<=200)&(oc<=100)&~flat;values.append(np.c_[m,early]);ok.append(g);strict.append(g&(oc<=75))
        del d
    np.savez_compressed(dst,values=np.asarray(values,dtype='float32'),valid=np.asarray(ok),strict=np.asarray(strict),ids=np.asarray(row['ids']));rec={'protocol':digest(PROTOCOL),'source_sha256':sha,'original_bytes':row['source_bytes'],'output':digest(dst),'valid':int(np.sum(ok)),'strict':int(np.sum(strict)),'epoch_words':144*21,'header':row['header'],'markers':row['markers']};write_json(receipt,rec);print('Reduced matched recording',row['id'],rec['valid'],'valid words',flush=True);return rec


def extract():
    from research.backbone_transfer.extract import stream_rows
    folder=ROOT/'wave';p=folder/'geometry.jsonl';lock=read_json(OUT/'eeg-input-seal.json');
    if digest(PROTOCOL)!=lock['plan'] or digest(p)!=lock['geometry']:raise ValueError('Original EEG study registration changed')
    rows=list(stream_rows(p));
    with ThreadPoolExecutor(max_workers=3) as workers:summary=list(workers.map(lambda r:reduce(r,folder),rows))
    write_json(OUT/'eeg-extraction.json',{'geometry':digest(p),'protocol':digest(PROTOCOL),'program':digest(Path(__file__)),'sources':summary,'people':len(rows),'word_epochs':sum(x['epoch_words'] for x in summary),'usable':sum(x['valid'] for x in summary),'strict_usable':sum(x['strict'] for x in summary)})

if __name__=='__main__':
    p=ArgumentParser();p.add_argument('action',choices=('geometry','extract'));getattr(__import__(__name__,fromlist=['']),p.parse_args().action)()
