"""Render public human causal judgments as plain text without answer choices or labels in the encoder packet."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import tarfile
import xml.etree.ElementTree as ET
import zipfile

from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.identity_prepare import DATA,PREVIOUS,order,sealed_rows


OUT=Path('research/results/reading-validation-causal'); PROTOCOL=Path(__file__).with_name('identity-causal-protocol.md')
FILEHASH={'e-care.zip':'7be3c0e51235aa3721afaca82577be1af46fbc2fd6032f08552a4cc84bf65dda',
          'copa.tgz':'5145348834d2081ad90da0397d1db3d70fa044e506bd8ce224194d24b04cdbbe'}
SPLIT=re.compile(r'(?<=[.!?])\s+(?=\S)|\s*\n+\s*(?=\S)')


def sentences(text):
    start=0; result=[]
    for separator in SPLIT.finditer(text):
        if start < separator.start():
            result.append((start,separator.start()))
        start=separator.end()
    if start < len(text):
        result.append((start,len(text)))
    return result


def normalized(text):
    return ' '.join(text.casefold().split())


def raw():
    for name,check in FILEHASH.items():
        if digest(DATA/name)!=check:
            raise ValueError('Source version mismatch')
    with zipfile.ZipFile(DATA/'e-care.zip') as archive:
        for split in ('train','dev'):
            for line in archive.open(f'dataset/{split}_full.jsonl'):
                item=json.loads(line)
                yield ('ecare-'+split,item['index'],item['premise'],[item['hypothesis1'],item['hypothesis2']],int(item['label']),item['ask-for'])
    with tarfile.open(DATA/'copa.tgz') as archive:
        for split in ('dev','test'):
            doc=ET.parse(archive.extractfile(f'COPA-resources/datasets/copa-{split}.xml'))
            gold={}
            for word in archive.extractfile(f'COPA-resources/results/gold.{split}'):
                key,a,b=map(int,word.split()); gold[str(key)]=0 if a==1 and b==0 else 1 if a==0 and b==1 else -1
            for node in doc.findall('./item'):
                key=node.attrib['id']; label=int(node.attrib['most-plausible-alternative'])-1
                if gold[key]!=label:
                    raise ValueError('COPA gold XML and original independent answer key differ')
                yield ('copa-'+split,'copa-'+key,node.findtext('p'),[node.findtext('a1'),node.findtext('a2')],label,node.attrib['asks-for'])


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    pool={};keys=set(); split_keys={};counts=Counter();excluded=Counter()
    # Reserve independent COPA items before any training record with the same displayed text.
    records=sorted(raw(),key=lambda x:({'copa-test':0,'copa-dev':1,'ecare-dev':2,'ecare-train':3}[x[0]],order(x[1])))
    inputs={}; truth={}
    for source,i,premise,choices,correct,role in records:
        parts=[p.strip() for p in [premise,*choices]]
        if any(not p or len(sentences(p))!=1 for p in parts):
            # Preserve the two semantic units even for internal punctuation with a source-specific span probe;
            # this is deliberately excluded from the primary automatic boundary criterion.
            excluded[(source,'has_multiple_sentences_inside_a_source_statement')]+=1;continue
        premise,*choices=parts
        triple=(normalized(premise),*sorted(normalized(x) for x in choices))
        if triple in keys or choices[0].casefold()==choices[1].casefold():
            excluded[(source,'duplicate_triple')]+=1;continue
        if source=='ecare-train' and counts[source]>=10000:continue
        keys.add(triple);counts[source]+=1
        u=order(i); calibration=source=='ecare-train' and u[1]<25
        split='calibration' if calibration else source
        split_keys[i]=split
        def view(kind,lines,labels):
            text='\n'.join(lines);key=sha256(text.encode()).hexdigest()
            if key not in inputs:
                inputs[key]={'id':key,'text':text}
            assert len(sentences(text))==len(lines)
            return {'id':key,'kind':kind,'targets':labels}
        views=[]
        orders=[u[0]%2] if source=='ecare-train' and not calibration else [0,1]
        for first in orders:
            indices=[first,1-first];views.append(view('choice-'+str(first),[choices[j] for j in indices]+[premise],[int(j==correct) for j in indices]))
        if source=='ecare-train':
            alt=u[2]%2;views.append(view('single',[premise,choices[alt]],[int(alt==correct)]))
            if u[3]<52 or calibration:views.append(view('repeat',[premise,premise],[0]))
        truth[i]={'id':i,'source':split,'original_source':source,'ask_for':role,'views':views,'correct':correct}
    original={r['id']:r['text'] for r in stream_rows(PREVIOUS/'stimuli.jsonl')}
    geometry=read_json(PREVIOUS/'qwen.json')['rows']
    for key,r in geometry.items():
        if r['kind'] in ('ab','controls'):
            inputs[key]={'id':key,'text':original[key]}
    # Forecast-only document and evidence metadata are separate even on local disk.
    rows=sorted(inputs.values(),key=lambda x:x['id']);sealed_rows(DATA/'causal-stimuli.jsonl',rows)
    paths={}
    for split in ('ecare-train','calibration','ecare-dev','copa-dev','copa-test'):
        file=DATA/f'causal-{split}.jsonl';sealed_rows(file,[v for _,v in sorted(truth.items()) if v['source']==split]);paths[split]=digest(file)
    historical=sorted(k for k,v in geometry.items() if v['kind'] in ('ab','controls'))
    calibration_ids={kind:sorted({v['id'] for r in truth.values() if r['source']=='calibration'
                                for v in r['views'] if v['kind']==kind})[:320] for kind in ('repeat','single')}
    baseline=sorted({v['id'] for r in truth.values() if r['source'].startswith('copa-') for v in r['views']})
    subsets={'historical':historical,'dense':sorted(set(historical+calibration_ids['repeat']+calibration_ids['single'])),'baseline':baseline}
    subset_path=DATA/'causal-subsets.json'
    if subset_path.exists():
        if read_json(subset_path)!=subsets:raise ValueError('Changed subset selection')
    else:write_json(subset_path,subsets)
    marker={'protocol_sha256':digest(PROTOCOL),'archives':FILEHASH,'subsets_sha256':digest(DATA/'causal-subsets.json'),'total_inputs':len(rows),
            'source_eligible':dict(counts),'source_exclusions':{'/'.join(k):v for k,v in sorted(excluded.items())},
            'source_grades_sha256':paths,'graded_items':dict(Counter(r['source'] for r in truth.values())),
            'input_sha256':digest(DATA/'causal-stimuli.jsonl'),'grades_in_encoder_inputs':False}
    previous=OUT/'input-seal.json'
    current=OUT/'input-seal-v2.json'
    if previous.is_file() and {k:v for k,v in marker.items() if k!='subsets_sha256'}!=read_json(previous):
        raise ValueError('Changed the locked source or protocol')
    if current.exists():
        if read_json(current)!=marker:raise ValueError('Changed feature subset registration')
    else:write_json(current,marker)
    print(marker['graded_items'],len(rows),marker['source_exclusions'])


if __name__=='__main__':prepare()
