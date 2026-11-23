"""Lock individual posteriors from earlier observed answers before evaluating genuinely later choices."""
from argparse import ArgumentParser
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
from hashlib import sha256
from pathlib import Path
import csv,string
import numpy as np
from scipy.special import logsumexp,softmax
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.reader_history_materials import DIR,OUT,PLAN
from research.reading_validation.reader_history_model import response_law,scores,match,report
from research.reading_validation.cloze_population_data import ROOT as POP,normalized
from research.reading_validation.cloze_population_model import material,distribution
from research.reading_validation.population_inference import DEFAULT

FILENAMES=('anonymized_fill-in-1x35.csv','anonymized_fill-in-1x50.csv','anonymized_fill-in-61x50.csv')


def clock(text):
    parts=text.split()
    zone={'PST':-8,'PDT':-7}[parts[-2]]
    return datetime.strptime(' '.join(parts[:-2]+parts[-1:]),'%a %b %d %H:%M:%S %Y').replace(tzinfo=timezone(timedelta(hours=zone))).timestamp()


def sources():
    return {key:digest(DIR/key) for key in [*FILENAMES,'output.tsv','replacements.csv','censors.csv','excluded_workers.txt']}


def records():
    exclusions=set((DIR/'excluded_workers.txt').read_text().splitlines())
    for f in FILENAMES:
        with (DIR/f).open() as fp:
            for k,row in enumerate(csv.DictReader(fp)):
                if row['WorkerId'] in exclusions:continue
                pairs=[(a,a.replace('Input.','Answer.')) for a in row if a.startswith('Input.sentence')]
                yield f,k,row,pairs


def get_geometry():
    qmap=read_json(DIR/'verified-question-map.json')
    number={};mapping={};assignments={};uid=0
    for f,k,row,fields in records():
        participant=row['WorkerId']
        if participant not in mapping:mapping[participant]=len(mapping)
        r=mapping[participant]
        start,end=clock(row['AcceptTime']),clock(row['SubmitTime'])
        if not start<=end:raise ValueError("Chronology inversion")
        fields_with_items=[]
        for a,b in fields:
            sentence=row[a]
            if sentence not in number:number[sentence]=len(number)+1
            combined=qmap.get(sentence)
            if combined is not None:fields_with_items.append((combined,b,number[sentence]))
        assignments[(f,k)]=(r,start,end,fields_with_items)
        uid+=len(fields_with_items)
    if set(number)!=set(qmap):raise ValueError("Official public question inventory differs")
    return assignments,number


def answer_normalizer():
    corrections=defaultdict(dict)
    with (DIR/'replacements.csv').open() as stream:
        for line in stream:
            v=line.rstrip(string.whitespace).split(',')
            if len(v)<3:continue
            j=''.join(c for c in v[0] if c.isdigit())
            if j:corrections[int(j)][v[1]]=v[2]
    censors=set((DIR/'censors.csv').read_text().splitlines())
    vocab={r['word']:i for i,r in enumerate(read_json(POP/'dictionary.json'))}
    def apply(answer,question):
        v=answer.lower().rstrip(string.whitespace).rstrip(string.punctuation)
        if v in censors:v=v[:1]+'*'*(len(v)-1)
        v=corrections[question].get(v,v)
        if not v or v.casefold()=='no response':return None,None
        v=normalized(v)
        return vocab.get(v,len(vocab)),v
    return apply


def base(native):
    source=read_json(OUT/'source-lock.json')
    model=OUT/'fold-5.json'
    if source['fits'][-1]!=digest(model) or source['protocol']!=digest(PLAN):
        raise ValueError("Source-only population prior changed")
    m=read_json(model)
    data=read_json(OUT/'verified-input-seal.json')
    receipt=read_json(native/'receipt.json')
    if receipt['input']!=data['packet'] or receipt['output']!=digest(native/'logs.npz'):
        raise ValueError("Wrong native observations")
    with np.load(native/'logs.npz') as z:
        ids=z['ids'].tolist();raw=z['native']
    with np.load(DEFAULT/'weights.npz') as f:
        p=material(np.arange(len(raw)),raw,f['route'],f['share'])
        lp=distribution(p,np.empty((len(ids),0)),m['single'],f['U'],f['V']).astype(np.float32)
    return lp,dict(zip(ids,range(len(ids)),strict=True)), np.array(m['prior'])


def exposures(geom,idx):
    assigned=defaultdict(list)
    for key,(r,start,end,fields) in geom.items():assigned[r].append((start,end,key,fields))
    sets={'chronological':{},'single_assignment':{}};counts=Counter()
    for r,records_for_reader in assigned.items():
        sequence=sorted(records_for_reader,key=lambda a:(a[1],a[0],a[2]))
        start,end,bid,fields=sequence[0]
        earlier=[(bid,field,number,idx[uid]) for uid,field,number in fields]
        reserved={z[-1] for z in earlier}
        if len(reserved)!=len(earlier):raise ValueError("Repeat sentence within starting assignment")
        key=lambda t:sha256(('fixed reader history '+str(t[-1])).encode()).digest()
        past=sorted(earlier,key=key)
        if len(sequence)==1:
            sets['single_assignment'][r]=(past,[],set(reserved))
            continue
        later=[];used=set(reserved)
        for st,en,k,data in sorted(sequence[1:],key=lambda a:(a[0],a[1],a[2])):
            if st<end:counts['assignments_overlapping_earliest']+=1;continue
            for uid,field,number in data:
                w=idx[uid]
                if w in used:counts['repeat_stem_observations']+=1;continue
                used.add(w);later.append((k,field,number,w))
        if later:sets['chronological'][r]=(past,later,reserved)
    return sets,counts


def lock(native):
    lp,idx,prior=base(native);z,_=response_law(lp)
    G,_=get_geometry();studies,cov=exposures(G,idx);normalize=answer_normalizer()
    requests=defaultdict(list)
    for mode,data in studies.items():
        for r,(history,_,_) in data.items():
            # Only the candidate history page can be read at forecast-lock time.
            # Single-page readers reserve the first 20 offered items independently of their answers.
            requested=history if mode=='chronological' else history[:20]
            for key,field,no,item in requested:requests[key].append((field,no,item))
    history_values={}
    for f,k,row,_ in records():
        key=(f,k)
        for field,no,item in requests.get(key,()):
            val,_=normalize(row[field],no)
            if val is not None:history_values[key,field]=val
    buf={};counts={};matches={}
    for mode,data in studies.items():
        priors={};candidates={};requests_for_mode=[]
        for r,(hist,later,available) in data.items():
            selected=hist if mode=='chronological' else hist[:20]
            actual=[(r,item,history_values[key,field]) for key,field,no,item in selected if (key,field) in history_values][:20]
            if len(actual)!=20:continue
            future=later if mode=='chronological' else hist[20:]
            if not future:continue
            priors[r]=softmax(np.log(prior)+scores(lp,z,np.asarray(actual)).sum(0))
            candidates[r]=available
            requests_for_mode.extend((r,k[0],k[1],field,no,item) for k,field,no,item in future)
        null,overlap=match(list(priors),candidates)
        a=sorted(priors);buf[mode+'_readers']=a;buf[mode+'_post']=np.stack([priors[r] for r in a]);buf[mode+'_null']=[null[r] for r in a]
        counts[mode]={'eligible_readers':len(a),'offered_future_answers':len(requests_for_mode),'history_words_each':20,
                      'null_material_overlap_mean':float(np.mean(overlap)),'null_material_overlap_min':float(np.min(overlap))}
        matches[mode]=requests_for_mode
    path=DIR/'independent-reader-posteriors.npz';np.savez_compressed(path,**buf)
    selection=DIR/'independent-future-items.json';write_json(selection,matches)
    write_json(OUT/'external-lock.json',dict(native=digest(native/'receipt.json'),source=digest(OUT/'source-lock.json'),inputs=sources(),prior_posteriors=digest(path),
        future_item_assignments=digest(selection),conditioning=counts,chronology_audit=dict(cov),ordered_native_probability_table=sha256(lp.tobytes()).hexdigest(),
        conditioned_on='Twenty genuine earlier choices. No future response word used in determining a reader posterior.'))
    print(counts)


def score(native):
    lock=read_json(OUT/'external-lock.json');lp,idx,prior=base(native);z,other=response_law(lp)
    post=DIR/'independent-reader-posteriors.npz';future=DIR/'independent-future-items.json'
    if (digest(post)!=lock['prior_posteriors'] or digest(future)!=lock['future_item_assignments'] or sources()!=lock['inputs'] or
        sha256(lp.tobytes()).hexdigest()!=lock['ordered_native_probability_table']):
        raise ValueError("Independent prior/personal forecasts changed")
    selected=read_json(future);normalize=answer_normalizer()
    wanted=defaultdict(list)
    for mode,queries in selected.items():
        for r,f,k,field,no,item in queries:wanted[f,k].append((mode,r,field,no,item))
    cases=defaultdict(list);missing=Counter()
    for f,k,row,_ in records():
        for mode,r,field,no,item in wanted.get((f,k),()):
            val,_=normalize(row[field],no)
            if val is None:missing[mode]+=1;continue
            cases[mode].append((r,item,val))
    out={}
    with np.load(post) as P:
        for mode,records0 in cases.items():
            ev=np.array(records0);E=[];L=[];O=[]
            readers=P[mode+'_readers'];mids={r:i for i,r in enumerate(readers)};p=P[mode+'_post'];n=P[mode+'_null']
            for i,r in enumerate(readers):
                q=ev[ev[:,0]==r]
                if not len(q):continue
                ps=np.stack((p[i],prior,p[mids[n[i]]]))
                all_log=scores(lp,z,q)
                ll=logsumexp(all_log[:,:,None]+np.log(ps.T)[None,:,:],axis=1)
                L.append(-np.column_stack((ll,lp[q[:,1],q[:,2]])))
                O.append(np.column_stack((other[q[:,1]]@ps.T,np.exp(lp[q[:,1],-1]))));E.append(q)
            e=np.vstack(E);l=np.vstack(L);o=np.vstack(O)
            from research.reading_validation.reader_history_decompose import decompose
            out[mode]=report(e,l,o);out[mode]['empty_future_fields']=missing[mode]
            out[mode]['choice_content']=decompose(e,l,o)
    write_json(OUT/'external-scores.json',dict(forecast=digest(OUT/'external-lock.json'),study=out,
        interpretations={'chronological':'The earlier questionnaire had already been submitted when each later questionnaire was accepted.',
                         'single_assignment':'Disjoint fields in a single assignment; actual within-page question order is unknown.'}))
    print({k:(b['readers'],b['scored_responses'],b['paired']) for k,b in out.items()})


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('action',choices=['lock','score']);p.add_argument('native',type=Path);a=p.parse_args()
    (lock if a.action=='lock' else score)(a.native)
