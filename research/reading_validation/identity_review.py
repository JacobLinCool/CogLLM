"""Register and evaluate correction of reader-raised concerns in publicly released scientific reviews."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
import tarfile

import numpy as np
from tokenizers import Tokenizer

from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.identity_causal_score import confidence
from research.reading_validation.identity_fit import ARMS,RESULT,predict
from research.reading_validation.identity_prepare import DATA,TOKENIZER,sealed_rows,to_tokens
from research.reading_validation.identity_causal_fit import WORD,IGNORE as STOP


ROOT=DATA/'aries'; OUT=Path('research/results/reading-validation-identity-review')
PROTOCOL=Path(__file__).with_name('identity-review-protocol.md')


def key(*values):
    return sha256(json.dumps(values,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def paragraphs(papers,source,indices):
    if not indices:return ''
    if list(indices)!=list(range(indices[0],indices[-1]+1)):return None
    raw=papers[source]
    chunks=[raw[i]['text'].strip() for i in indices]
    return '\n'.join(chunks) if all(chunks) else None


def prepare():
    OUT.mkdir(exist_ok=True,parents=True)
    tokenizer=Tokenizer.from_file(str(TOKENIZER)); source_files=('LICENSE','s2orc.tar.gz','paper_edits.jsonl','review_comments.jsonl','edit_labels_test.jsonl')
    hashes={n:digest(ROOT/n) for n in source_files};labels=defaultdict(dict)
    for row in stream_rows(ROOT/'edit_labels_test.jsonl'):
        if row['split']!='test':raise ValueError('Unexpected ARIES partition')
        pos,neg=map(set,(row['positive_edits'],row['negative_edits']))
        if pos&neg or row['comment_id'] in labels[row['doc_id']]:raise ValueError('Duplicated or conflicting human decisions')
        labels[row['doc_id']][row['comment_id']]=(pos,neg)
    edits={r['doc_id']:r for r in stream_rows(ROOT/'paper_edits.jsonl') if r['doc_id'] in labels}
    comments={(r['doc_id'],r['comment_id']):r['comment'].strip() for r in stream_rows(ROOT/'review_comments.jsonl') if r['doc_id'] in labels}
    pdfs={r[k] for r in edits.values() for k in ('source_pdf_id','target_pdf_id')}
    papers={}
    with tarfile.open(ROOT/'s2orc.tar.gz',mode='r|gz') as archive:
        for member in archive:
            pdf=Path(member.name).stem
            if pdf not in pdfs or not member.isfile():continue
            source=json.load(archive.extractfile(member))
            if pdf in papers:raise ValueError('Conflicting PDF identity')
            papers[pdf]=source['pdf_parse']['body_text']+source['pdf_parse']['back_matter']
    if set(papers)!=pdfs:raise ValueError('Missing one or more original or revised PDFs')
    catalog={};closures={}; counts=Counter();qualifying=Counter(); source_ids={}
    tokens=lambda t:tokenizer.encode(t,add_special_tokens=False)
    for doc,group in sorted(labels.items()):
        meta=edits[doc];changes={e['edit_id']:e for e in meta['edits']};editions=('old','revised')
        def passage(ident,version):
            e=changes[ident]
            return paragraphs(papers,meta['source_pdf_id' if version=='old' else 'target_pdf_id'],e['source_idxs' if version=='old' else 'target_idxs'])
        def format_row(ids,revision):
            texts=[comments[(doc,c)] for c in ids]
            if any(not t or len(tokens(t).ids)>200 for t in texts):return None
            if not revision or len(tokens(revision).ids)>600:return None
            full='\n'.join([*texts,revision]);enc=tokens(full)
            if len(enc.ids)>1024:return None
            a=0;b=len(texts[0])+1;c=b+len(texts[1])+1
            bounds=[(a,b-1),(b,c-1)];anchors={str(i):to_tokens(enc.offsets,span) for i,span in enumerate(bounds)}
            p=to_tokens(enc.offsets,(c,len(full)))
            if any(v[1]>p[0] for v in anchors.values()):raise ValueError('Unexpected annotation/passage token overlap')
            ident=key(full,*ids)
            return {'id':ident,'text':full,'anchor_tokens':anchors}, {'id':ident,'window':p}
        # Choose one acceptable edit per comment and compare using the same paper's explicit cross-negatives.
        candidates=[]
        for c1,c2 in combinations(sorted(group),2):
            p1,n1=group[c1];p2,n2=group[c2];sets=[sorted(p1&n2),sorted(p2&n1)]
            if not all(sets):continue
            kval=key(doc,c1,c2);ids=[c1,c2] if int(kval[:2],16)%2 else [c2,c1]
            for e1,e2 in sorted(product(*sets),key=lambda ee:key(doc,c1,c2,*ee)):
                instances={};rendered=[];scoreit=True
                for c,e in zip((c1,c2),(e1,e2),strict=True):
                    versions={}
                    for version in editions:
                        revised=passage(e,version);result=format_row(ids,revised)
                        if result:
                            r,g=result;rendered.append(r);versions[version]=g
                        elif version=='revised':
                            scoreit=False;break
                    if not scoreit: break
                    instances[str(ids.index(c))]=versions
                if scoreit:
                    candidates.append((kval,{'group':doc,'candidate_pair':ids,'targets':instances,'source_edits':[e1,e2]},rendered));break
        qualifying[doc]=len(candidates)
        for kval,index,rows in sorted(candidates,key=lambda x:x[0])[:8]:
            closures[kval]=index
            for r in rows:catalog[r['id']]=r
    sealed_rows(ROOT/'stimuli.jsonl',[r for _,r in sorted(catalog.items())]);sealed_rows(ROOT/'labels.jsonl',[{'id':k,**v} for k,v in sorted(closures.items())])
    marker={'source_files':hashes,'protocol':digest(PROTOCOL),'stimuli':digest(ROOT/'stimuli.jsonl'),'outcomes':digest(ROOT/'labels.jsonl'),
            'official_test_documents':len(labels),'official_test_comments':sum(map(len,labels.values())),
            'all_eligible_comment_pairs':sum(qualifying.values()),'articles_with_pairs':sum(v>0 for v in qualifying.values()),'contrast_pairs':len(closures),
            'stimuli_count':len(catalog),'input_token_limit':1024,'unique_edits':len({(v['group'],idx) for v in closures.values() for idx in v['source_edits']}),
            'both_originals_available':sum(all('old' in fields for fields in v['targets'].values()) for v in closures.values()),
            'dataset_license':'ODC-BY 1.0','model_weights':digest(RESULT/'readout.json')}
    write_json(OUT/'input-seal.json',marker);print({k:v for k,v in marker.items() if type(v)==int},flush=True)


def forecast(path):
    marker=read_json(OUT/'input-seal.json');w=read_json(RESULT/'readout.json')
    if digest(RESULT/'readout.json')!=marker['model_weights'] or read_json(path/'geometry.json')['input_sha256']!=marker['stimuli']:
        raise ValueError('Model weights or input packet differs from the registration')
    arrays={}
    for r in stream_rows(ROOT/'stimuli.jsonl'):
        with np.load(path/(r['id']+'.npz'),allow_pickle=False) as a:
            arrays[r['id']]=predict(w['models']['official'],a['pair'])
    np.savez_compressed(ROOT/'forecasts.npz',**arrays)
    write_json(OUT/'forecast-seal.json',{'forecasts_sha256':digest(ROOT/'forecasts.npz'),'weights_sha256':marker['model_weights'],
                                          'test_labels_loaded_during_forecasting':False})


def score():
    src=read_json(OUT/'input-seal.json');lock=read_json(OUT/'forecast-seal.json')
    if digest(ROOT/'forecasts.npz')!=lock['forecasts_sha256'] or digest(ROOT/'labels.jsonl')!=src['outcomes']:raise ValueError('Source or forecasts changed')
    items={r['id']:r for r in stream_rows(ROOT/'stimuli.jsonl')};labels=list(stream_rows(ROOT/'labels.jsonl'))
    with np.load(ROOT/'forecasts.npz',allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
    by_doc=defaultdict(list)
    tokenize=lambda t:[w for w in WORD.findall(t.casefold()) if w not in STOP]
    def lexical_score(texts,passage):
        docs=list(map(tokenize,texts));Q=tokenize(passage);D=len(docs);avg=np.mean([len(x) for x in docs]);values=[]
        for d in docs:
            bag=Counter(d); total=0
            for w in set(Q):
                f=bag[w]; df=sum(w in rival for rival in docs)
                total+=np.log1p((D-df+.5)/(df+.5))*f*2.2/(f+1.2*(.25+.75*len(d)/max(1.,avg)))
            values.append(total)
        return np.array(values)
    arms=(*ARMS,'bm25'); tok=Tokenizer.from_file(str(TOKENIZER)); detail=[]
    for r in labels:
        cell={}
        for t,ed in r['targets'].items():
            target=int(t);grad={}
            for version,g in ed.items():
                k=g['id'];lo,hi=g['window'];row=items[k];enc=tok.encode(row['text'],add_special_tokens=False);off=enc.offsets
                def part(a,b):return row['text'][off[a][0]:off[b-1][1]].strip()
                earlier=[part(*bounds) for _,bounds in sorted(row['anchor_tokens'].items())];passage=part(lo,hi)
                risk=-np.log1p(-np.clip(arrays[k][:,:,lo:hi],0,1-1e-12)).mean(-1)
                bm25=lexical_score(earlier,passage)/(hi-lo)
                diff=[*list(risk[:,target]-risk[:,1-target]),bm25[target]-bm25[1-target]]
                grad[version]=dict(zip(arms,map(float,diff),strict=True))
            cell[target]=grad
        def win(v):return .5 if abs(v)<1e-10 else float(v>0)
        scores={m:{'both':win(cell[0]['revised'][m])*win(cell[1]['revised'][m]),'single':np.mean([win(cell[t]['revised'][m]) for t in (0,1)]),
                   'old_both':win(cell[0]['old'][m])*win(cell[1]['old'][m]) if all('old' in cell[t] for t in (0,1)) else None,
                   'paired_revision_gain':float(np.mean([cell[t]['revised'][m]-cell[t]['old'][m] for t in (0,1)])) if all('old' in cell[t] for t in (0,1)) else None,
                   'addition_margin':float(np.mean([cell[t]['revised'][m] for t in (0,1) if 'old' not in cell[t]])) if any('old' not in cell[t] for t in (0,1)) else None}
                for m in arms}
        by_doc[r['group']].append(scores);detail.append({'id':r['id'],'paper':r['group'],'scores':scores})
    means={}
    for doc,group in by_doc.items():
        means[doc]={m:{s:float(np.mean(v)) if (v:=[d[m][s] for d in group if d[m][s] is not None]) else None for s in group[0][m]} for m in arms}
    summary={m:{s:confidence([v[m][s] for v in means.values() if v[m][s] is not None]) for s in next(iter(means.values()))[m]} for m in arms}
    deltas={m:confidence([v['new_readout']['both']-v[m]['both'] for v in means.values()]) for m in arms if m!='new_readout'}
    out={'articles':len(means),'pairs':len(labels),'arms':summary,'paired_primary':deltas,'gate':all(deltas[k]['paired_item_bootstrap_95'][0]>0 for k in ('joint','lexical')),
         'papers_with_both_originals':sum(v['new_readout']['old_both'] is not None for v in means.values()),
         'papers_with_addition':sum(v['new_readout']['addition_margin'] is not None for v in means.values())}
    write_json(OUT/'scores.json',out);write_json(OUT/'pair-scores.json',detail);print(out['articles'],out['pairs'],out['arms']['new_readout']['both'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','forecast','score'));p.add_argument('--extraction',type=Path)
    args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='forecast':forecast(args.extraction)
    else:score()
