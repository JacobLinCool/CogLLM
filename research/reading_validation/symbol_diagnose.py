"""Separate exact proposal coverage, conditional source ranking and possibility of causal first crossing."""
from argparse import ArgumentParser
from bisect import bisect_left
from collections import Counter
from pathlib import Path
import numpy as np
import json
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.symbol_model import Bank,case_geometry,anchor
from research.reading_validation.symbol_geometry import OUT,competing
from research.reading_validation.symbol_evaluate import weights,probabilities,decode
from research.reading_validation.symbol_features import canonical


def diagnostic(bank,split,weights,cut):
    count=Counter();opening=Counter();duals=set();docs=set();families=set();papers=set();rank=[];off=[];source_dist=[];issue_events=[];ambig=[];pair_seq=[];geometry=Counter()
    for page,doc in bank.pages(split):
        spans=[(m.start,m.end) for m in page.mentions];loc={k:i for i,k in enumerate(spans)};names=[canonical(page.text[a:b]) for a,b in spans];true=case_geometry(page,doc);gold={(m.start,m.end):t for m,d,t in true};count['eligible_first_forward_events']+=len(gold)
        for r in doc['issues']:opening[r['kind']]+=1
        for j,(a,_,ts) in enumerate(true):
            for b,_,s in true[j+1:]:
                if max(a.end,b.end)<=min(page.start[ts[0]],page.start[s[0]]) and len(np.intersect1d(ts,s)):
                    count['simultaneously_visible_with_overlapping_definition_tokens']+=1
        if not len(gold):
            count['true_none_paragraphs']+=1
        forecast=np.full((len(spans),page.n),-1.)
        for i,m in enumerate(page.mentions):
            ts=np.arange(bisect_left(page.end,m.committed),page.n)
            if len(ts):forecast[i,ts]=probabilities(page,m,ts,weights)[:,2]
        events=decode(forecast,names,cut);out={spans[i]:t for i,t,v in events};canonical_out={names[i]:(spans[i],t) for i,t,v in events};any_zone=set.union(set(),*[set(q) for q in gold.values()])
        for key,t in out.items():
            if key in gold:
                z=gold[key];distance=min(map(abs,z-t));off.append(distance)
                if t in z:count['correct_source_and_time']+=1
                elif t<z[0]:count['correct_source_too_early']+=1
                else:count['correct_source_too_late']+=1
                if distance<=3:count['correct_source_within3_tokens']+=1
            elif t in any_zone:count['other_source_at_any_gold_evidence']+=1
            else:count['other_source_outside_all_evidence']+=1
        if len(gold)==0:
            count['true_none_paragraphs_ever_flagged']+=bool(events)
        pairs=competing(doc['issues'])
        if not len(pairs):continue
        docs.add(doc['id']);papers.add(doc['paper'])
        ranks={}
        for r in {tuple(q['sources'][0]) for pair in pairs for q in pair}:
            if r not in gold:continue
            ends=gold[r];i=loc.get(r);m=anchor(page,r);name=canonical(page.text[r[0]:r[1]]);start=bisect_left(page.end,m.committed);z=np.full(page.n,-1.)
            pp=np.arange(start,page.n)
            if i is None:z[pp]=probabilities(page,m,pp,weights)[:,2]
            else:z=forecast[i]
            x=z[ends].max();before=z[:ends[0]].max(initial=-1);cross=np.flatnonzero(z>=cut)
            best=float(z[start:].max());inside=x>=best-1e-10;local_top=x>before;correct_actual=r in out and out[r] in ends
            ended=ends[0];available=np.flatnonzero((forecast[:,ended]>=0));rr=forecast[:,ends].mean(axis=1)
            if len(available):
                k=available[int(rr[available].argmax())];exact_top=spans[k]==r;right_name=names[k]==name
                sorting=1+np.count_nonzero(rr[available]>rr[i]+1e-9) if i is not None else None
            else:exact_top=right_name=False;sorting=None
            ranks[r]=dict(available=i is not None,name=name,prior=before,peak=x,oracle_cut=local_top,cross=(len(cross)>0 and cross[0] in ends),peak_inside=inside,conditional_top=exact_top,conditional_name=right_name,actual=correct_actual,zone=ends.tolist())
            if sorting is not None:rank.append(int(sorting));source_dist.append(len(available))
            key=(doc['id'],r);families.add(key);issue_events.append(dict(paper=doc['paper'],document=doc['id'],source=r,rank=sorting,candidates=len(available),**ranks[r]))
        for r,s in pairs:
            a,b=tuple(r['sources'][0]),tuple(s['sources'][0])
            if a not in ranks or b not in ranks:continue
            x,y=ranks[a],ranks[b];count['all_dual']+=1
            tests=dict(exact_two_source_proposals=x['available'] and y['available'],different_canonical_names=x['name']!=y['name'],first_visible_by_both_definitions=max(a[1],b[1])<=min(r['definitions'][0][0],s['definitions'][0][0]),
                 supplied_two_sources_independent_fixed_cut=x['cross'] and y['cross'],supplied_two_sources_each_has_a_possible_causal_cut=x['oracle_cut'] and y['oracle_cut'],supplied_two_sources_share_a_possible_causal_cut=max(x['prior'],y['prior'])<min(x['peak'],y['peak']),both_offline_maxima_inside_gold=x['peak_inside'] and y['peak_inside'],both_exact_source_top_among_all_at_gold=x['conditional_top'] and y['conditional_top'],both_canonical_names_top_at_gold=x['conditional_name'] and y['conditional_name'],both_detected_in_full_actual_decoder=x['actual'] and y['actual'])
            tests['actually_representable']=tests['exact_two_source_proposals'] and tests['different_canonical_names'] and tests['first_visible_by_both_definitions'] and all(page.mentions[loc[k]].committed<=page.end[min(t['zone'])] for k,t in ((a,x),(b,y)) if k in loc)
            for name,val in tests.items():
                count[name]+=bool(val)
                if tests['actually_representable']:geometry[name]+=bool(val)
            pair_seq.append(dict(document=doc['id'],paper=doc['paper'],source_a=a,source_b=b,**tests))
    return dict(split=split,papers_with_two_sources=len(papers),paragraphs_with_two_sources=len(docs),unique_source_lifetimes_in_duals=len(families),article_statuses=dict(opening),counts=dict(count),among_representable=dict(geometry),median_intervening_or_nested_candidates=float(np.median(source_dist)) if source_dist else None,median_correct_rank=float(np.median(rank)) if rank else None,median_absolute_token_error_correct_source=float(np.median(off)) if off else None,source_details=issue_events,pair_details=pair_seq)


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args();bank=Bank(args.folder);w=weights(read_json(OUT/'coefficients.json')['parameters']);cut=read_json(OUT/'thresholds.json')['thresholds']['native']
    for s in ('held','external','calibration'):
        result=diagnostic(bank,s,w,cut);result=json.loads(json.dumps(result,default=lambda x:x.item()));write_json(OUT/('timing-diagnostic-'+s+'.json'),dict(code=digest(Path(__file__)),source=digest(OUT/'thresholds.json'),**result))
