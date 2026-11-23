"""Export explicitly aligned human and locked model outcomes at individual Provo word IDs."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from research.backbone_transfer.extract import digest, read_json, write_json, stream_rows, write_rows
from research.reading_validation.cloze_population_data import OUT,ROOT
from research.reading_validation.cloze_population_fit import data
from research.reading_validation.cloze_population_model import predictions
from research.reading_validation.cloze_population_score import opened


def main(native):
    scores=read_json(OUT/'scores.json');lock=read_json(OUT/'forecast-lock.json')
    if scores["forecasts"]!=digest(OUT/"forecast-lock.json"):
        raise ValueError("Changed scored forecast")
    raw,x,u,v,tc,meta=data(native)
    sel=np.flatnonzero(~meta["source"])
    obs, exact, bad, _=opened(meta,tc)
    fit=read_json(OUT/'fold-5.json')
    signature,rows=predictions(sel,raw,x,fit['geometry'],fit['models'],u,v,obs,meta['targets'])
    if signature!=lock['external_forecast_sha256']:
        raise ValueError("Model distribution differed")
    geometry=read_json(ROOT/'geometry.json')
    texts={r['id']:r for r in stream_rows(ROOT/'semantic-stimuli.jsonl')}
    wordmap={t['source_id']:(k,g,t) for k,g in geometry.items() if g['source']=='Provo' for t in g['items']}
    results=[]
    for j,ii in enumerate(sel):
        uid=str(meta['ids'][ii]);key,g,t=wordmap[uid];d=texts[key]
        pos=t['word'];a,b,first,last=d['ends'][pos];ap,bp,_,prev_t=d['ends'][pos-1]
        S={k:vals[j] for k,vals in rows.items()}
        if any(r[0]!=ii for r in S.values()):raise ValueError("Mismatched word")
        n,granted=S['context'][1],S['context'][4]
        results.append(dict(id=uid,entity=key,Text_ID=g['article'],target_ordinal_zero_based=pos,
              normalized_reference_word=t['target'],shown_target=d['text'][a:b],target_chars=[a,b],target_tokens=[first,last],
              preceding_print=d['text'][ap:bp],preceding_chars=[ap,bp],preceding_last_token=prev_t,
              state_at_token=first-1,adjacent_tokens=(prev_t==first-1),full_article=g["complete_article_lexeme_alignment"],
              normative_denominator_verified=uid not in bad,reported_mismatch=bad.get(uid),
              released_nonempty_response_mass=n,human_category_disagreement=1-granted,human_spelling_disagreement=exact[ii][1],
              probability_actual_word={k:(r[8] if r[8]>=0 else None) for k,r in S.items()},
              model_category_disagreement={k:1-r[3] for k,r in S.items()},other_mass={k:r[10] for k,r in S.items()}))
    path=ROOT/'external-scored-word-rows.jsonl'
    write_rows(path,results)
    write_json(OUT/'word-export.json',dict(file=digest(path),rows=len(results),outcomes=digest(OUT/'scores.json'),
              coordinates="Disagreement indexed by target t applies BEFORE t. The signed change at consumed word t equals disagreement_before(t+1) minus disagreement_before(t).",
              listed_word_index_base=0,model_positions='last visible prefix token; same state and no answer decoding'))
    print(path,len(results),sum(q['normative_denominator_verified'] for q in results))


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('native',type=Path);main(p.parse_args().native)
