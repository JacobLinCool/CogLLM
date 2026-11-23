"""Freeze English population word-response weights and select outcome-free original texts for serving parity."""
from pathlib import Path
import numpy as np
from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.convergent_word_prepare import OUT,ROOT
from research.reading_validation.identity_prepare import sealed_rows


def run():
    fit=OUT/'adaptive/context-fold-5.json';d=read_json(fit);head=d['models']['shared'];pr=d['projection'];dest=OUT/'serving';dest.mkdir(exist_ok=True);p=dest/'weights.npz'
    np.savez_compressed(p,**{k:np.asarray(pr[k]) for k in ('mean','sd','context_mean','context_scale','basis')},**{k:np.asarray(head[k]) for k in ('B','mu','scale')})
    j=Path('.cache/cogllm/reading-validation/l4/cogllm-reading-game-recovery-20261122T191949Z-f77573/extraction/convergent-words-forecast');native=read_json(j/'receipt.json');map_=list(stream_rows(ROOT/'map.jsonl'));docs={r['id']:r for r in stream_rows(ROOT/'stimuli.jsonl')};selected=sorted(r['id'] for r in map_ if r['language']=='English')[:9];path=ROOT/'word-serving-stimuli.jsonl';sealed_rows(path,[{'id':k,'text':docs[k]['text']} for k in selected]);write_json(dest/'spec.json',dict(weights_sha256=digest(p),fit=digest(fit),evaluation=digest(OUT/'adaptive/scores.json'),revision=native['revision'],training_endpoint='Three independently measured word responses; English noninitial printed words in 5–15-word sentences',point_commit='Last visible native token of the original printed word',complete_sentence_call_count=1,weights=d['models']['shared']['task'],verification_packet=digest(path),source_reference_features=native['word_states']))
    print('Sealed 9 original sentence presentations',p.stat().st_size)


if __name__=='__main__':run()
