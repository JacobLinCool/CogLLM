"""Certify numerical probability parity and export exact authentic whole-word surprisal."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json,write_rows
from research.reading_validation.cloze_population_data import ROOT,OUT
from research.reading_validation.population_inference import DEFAULT


def main(native):
    seal=read_json(OUT/'serving-export.json');ref=ROOT/'serving-reference.jsonl'
    summary=read_json(native/'receipt.json');observed=native/'article-forecasts.jsonl'
    if seal['references']!=digest(ref) or seal['readout']!=digest(DEFAULT/'spec.json') or summary['output']!=digest(observed) or summary['input']!=seal['stimuli']:
        raise ValueError("Native replay provenance mismatch")
    known=read_json(DEFAULT/'dictionary.json')+[None]
    pub={r['id']:r for r in stream_rows(observed)}
    ix={did:{r['word_index']:r for r in d['words']} for did,d in pub.items()}
    join={r['id']:r for r in stream_rows(ROOT/'external-scored-word-rows.jsonl')}
    rows=[];diff=[];ctx=[];rank=[]
    for item in stream_rows(ref):
        d=pub[item["entity"]];r=ix[item["entity"]][item["word"]];entry=join[item["source_id"]]
        if entry['preceding_last_token']!=r['token'] or entry['preceding_chars']!=[r['start'],r['end']]:
            raise ValueError("Native user-facing word differs from measured preceding prefix")
        diff.append(abs(r['after']-item['disagreement']));ctx.append(abs(r['other_after']-item['other']))
        actual=[v['word'] for v in r['top_next']];expected=[known[n] for n in item['top']]
        rank.append(actual[0]==expected[0])
        j=entry['target_ordinal_zero_based']
        rows.append(dict(id=item['source_id'],Text_ID=entry['Text_ID'],uid=item["entity"],target_word=j,word=entry['shown_target'],
                         native_observed_whole_word_surprisal=d['surprisal'][j],
                         before_disagreement=r['after'],before_other=r['other_after'],
                         most_likely_next=r['top_next'],verified_normative_denominator=entry['normative_denominator_verified']))
    if len(rows)!=seal['word_comparisons'] or max(diff+ctx)>.001 or not all(rank):
        raise ValueError(f"Native serving parity failed: {len(rows)}, {max(diff)}, {max(ctx)}, {sum(rank)}")
    output=ROOT/'native-verified-word-surprisal.jsonl';write_rows(output,rows)
    report=dict(registered_export=digest(OUT/'serving-export.json'),native=digest(native/'receipt.json'),article_forwards=len(pub),
                verified_original_word_positions=len(rows),max_disagreement_error=max(diff),max_other_mass_error=max(ctx),matching_modal_category=sum(rank),
                genuine_wholeword_surprisal_file=digest(output),technical_prefix_tests=summary['checks'],training_choices=135905)
    write_json(OUT/'native-serving-verification.json',report);print(report)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('native',type=Path);main(p.parse_args().native)
