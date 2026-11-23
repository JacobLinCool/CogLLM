"""Check all per-source flows and every prefix value against the sealed offline reference."""
from pathlib import Path
import numpy as np
from research.backbone_transfer.extract import digest,stream_rows,write_json
from research.reading_validation.stock_inference import ExpectedConceptLedger


def maximum_error(a,b):
    x=np.asarray(a);y=np.asarray(b)
    if x.shape!=y.shape:raise ValueError('Posterior coordinate mismatch')
    return float(np.max(np.abs(x-y),initial=0))


def run(runtime,reader,checkpoint,input_path,output_dir):
    source=Path(input_path);out=Path(output_dir);out.mkdir()
    coefficients=ExpectedConceptLedger.load();checks=[];counter=0;reference=source.with_name('stock-replay.npz')
    def seen(_module,_args):
        nonlocal counter
        counter+=1
    with runtime.encoder.register_forward_pre_hook(seen),np.load(reference) as refs:
        for row in stream_rows(source):
            key=row['id'];full=coefficients.analyze(row['text'],runtime,True)
            prefix=coefficients.analyze(row['text'][:row['cut']],runtime,True)
            active=np.asarray(full['expected_pending_by_word_and_source']);closed=np.asarray(full['resolved_mass_by_word_and_source'])
            m=len(prefix['sources']);pp=prefix['tokens'];tt=full['tokens'][:len(pp)]
            if [x['span'] for x in pp]!=[x['span'] for x in tt] or [x['span'][0] for x in full['sources']]!=refs[key+'/source_char_onsets'].tolist():
                raise ValueError('Source or token identity changed')
            if [x['span'] for x in full['sources'][:m]]!=[x['span'] for x in prefix['sources']]:
                raise ValueError('A later sentence changed a source anchor')
            replay=max(maximum_error(active,refs[key+'/active']),maximum_error(closed,refs[key+'/resolved']),
                       maximum_error([w['opening_mass'] for w in full['sources']],refs[key+'/open']))
            truncated=max(maximum_error(active[:m,:m],prefix['expected_pending_by_word_and_source']),
                          maximum_error(closed[:m,:m],prefix['resolved_mass_by_word_and_source']),
                          maximum_error([t['expected_pending'] for t in pp],[t['expected_pending'] for t in tt]))
            balance=np.cumsum([t['opening_mass']-t['resolved_mass'] for t in full['tokens']])
            error=maximum_error(balance,[t['expected_pending'] for t in full['tokens']])
            if replay>.003 or truncated>.003 or error>1e-8:raise ValueError(f'Posterior parity, invariance, or conservation failed: {replay}, {truncated}, {error}')
            checks.append({'id':key,'tokens':len(full['tokens']),'sources':len(full['sources']),'prefix_tokens':len(pp),
                           'replay_max_error':replay,'prefix_max_error':truncated,'conservation_max_error':error})
    if counter!=2*len(checks) or counter==0:raise ValueError('Unexpected number of article/prefix calls')
    write_json(out/'receipt.json',{'source':digest(source),'posterior_references':digest(reference),'checks':checks,'forwards':counter,
        'per_presentation':1,'revision':runtime.revision,'word_complete_updates_only':True})
