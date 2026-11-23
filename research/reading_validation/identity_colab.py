"""Prepare and operate one owned L4 for sealed outcome-free feature extraction."""

import argparse
from pathlib import Path

from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest, read_json
from research.token_readout.colab import pack_job
from research.token_readout.transfer_colab import CODE


ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'.cache/cogllm/reading-validation'
PROTOCOL=Path(__file__).with_name('identity-protocol.md')
CHECKPOINT=ROOT/'research/results/adaptive-joint-fixed-retention-l4/guarded-fixed-1729/curve.pt'


def prepare(transfer=False, causal=False, review=False, definitions=False):
    code = (*CODE,'research/token_readout/runner.py',
            *(f'research/reading_validation/{name}.py' for name in
              ('identity_features','identity_run_l4','bridge_forecast','external_forecast','local_forecast')))
    if definitions:
        protocol=Path(__file__).with_name('definition-protocol.md');seal=read_json(ROOT/'research/results/reading-validation-definitions/input-seal.json')
        inputs={'stimuli.jsonl':DATA/'definitions/stimuli.jsonl','curve.pt':CHECKPOINT,'discourse-stimuli.jsonl':DATA/'discourse-stimuli.jsonl'}
        if digest(inputs['discourse-stimuli.jsonl'])!='ca339edb815a029cb1571ebc6ac4e5a636988f9cca5eb7d1643c03ef7d0dd2e8':raise ValueError('Unexpected discourse packet')
        if seal['inputs_sha256']!=digest(inputs['stimuli.jsonl']) or seal['protocol_sha256']!=digest(protocol):
            raise ValueError('The textbook packet changed after registration')
        code=(*code,*(f'research/reading_validation/{n}.py' for n in ('definition_l4','definition_forecast','identity_causal_features')))
        pack_job(DATA/'l4',protocol,'reading-definitions','reading-validation-definitions',code,inputs,{},3600,
                 {'purpose':'Localize term identities and their own later definitions','curve_sha256':digest(CHECKPOINT)})
        return
    if causal or review:
        shared=('transition_external_forecast','transition_matched_forecast','local_inference','local_features','writer_verify')
        diff=('identity_causal_features','identity_causal_l4') if causal else (
            'event_forecast','transition_serving_verify','transition_inference','transition_snac_forecast','revision_forecast','identity_review_l4','transition_external_prepare','transition_external_source')
        code=(*code,*(f'research/reading_validation/{n}.py' for n in (*shared,*diff)),
              'research/stage6/medreadme_span_audit.py','research/semantic_tracking/iirc_anchor_fit.py','research/stage2/records.py',
              'research/semantic_tracking/iirc_prepare.py','research/semantic_tracking/qasper_fit.py','research/core_features/fit.py',
              'research/semantic_tracking/joint_model.py','research/token_readout/app.py','research/token_readout/writing_review.py')
        modules=tuple(dict.fromkeys(code))
        if causal:
            protocol=Path(__file__).with_name('identity-causal-protocol.md')
            inputs={n:DATA/n for n in ('prior-stimuli.jsonl','cohesion-matched-stimuli.jsonl','causal-subsets.json')}
            inputs['stimuli.jsonl']=DATA/'causal-stimuli.jsonl'
            seal=read_json(ROOT/'research/results/reading-validation-causal/input-seal-v2.json')
            if (seal['input_sha256']!=digest(inputs['stimuli.jsonl']) or seal['protocol_sha256']!=digest(protocol)
                  or seal['subsets_sha256']!=digest(inputs['causal-subsets.json'])):
                raise ValueError('Causal inputs differ from their registered hashes')
            suffix='causal'
        else:
            protocol=Path(__file__).with_name('identity-review-protocol.md')
            inputs={n:DATA/n for n in ('transition-serving-stimuli.jsonl','revision/stimuli.jsonl','revision/lexical.npz',
                                        'snac-stimuli.jsonl','event-stimuli.jsonl')}
            inputs['stimuli.jsonl']=DATA/'aries'/'stimuli.jsonl'
            readouts='research/results/reading-validation-transition-external/readouts.json';inputs[readouts]=ROOT/readouts
            seal=read_json(ROOT/'research/results/reading-validation-identity-review/input-seal.json')
            if seal['stimuli']!=digest(inputs['stimuli.jsonl']) or seal['protocol']!=digest(protocol):
                raise ValueError('Review inputs differ from their registered hashes')
            suffix='review'
        inputs['curve.pt']=CHECKPOINT
        for name in ('manifest.json','weights.npz','lexicon.json.gz'):
            path=f'research/results/reading-validation-local/serving/{name}';inputs[path]=ROOT/path
        pack_job(DATA/'l4',protocol,'reading-'+suffix,'reading-validation-'+suffix,modules,inputs,{},5400,
                 {'purpose':'Independent reading studies with one shared frozen Qwen','curve_sha256':digest(CHECKPOINT)})
        return
    if transfer:
        code=(*code,'research/reading_validation/identity_followup_l4.py','research/reading_validation/transition_external_forecast.py')
        inputs={name:DATA/name for name in ('jargon/local-stimuli.jsonl','jargon/local-lexical.npz','conflict-stimuli.jsonl','cohesion-stimuli.jsonl','rewrite-stimuli.jsonl')}
        inputs['stimuli.jsonl']=DATA/'multirc-stimuli.jsonl'
        inputs['curve.pt']=CHECKPOINT
        seal=read_json(ROOT/'research/results/reading-validation-identity-transfer/input-seal.json')
        protocol=Path(__file__).with_name('identity-transfer-protocol.md')
        if seal['inputs_sha256']!=digest(inputs['stimuli.jsonl']) or seal['protocol_sha256']!=digest(protocol):
            raise ValueError('External identity packet or protocol changed')
        pack_job(DATA/'l4',protocol,'reading-transfer','reading-validation-transfer',code,inputs,{},3600,
                 {'purpose':'four external transfer banks','curve_sha256':digest(CHECKPOINT)})
        return
    inputs = {name:DATA/name for name in ('identity-stimuli.jsonl','local-stimuli.jsonl','local-lexical.npz',
                                          'bridge-stimuli.jsonl','external-stimuli.jsonl')}
    seal = read_json(ROOT/'research/results/reading-validation-identity/input-seal.json')
    if seal['files']['stimuli']!=digest(inputs['identity-stimuli.jsonl']) or seal['protocol_sha256']!=digest(PROTOCOL):
        raise RuntimeError('Input or protocol changed after sealing')
    inputs['stimuli.jsonl']=inputs.pop('identity-stimuli.jsonl')
    inputs['curve.pt']=CHECKPOINT
    pack_job(DATA/'l4',PROTOCOL,'reading-validation','reading-validation',code,inputs,{},3500,
             {'purpose':'four outcome-blind feature sets','curve_sha256':digest(CHECKPOINT)})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('job',type=Path,nargs='?')
    flags=parser.add_mutually_exclusive_group();flags.add_argument('--transfer',action='store_true');flags.add_argument('--causal',action='store_true');flags.add_argument('--review',action='store_true');flags.add_argument('--definitions',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':
        prepare(args.transfer,args.causal,args.review,args.definitions)
    else:
        run(args.job,protocol=Path(__file__).with_name('identity-transfer-protocol.md') if args.transfer else Path(__file__).with_name('identity-causal-protocol.md') if args.causal else Path(__file__).with_name('identity-review-protocol.md') if args.review else Path(__file__).with_name('definition-protocol.md') if args.definitions else PROTOCOL)
