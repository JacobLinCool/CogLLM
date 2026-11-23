"""Validate the original retrieval inputs and append a separately frozen news-article assessment."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest,read_json
from research.reading_validation.identity_colab import CHECKPOINT,DATA,ROOT
from research.reading_validation.memory_colab import dependencies
from research.reading_validation.retrieval_colab import PROTOCOL
from research.token_readout.colab import pack_job


def prepare():
    prior=DATA/'l4/cogllm-reading-retrieval-20261122T174036Z-3a981e'
    first=read_json(prior/'plan.json');gold=first['payload_members_sha256']
    if first['protocol_sha256']!=digest(PROTOCOL):raise ValueError('Retrieval protocol was modified')
    inputs={'stimuli.jsonl':DATA/'retrieval/packet.jsonl','curve.pt':CHECKPOINT,'frozen.pt':DATA/'l4/cogllm-reading-memory-20261122T170006Z-b202a3/extraction/memory-forecast/frozen.pt'}
    for name,p in inputs.items():
        if gold[name]!=digest(p):raise ValueError(f'Unmatched sealed retrieval input {name}')
    for file in dependencies('research/reading_validation/retrieval_l4.py'):
        if digest(ROOT/file)!=gold[file]:raise ValueError(f'Changed original retrieval source dependency {file}')
    guardian=read_json(ROOT/'research/results/reading-validation-guardian/input-seal.json')
    source=DATA/'guardian/matched-stimuli.jsonl';protocol=Path(__file__).with_name('guardian-protocol.md')
    if guardian['stimuli']!=digest(source) or guardian['protocol']!=digest(protocol):raise ValueError('News source or directions changed')
    inputs.update({'guardian-stimuli.jsonl':source,'guardian-protocol.md':protocol})
    supplemental=DATA/'relay/summary-stimuli.jsonl'
    worker=Path(__file__).with_name('relay_summary_l4.py')
    if digest(supplemental)!='955fbebd97551c526d0149e2ccbdd8b6fd2a30dcf69d456c78fa92a74e5de681' or digest(worker)!='837eefcd736070a9dc9d80471eadce44bffa024a0fad4952c820b249e17fc2c3':
        raise ValueError('Supplemental Cochrane input or serving procedure changed')
    inputs['relay-summary-stimuli.jsonl']=supplemental
    for survey, folders, entries in [
            ('reading-validation-meaning/input-seal.json','exposure/judgment',
             [('judgment/stimuli.jsonl','stimuli.jsonl','stimuli'),('judgment/vocabulary.json','vocabulary.json','vocabulary')]),
            ('reading-validation-definitions/stock-probability-replay-inputs.json','definitions',
             [('stock-replay.jsonl','stock-replay.jsonl','articles'),('stock-replay.npz','stock-replay.npz','references')])]:
        record=read_json(ROOT/'research/results'/survey)
        for name,leaf,key in entries:
            path=DATA/folders/leaf
            if digest(path)!=record[key]:raise ValueError('Model verification or stimulus file changed: '+name)
            inputs[name]=path
    for leaf in ('weights.json','stock-thresholds.json','stock-count-scales.json','stock-probability-weights.json'):
        name='research/results/reading-validation-definitions/'+leaf;inputs[name]=ROOT/name
    for file in ('reading-validation-transition-external/readouts.json','reading-validation-local/serving/manifest.json',
                 'reading-validation-local/serving/lexicon.json.gz','reading-validation-local/serving/weights.npz'):
        name='research/results/'+file;inputs[name]=ROOT/name
    pack_job(DATA/'l4',PROTOCOL,'reading-retrieval-news','reading-validation-retrieval-news',dependencies('research/reading_validation/guardian_retrieval_l4.py'),
             inputs,{},5400,{'purpose':'Original cross-domain memory retrieval and separate untouched news appraisal','original_retrieval_payload':first['payload_sha256'],
             'guardian_protocol':digest(protocol),'curve_sha256':digest(CHECKPOINT)})


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run'));p.add_argument('job',nargs='?',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:run(a.job,protocol=PROTOCOL)
