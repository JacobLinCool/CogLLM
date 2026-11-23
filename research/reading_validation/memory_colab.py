"""Package registered memory fine-tuning and independent reading probes for one owned L4."""

import argparse
import ast
from importlib.util import resolve_name
from pathlib import Path

from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest,read_json
from research.reading_validation.identity_colab import CHECKPOINT,DATA,ROOT
from research.token_readout.colab import pack_job

PROTOCOL=Path(__file__).with_name('memory-protocol.md')


def dependencies(entry):
    pending=[entry];found=set()
    while pending:
        path=pending.pop()
        if path in found:continue
        found.add(path);node=ast.parse((ROOT/path).read_bytes());pkg='.'.join(Path(path).parent.parts)
        for x in ast.walk(node):
            if isinstance(x,ast.Import):names=[a.name for a in x.names]
            elif isinstance(x,ast.ImportFrom):
                base=resolve_name('.'*x.level+(x.module or ''),pkg);names=[base,*[base+'.'+a.name for a in x.names]]
            else:continue
            for name in names:
                if name.startswith('research.'):
                    stem=name.replace('.','/')
                    for leaf in (stem+'.py',stem+'/__init__.py'):
                        if (ROOT/leaf).is_file() and leaf not in found:pending.append(leaf)
    return sorted(found)


def prepare():
    meta=read_json(ROOT/'research/results/reading-validation-memory/input-seal.json')
    inputs={'curve.pt':CHECKPOINT}
    if meta['protocol']!=digest(PROTOCOL):raise ValueError('Memory protocol changed')
    for leaf in ('cad-train.jsonl','cad-test.jsonl','stories.jsonl'):
        p=DATA/'memory'/leaf
        if digest(p)!=meta['files'][leaf]:raise ValueError('Memory input differs from the registered SHA')
        inputs['memory/'+leaf]=p
    rec=read_json(ROOT/'research/results/reading-validation-recognition/input-seal.json')
    refs={'stimuli.jsonl':(DATA/'recognition/stimuli.jsonl',rec['stimuli_sha256']),
          'narrative-serving-stimuli.jsonl':(DATA/'narrative-serving-stimuli.jsonl','298e8f9490dfc0a7a60e77a426a5ff6b9a5047fdffa6e60c2669f0600565658d'),
          'ats-comprehension/stimuli.jsonl':(DATA/'ats-comprehension/stimuli.jsonl','d2d13db6f3ac472d7fd6c92db136a0a6db9bd9e1d6c0f5517ae8e3d886c20de1'),
          'ats-comprehension/lexical.npz':(DATA/'ats-comprehension/lexical.npz','caef04fb8b90154e208e528fee39fc2ccba96cd34a6945fac836771427784254')}
    for name,(p,sha) in refs.items():
        if digest(p)!=sha:raise ValueError(f'Companion input mismatch: {name}')
        inputs[name]=p
    for leaf in ('reading-validation-local/serving/manifest.json','reading-validation-local/serving/weights.npz',
                 'reading-validation-local/serving/lexicon.json.gz','reading-validation-transition-external/readouts.json',
                 'reading-validation-transition-snac/serving/character-word.json'):
        p='research/results/'+leaf;inputs[p]=ROOT/p
    pack_job(DATA/'l4',PROTOCOL,'reading-memory','reading-validation-memory',dependencies('research/reading_validation/memory_l4.py'),
             inputs,{},10800,{'purpose':'Last global attention LoRA, human response timing and word-level serving tests','curve_sha256':digest(CHECKPOINT)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run'));p.add_argument('job',nargs='?',type=Path);args=p.parse_args()
    if args.action=='prepare':prepare()
    else:run(args.job,protocol=PROTOCOL)
