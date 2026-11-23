"""Persist a completed Colab phase immediately, checking every file against its scientific receipt."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SPECS = {
    't5-adapter': ('receipt.json', lambda r:{'movie-generations.jsonl':r['forecast_sha256'],'selected-adapter.pt':r['best_adapter_sha256']}),
    'learner-writer': ('verification.json', lambda r:r['files']),
    'word-responses': ('receipt.json', lambda r:{'responses.json':r['output']}),
    'manual-writing': ('verification.json', lambda r:r['files']),
    'physical-writer': ('verification.json', lambda r:r['files']),
    'referent-sketch': ('receipt.json', lambda r:{'word-states.npz':r['archive_sha256']}),
    'mandarin-lexical-prior': ('receipt.json', lambda r:{'word-representations.npz':r['weights_sha256']}),
    'cloze-population': ('receipt.json',lambda r:r['files']),
    'columbia-order': ('receipt.json',lambda r:{'features.npz':r['file']}),
    'summeval-coherence': ('receipt.json',lambda r:{'features.npz':r['file']}),
    'reader-history': ('receipt.json',lambda r:{'logs.npz':r['output']}),
    **{phase: ('receipt.json',lambda r:{'goal-accessibility.jsonl':r['forecast_sha256']}) for phase in ('character-names','character-objects')},
    **{'seahorse-'+phase: ('receipt.json',lambda r:{'features.npz':r['file']}) for phase in ('train0','validation','test','train1','train2','train3')},
    'population-writer': ('verification.json',lambda r:r['files']),
    'population-parity': ('receipt.json',lambda r:{'article-forecasts.jsonl':r['output']}),
    'goal-access': ('receipt.json',lambda r:{'goal-accessibility.jsonl':r['forecast_sha256']}),
    'essay-choices': ('receipt.json',lambda r:{'sentence-features.jsonl':r['file_sha256']}),
    'nominal-reports': ('receipt.json',lambda r:{'word-states.npz':r['archive_sha256']}),
    'nominal-external': ('receipt.json',lambda r:{'word-states.npz':r['archive_sha256']}),
    'summeval-essay-parity': ('receipt.json',lambda r:{'features.npz':r['file']}),
    'balanced-writer': ('verification.json',lambda r:r['files']),
    'last-movie': ('receipt.json', lambda r:{'word-features.npz':r['feature_sha256']}),
    'balanced-narrative': ('receipt.json', lambda r:{}),
    'ambicoref': ('receipt.json', lambda r:{'prior-words.npz':r['prior_forecast_sha256'],'evidence-words.npz':r['evidence_forecast_sha256']}),
    'danish-coherence': ('receipt.json', lambda r:{'features.npz':r['file']}),
    'italian-order-source': ('receipt.json', lambda r:{'features.npz':r['file']}),
    'italian-order-control': ('receipt.json', lambda r:{'features.npz':r['file']}),
    'physiology': ('receipt.json', lambda r:r['files']),
    'multilab': ('receipt.json', lambda r:{'word-states.npz':r['word_states']}),
    'state-transport': ('receipt.json', lambda r:{'features.npz':r['file']}),
    'reasoning': ('receipt.json', lambda r:{'features.npz':r['file']}),
    'mandarin-reading': ('receipt.json', lambda r:{'unprompted-states.npz':r['features_sha256']}),
    'null-choice': ('receipt.json', lambda r:{'all-prior-logits.npz':r['forecasts_sha256'],'factorized-forecasts.json':r['factorized_sha256']}),
    'semlex': ('extraction.json', lambda r:r['files_sha256']),
    'discourse-order': ('receipt.json', lambda r:{'features.npz':r['file']}),
    'argument': ('receipt.json', lambda r:{'features.npz':r['file']}),
    'convergent-original': ('receipt.json', lambda r:{'word-states.npz':r['word_states']}),
    'pointer-writer': ('receipt.json', lambda r:{}),
    'gate-pointer': ('receipt.json', lambda r:{'gate-weights.npz':r['weights_sha256'],'word-forecasts.json':r['forecasts_sha256']}),
    'semantic-pronoun': ('receipt.json', lambda r:{'source-semantic-weights.npz':r['weights_sha256'],'all-word-utilities.npz':r['forecast_sha256']}),
    'jones': ('receipt.json', lambda r:{'all-earlier-word-logits.npz':r['forecast_sha256'],'surprisal.npz':r['surprisal_sha256']}),
    'marginal-narrative': ('receipt.json', lambda r:{}),
    'narrative-writer': ('verification.json', lambda r:r['files']),
    'term-writer': ('verification.json', lambda r:r['files']),
    'emtec': ('extraction.json', lambda r:r['files_sha256']),
    'spanish-expanded': ('receipt.json', lambda r:{'signals.npz':r['signals']}),
    'convergent-words': ('receipt.json', lambda r:{'word-states.npz':r['word_states']}),
    'spanish-clues': ('receipt.json', lambda r:{'signals.npz':r['signals']}),
    'nns-rewrite': ('extraction.json', lambda r:r['files_sha256']),
    'expectation': ('receipt.json', lambda r:{'features.npz':r['file']}),
    'plot-survival': ('receipt.json', lambda r:{'forecasts.npz':r['file']}),
    'nns-dense': ('extraction.json', lambda r:r['files_sha256']),
    'nns-sparse': ('extraction.json', lambda r:r['files_sha256']),
    'retrieval': ('receipt.json', lambda r:r['files']),
    'direct': ('receipt.json', lambda r:{'word-probabilities.npz':r['predictions_sha256'],'weights.npz':r['weights_sha256']}),
    'guardian': ('receipt.json', lambda r:{'predictions.json':r['forecasts']}),
    'muse': ('receipt.json', lambda r:{'native.npz':r['file']}),
    'stock-replay': ('receipt.json', lambda r:{}),
    'relay-summary': ('extraction.json', lambda r:r['files_sha256']),
    'meaning': ('receipt.json', lambda r:{name+'.npz':sha for name,sha in r['sha256'].items()}),
}


def main(job, name):
    manifest = json.loads((job/'manifest.json').read_text())
    if manifest['run_id'] != job.name or 'allocated' not in manifest['timeline']:
        raise ValueError('The selected run was never allocated')
    target=job/'stage-copies'/(name+'-forecast');target.mkdir(parents=True,exist_ok=True)
    def pull(name):
        file=Path(name)
        if not file.parts or file.is_absolute() or any(part in ('..','.') or not all(c.isalnum() or c in '._-' for c in part) for part in file.parts):raise ValueError('Unexpected relative file path')
        dest=target/file
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            part=dest.with_name(dest.name+'.pending')
            remote='/content/'+job.name+'/extraction/'+target.name+'/'+name
            process=subprocess.run(['colab','--auth','adc','download','-s',job.name,remote,str(part)],stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,timeout=150)
            if process.returncode:raise RuntimeError('Verified phase output is not yet reachable')
            part.replace(dest)
        return dest
    filename, index=SPECS[name]
    r=json.loads(pull(filename).read_text())
    for name,sha in index(r).items():
        with pull(name).open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=sha:raise ValueError('Mismatch from remote scientific receipt')
    print(target,flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('job',type=Path);p.add_argument('phase',choices=SPECS)
    args=p.parse_args();main(args.job,args.phase)
