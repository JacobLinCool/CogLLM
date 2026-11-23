"""Verify every original movie-word probability through the ambiguity-aware writer path."""
from hashlib import sha256
import json
from pathlib import Path

from research.reading_validation.narrative_inference import MARGINAL_SHA,MarginalNarrativeInference
from research.reading_validation.transition_narrative_verify import run as verify_words

PACKET='b915ebd74440c810817ba7e9de90e247531fae73b04bc3270b08863a9f3e2d25'


def run(runtime,reader,checkpoint,input_path,output_dir):
    if sha256(Path(input_path).read_bytes()).hexdigest()!=PACKET:raise ValueError('Changed complete movie forecasts')
    head=MarginalNarrativeInference.load()
    verify_words(runtime,reader,checkpoint,input_path,output_dir,model=head)
    path=Path(output_dir)/'receipt.json'
    result=json.loads(path.read_text())
    result.update(verified_bundle_sha256=MARGINAL_SHA,threshold=head.threshold,recipe='exact-observed-annotation-set')
    path.write_text(json.dumps(result,indent=2)+'\n')
