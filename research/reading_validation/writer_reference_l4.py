"""Exercise the production article reviewer and its manually selected reference link on an L4."""
from pathlib import Path
import json
from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.local_forecast import CHECKPOINT
from research.reading_validation.local_inference import LocalInference
from research.reading_validation.narrative_inference import MarginalNarrativeInference
from research.reading_validation.transition_inference import TransitionInference
from research.reading_validation.writer_verify import DraftText
from research.token_readout.runner import TokenRunner
from research.token_readout.writing_review import NARRATIVE_EXAMPLE,render

def run(runtime,reader,packet,source,out):
    out.mkdir();cases=json.loads(source.read_bytes())
    runner=TokenRunner(runtime,reader,None,packet['priors'],CHECKPOINT,scale=packet['scale'],anchor_width=reader.anchor_tokens,
                       english_head=LocalInference.load(),transition_head=TransitionInference.load(),narrative_head=MarginalNarrativeInference.load())
    records={};n=0
    for i,case in enumerate(cases,1):
        text=case['text'];entry=runner.review_english(text,references=True,narrative=case['narrative'])
        measured=entry['english']['references'];html=render(text,entry['english']);read=DraftText();read.feed(html);read.close()
        if ''.join(read.parts)!=text or read.depth or measured['scope']!='given_prior_reference' or runner.forward_count()!=i:
            raise ValueError('Manual earlier-phrase display changed the text or repeated the backbone')
        if not measured['pronouns'] and measured['links']:raise ValueError('A pronoun-free article acquired a link')
        n+=len(measured['links']);records[str(i)]=entry
    if not n or all(r['english']['references']['pronouns'] for r in records.values()):raise ValueError('Both linked and pronoun-free native inputs are required')
    write_json(out/'responses.json',records)
    write_json(out/'verification.json',dict(forwards=runner.forward_count(),articles=len(records),emitted=n,source=digest(source),files={'responses.json':digest(out/'responses.json')},scope='given_prior_reference'))
