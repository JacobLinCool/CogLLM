"""Package the direct sufficiency transfer test with outcome-blind native writer verification."""
from research.backbone_transfer.extract import digest,read_json
from research.reading_validation.identity_colab import CHECKPOINT,ROOT
from research.reading_validation.memory_colab import dependencies
from research.reading_validation.reason_quality_data import DATA,OUT,PLAN
from research.token_readout.colab import pack_job
from research.token_readout.writing_review import NARRATIVE_EXAMPLE

def prepare():
    seals=read_json(OUT/'material.json');source=DATA/'stimuli.jsonl'
    if seals['source']!=digest(source) or seals['protocol']!=digest(PLAN):raise ValueError('Unverified original reasoning archive')
    items={'stimuli.jsonl':source,'curve.pt':CHECKPOINT}
    for group,files in [('reading-validation-local/serving',('manifest.json','weights.npz','lexicon.json.gz')),
                    ('reading-validation-transition-external',('readouts.json',)),('reading-validation-transition-partial/serving',('character-word.json',)),
                    ('reading-validation-transition-pointer/phrases',('source-boundary-model.json',)),
                    ('reading-validation-transition-pointer/serving',('word-pointer.json','averaged_perceptron_tagger_eng.zip'))]:
        for f in files:
            rel='research/results/'+group+'/'+f;items[rel]=ROOT/rel
    examples=[dict(text=NARRATIVE_EXAMPLE,narrative=x) for x in (False,True)]
    examples.extend([dict(text='Zoë visited Café René with her friend. She smiled.',narrative=False),dict(text='The purple flowers were already blooming in the garden.',narrative=False)])
    pack_job(DATA.parent/'l4',PLAN,'reading-reason-quality','reading-validation-reason-quality',dependencies('research/reading_validation/reason_quality_l4.py'),
        items,{'manual.json':examples},4500,{'purpose':'Estimate human sufficiency and verify conditional writer links from one causal pass','materials':seals['source']})
if __name__=='__main__':prepare()
