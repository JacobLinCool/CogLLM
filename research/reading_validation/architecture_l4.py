"""Run the three-backbone narrative replication, with independent Qwen-only studies."""
from argparse import ArgumentParser
import gc
from pathlib import Path
import torch

from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.transition_architecture_l4 import run as architecture
from research.reading_validation.definition_native import run as original
from research.reading_validation.exposure_forecast import run as exposure
from research.reading_validation.definition_forecast import run as textbooks
from research.reading_validation.local_help_l4 import run as help_preferences
from research.reading_validation.revision_forecast import run as revision
from research.reading_validation.identity_run_l4 import run_phases


def run(model,candidate,stimuli,output,protocol_sha256):
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_name()!='NVIDIA L4':raise ValueError('Expected one Colab L4')
    output.mkdir()
    architecture(model,candidate,stimuli,Path('span.pt'),output/'architecture-forecast',protocol_sha256)
    receipt=read_json(output/'architecture-forecast/receipt.json')
    meta={'candidate':candidate,'model_revision':receipt['revision'],'device':receipt['device'],'documents':receipt['documents'],
          'one_article_forward':receipt['forwards']==receipt['documents']}
    if candidate=='qwen':
        gc.collect();torch.cuda.empty_cache()
        jobs=[('help-controls',help_preferences,Path('guo-help-controls.jsonl')),('relay-help',help_preferences,Path('relay/local-stimuli.jsonl')),
              ('noun',revision,Path('noun/local-stimuli.jsonl')),('exposure',exposure,Path('exposure/stimuli.jsonl')),
              ('native',original,Path('stock-native-stimuli.jsonl')),('w00',textbooks,Path('w00-stimuli.jsonl'))]
        run_phases(model,candidate,output/'companion',protocol_sha256,jobs)
        meta['companion']=read_json(output/'companion/causality.json')['studies']
    write_json(output/'causality.json',meta)
    write_json(output/'seal.json',{'protocol_sha256':protocol_sha256,**meta,
        'files_sha256':{str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}})


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for n in ('model','stimuli','output'):p.add_argument('--'+n,required=True,type=Path)
    for n in ('candidate','protocol-sha256'):p.add_argument('--'+n,required=True)
    run(**vars(p.parse_args()))
