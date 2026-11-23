"""Share one frozen Qwen allocation across independent public reader interventions."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.extract import read_json
from research.reading_validation import reader_history_l4,transition_goal_access_l4,character_object_access_l4
from research.reading_validation.identity_run_l4 import run_phases


def run(model,candidate,stimuli,output,protocol_sha256):
    bank=read_json(Path('study-list.json'))
    if bank not in [['character-names'],['character-names','character-objects']]:
        raise ValueError("Unregistered narrative banks")
    phases=[('reader-history',reader_history_l4.run,stimuli)]
    modules={'character-names':transition_goal_access_l4.run,'character-objects':character_object_access_l4.run}
    phases.extend((name,modules[name],Path(name+'.jsonl')) for name in bank)
    run_phases(model,candidate,output,protocol_sha256,phases)


if __name__=='__main__':
    p=ArgumentParser()
    for k in ('model','stimuli','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('candidate','protocol-sha256'):p.add_argument('--'+k,required=True)
    run(**vars(p.parse_args()))
