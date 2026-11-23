"""Preserve independently certified reading-study outputs through infrastructure interruptions."""
from argparse import ArgumentParser
from pathlib import Path
from time import sleep
import json
from research.reading_validation.copy_finished_phase import main


def collect(job, phases=None):
    remaining=list(phases) if phases else ['learner-writer','discourse-order','convergent-original','semlex','argument','jones','gate-pointer','pointer-writer','semantic-pronoun']
    while remaining:
        status=json.loads((job/'manifest.json').read_text())
        if status.get('release_verified'):return
        if 'allocated' in status['timeline']:
            for key in list(remaining):
                try:main(job,key)
                except RuntimeError:continue
                remaining.remove(key)
        if remaining:sleep(25)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('job',type=Path);p.add_argument('--phases',nargs='+');a=p.parse_args();collect(a.job,a.phases)
