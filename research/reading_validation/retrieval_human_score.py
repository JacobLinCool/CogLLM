"""Score a sealed matched-response model against the two directed reading cohorts."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.reading_validation.memory_data import BASE, sha
from research.reading_validation.memory_score import reveal, summarize


def run(folder, focus, dest):
    seal=json.loads((folder/"seal.json").read_bytes())
    if any(sha(folder/name)!=value for name,value in seal["files"].items()):
        raise ValueError("Forecast archive differs from the lock")
    rows=json.loads((BASE/"memory/study-map.json").read_bytes())
    methods=seal["models"]
    result={}
    with np.load(folder/"forecasts.npz",allow_pickle=False) as forecasts:
        for study in (2,3):
            reports=reveal(study,rows)
            result[str(study)]=summarize(forecasts,reports,31780+study,cohorts=("strict",),
                                        methods=methods,focus=focus,controls=[m for m in methods if m!=focus])
    dest.write_text(json.dumps(dict(forecast=sha(folder/"seal.json"),cohorts=result),indent=2,allow_nan=False)+"\n")
    for s,study in result.items():
        print(s,{m:round(study["models"]["strict"][m]["condition_effect"]["estimate"],4) for m in methods})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("focus")
    parser.add_argument("dest",type=Path)
    args=parser.parse_args()
    run(args.folder,args.focus,args.dest)
