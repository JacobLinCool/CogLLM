"""Verify normal application, word-level source parity and separately scored goal accessibility."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest, read_json
from research.reading_validation.identity_colab import DATA,CHECKPOINT,ROOT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PLAN=Path(__file__).with_name('population-serving-protocol.md')


def prepare():
    result=ROOT/"research/results/reading-validation-cloze-population"
    meta=read_json(result/"serving-export.json")
    if meta["protocol"]!=digest(PLAN):raise ValueError("Serving plan changed")
    paths={"curve.pt":CHECKPOINT,"stimuli.jsonl":DATA/"cloze-population/serving-materials.jsonl",
           "goal-access.jsonl":DATA/"goal-access-stimuli.jsonl"}
    if digest(paths["stimuli.jsonl"])!=meta["stimuli"] or digest(paths["goal-access.jsonl"])!="77adbf425c89753e085513e1a02009c4a6850bee9e0468beaecccd86bb2fc120":
        raise ValueError("Registered public materials changed")
    if digest(Path(__file__).with_name("transition_goal_access_l4.py"))!="e7f4bdc7a32f54a2215ec4374db5cd5b12169658315f32d8cd89fa16f44215ee":
        raise ValueError("Goal intervention differs")
    files=read_json(DATA/"l4/cogllm-reading-cloze-population-20261122T213600Z-830419/plan.json")["payload_members_sha256"]
    for f,d in files.items():
        if f.startswith("research/results/reading-validation-local/serving/") or f=="research/results/reading-validation-transition-external/readouts.json":
            if digest(ROOT/f)!=d:raise ValueError("Deployed writer feature changed")
            paths[f]=ROOT/f
    for f in (result/"serving").iterdir():paths[str(f.relative_to(ROOT))]=f
    if digest(result/"serving/spec.json")!=meta['readout']:raise ValueError("Wrong readout")
    code=dependencies("research/reading_validation/population_serving_l4.py")
    pack_job(DATA/'l4',PLAN,'reading-population-serving','reading-validation-population-serving',code,paths,{},1500,
             {"purpose":"54 independent articles, three author captures, 96 matched goal interventions"})


if __name__=="__main__":
    p=ArgumentParser();s=p.add_subparsers(dest='action',required=True);s.add_parser('prepare');s.add_parser('run').add_argument('job',type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare()
    else:run(a.job,protocol=PLAN)
