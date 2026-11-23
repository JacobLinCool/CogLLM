"""Package source-aligned, grade-free English word and passage exposures for native Qwen."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import read_json, digest
from research.reading_validation.identity_colab import ROOT, DATA, CHECKPOINT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PLAN = Path(__file__).with_name("cloze-population-protocol.md")


def prepare():
    s = read_json(
        ROOT / "research/results/reading-validation-cloze-population/input-seal.json"
    )
    folder = DATA / "cloze-population"
    extra = DATA / "columbia-order/local-stimuli.jsonl"
    if (
        s["protocol"] != digest(PLAN)
        or digest(extra)
        != "fc9858559c0b86248ca6993621aebcd614efa0e225dc2171d5072da713f81f98"
        or digest(folder / "dictionary.json") != s["dictionary"]
        or digest(folder / "stimuli.jsonl") != s["stimuli"]
    ):
        raise ValueError("Registered source inputs changed")
    files = {
        "stimuli.jsonl": folder / "semantic-stimuli.jsonl",
        "dictionary.json": folder / "dictionary.json",
        "curve.pt": CHECKPOINT,
        "columbia-order.jsonl": extra,
    }
    appendix = read_json(
        ROOT
        / "research/results/reading-validation-cloze-population/semantic-input-seal.json"
    )
    if (
        "stimuli.jsonl" not in files
        or digest(files["stimuli.jsonl"]) != appendix["combined"]
    ):
        raise ValueError("Different independent physiology material")
    prior = read_json(
        DATA / "l4/cogllm-reading-reason-quality-20261122T202059Z-f0ee25/plan.json"
    )["payload_members_sha256"]
    for name, hash0 in prior.items():
        if name.startswith("research/results/"):
            if digest(ROOT / name) != hash0:
                raise ValueError("Changed author observer " + name)
            files[name] = ROOT / name
    add = {
        "balanced-cases.json": (
            DATA / "writer-balanced-cases.json",
            "33d26b3dbd24705717fda215c774cd57e1238287ebaf56f4fb5a7ae6c88c7ba7",
        ),
        "summeval.jsonl": (
            DATA / "summeval/local-stimuli.jsonl",
            "e4dd00c1d357f8ef06ab7bc4cf862868877d029449b54af6c5547bf5db14c40f",
        ),
    }
    for n, h0 in [
        (
            "research/results/reading-validation-transition-partial/serving/character-operating-points.json",
            "be8df42a",
        ),
        (
            "research/results/reading-validation-transition-external/essay-cross-topic/serving.json",
            "77a0634d12f400d1987d425a56f89c27e92d48025210f38c8e9cf7d0034cff13",
        ),
    ]:
        path = ROOT / n
        if not digest(path).startswith(h0):
            raise ValueError("Registered selected head changed")
        add[n] = (path, digest(path))
    add["essay-parity.jsonl"] = (
        DATA / "essay-batched-parity-stimuli.jsonl",
        "10aed7a6247204fac3f8d63365a794c964d5956bff5206e3d32c3442b07863c0",
    )
    add["unseen-nominal.jsonl"] = (
        DATA / "unseen-nominal-exposure.jsonl",
        "552f8d36ca9bb257b92832d3c4dd0033e5617970d114aa8dd2f7ea799301ec47",
    )
    for n, (path, h) in add.items():
        if digest(path) != h:
            raise ValueError("Wrong sealed article or head: " + n)
        files[n] = path
    pack_job(
        DATA / "l4",
        PLAN,
        "reading-cloze-population",
        "reading-validation-cloze-population",
        dependencies("research/reading_validation/cloze_population_l4.py"),
        files,
        {},
        3600,
        {
            "purpose": "Independent held human word distributions and judge passage ordering",
            "curve_sha256": digest(CHECKPOINT),
        },
    )


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare")
    sub.add_parser("run").add_argument("job", type=Path)
    a = p.parse_args()
    if a.action == "prepare":
        prepare()
    else:
        run(a.job, protocol=PLAN)
