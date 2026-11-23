"""Package the selected writer bundle and separate closed text inputs for native replay."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest, read_json
from research.reading_validation.identity_colab import ROOT, DATA, CHECKPOINT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PROTOCOL = Path(__file__).with_name("author-context-runtime.md")


def prepare():
    files = {}

    def add(name, p, expected):
        if digest(p) != expected:
            raise ValueError("Changed previously sealed input: " + name)
        files[name] = p

    selected = ROOT / "research/results/reading-validation-state-transport/serving"
    for name in ("spec.json", "weights.npz", "expected.json"):
        p = selected / name
        add(
            str(p.relative_to(ROOT)),
            p,
            "d9ff8c55beb2e0094fd5875efa998258d6609b207d6805dfbd06cfe9c04ce5c8"
            if name == "spec.json"
            else digest(p),
        )
    add(
        "stimuli.jsonl",
        DATA / "world-states/serving-stimuli.jsonl",
        "504c80040af4dfb9efc0373e6d684dc597b400e2a76242bef74cad8b15cd814c",
    )
    add(
        "last-movie.jsonl",
        DATA / "last-movie-stimuli.jsonl",
        "8bfdc6b25fbb813d04f854eac623cbe8b1ace3f5441c41b3fe2bade652e32000",
    )
    add(
        "mandarin-lexical-prior.jsonl",
        DATA / "mandarin-lexical-prior-stimuli.jsonl",
        "ec90fab2e5aa2a2aea2a4485891309d911edc61cde31a81a02b949d0b091cfe2",
    )
    add(
        "referent-sketch.jsonl",
        DATA / "referent-sketch-exposure.jsonl",
        "b12b3dc1ee66b2fa25eadfbdefc3a690405378b3c4ace34abf969c9e4adc54ca",
    )
    signature = (
        ROOT / "research/results/reading-validation-shared-referent/exposure-seal.json"
    )
    if (
        read_json(signature)["protocol_sha256"]
        != "cf20f50dd8da30deabd0fe2aec8d02bfef1ed563b4c1afbc24bd08003040541a"
    ):
        raise ValueError("Different source protocol hash")
    add("referent-seal.json", signature, digest(signature))
    add(
        "curve.pt",
        CHECKPOINT,
        "c1704d4436d767fa43bdb59e9216698514e1f6bd1d45cad4b60d0f018d84abb5",
    )
    pack_job(
        DATA / "l4",
        PROTOCOL,
        "reading-author-context",
        "reading-validation-author-context",
        dependencies("research/reading_validation/author_context_l4.py"),
        files,
        {},
        3600,
        {
            "purpose": "Same-pass physical-state serving and independent contextual readouts"
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
        run(a.job, protocol=PROTOCOL)
