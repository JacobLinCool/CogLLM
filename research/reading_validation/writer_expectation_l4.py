"""Capture genuine article responses for the word expectation view and verify its causal prefix."""

from pathlib import Path

from research.backbone_transfer.extract import digest, write_json
from research.reading_validation.local_forecast import CHECKPOINT
from research.reading_validation.local_inference import LocalInference
from research.reading_validation.transition_inference import TransitionInference
from research.reading_validation.writer_verify import DraftText
from research.token_readout import expectation_view
from research.token_readout.runner import TokenRunner
from research.token_readout.writing_review import render


def run(runtime, reader, packet, source, out):
    """Run three original examples. The first two have an identical complete prefix."""
    out = Path(out)
    out.mkdir()
    shared = expectation_view.EXAMPLE.rsplit(" A moment later", 1)[0]
    examples = (
        expectation_view.EXAMPLE,
        shared + " Outside, a bus pulled away from the corner.",
        "The painter mixed blue and yellow on her palette. The resulting color was green.",
    )
    runner = TokenRunner(runtime, reader, None, packet["priors"], CHECKPOINT, scale=packet["scale"],
                         anchor_width=reader.anchor_tokens, english_head=LocalInference.load(),
                         transition_head=TransitionInference.load())
    captures = {}
    for i, original in enumerate(examples, 1):
        record = runner.review_english(original, expectations=True)
        markup = render(original, record["english"])
        check = DraftText()
        check.feed(markup)
        check.close()
        if check.depth or "".join(check.parts) != original or runner.forward_count() != i:
            raise ValueError("Lost printed words or violated one encoder call per article")
        captures[str(i)] = record
    earlier = [[r for r in captures[k]["english"]["expectations"]["words"] if r["end"] <= len(shared)]
               for k in ("1", "2")]
    if len(earlier[0]) < 10:
        raise ValueError("Too few matching words to test prefix invariance.")
    max_drift = 0.0
    for left, right in zip(*earlier, strict=True):
        if (left["start"], left["end"], left["text"]) != (right["start"], right["end"], right["text"]):
            raise ValueError("Unchanged prefix acquired a different visible word.")
        max_drift = max(max_drift, abs(left["after"] - right["after"]))
    if max_drift > .005:
        raise ArithmeticError("A new ending changed earlier word expectations beyond numerical tolerance.")
    write_json(out / "responses.json", captures)
    write_json(out / "verification.json", dict(encoder_forwards=runner.forward_count(), candidate=runtime.candidate,
               revision=runtime.revision, same_prefix_positions=len(earlier[0]), prefix_max_drift=max_drift,
               source=__name__, files={"responses.json": digest(out / "responses.json")}))
