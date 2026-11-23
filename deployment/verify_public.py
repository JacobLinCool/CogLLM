"""Compare the public editor's Gradio response to the sealed L4 editor output."""

import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


REFERENCE = Path(__file__).resolve().parents[1] / "research/results/reading-validation-writer/responses.json"
PUBLIC = "https://jacoblincool-cogllm.hf.space"


def review(server, text):
    url = server.rstrip("/") + "/gradio_api/call/review_english"
    request = Request(url, data=json.dumps({"data": [text]}).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=240) as response:
        event = json.load(response)["event_id"]
    with urlopen(f"{url}/{event}", timeout=600) as response:
        kind = None
        for raw in response:
            line = raw.decode().rstrip("\r\n")
            if line.startswith("event: "):
                kind = line[7:]
            if line.startswith("data: ") and kind in {"complete", "error"}:
                payload = json.loads(line[6:])
                if kind == "error":
                    raise RuntimeError(f"The public Gradio queue returned an error: {payload}")
                (html,) = payload
                return html
    raise RuntimeError("Gradio closed the event stream before completing the review.")


class MarkedDraft(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_draft = False
        self.words = []
        self.ratings = []
        self.underlines = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "div" and "draft" in classes:
            self.in_draft = True
        if self.in_draft and tag == "mark":
            match = re.search(r"Estimated ease ([\d.]+) of 5", attrs["title"])
            if match:
                self.ratings.append(float(match[1]))
        if self.in_draft and "phrase" in classes:
            self.underlines.append((int(attrs["data-start"]), int(attrs["data-end"])))

    def handle_endtag(self, tag):
        if self.in_draft and tag == "div":
            self.in_draft = False

    def handle_data(self, data):
        if self.in_draft:
            self.words.append(data)


def verify(server):
    records = json.loads(REFERENCE.read_text())
    samples = []
    for record in records.values():
        text = record["text"]
        result = review(server, text)
        draft = MarkedDraft()
        draft.feed(result)
        draft.close()
        if "".join(draft.words) != text:
            raise AssertionError("The public editor changed the submitted passage.")
        sentences = record["english"]["sentences"]
        expected = [(s["phrases"][0]["start"], s["phrases"][0]["end"]) for s in sentences if s["phrases"]]
        if expected != draft.underlines:
            raise AssertionError("The public underlines differ from the measured L4 output.")
        scores = [s["rating"]["ease"] for s in sentences if s["rating"]]
        error = max(abs(a - b) for a, b in zip(scores, draft.ratings, strict=True))
        if error > 0.02:
            raise AssertionError(f"Displayed sentence ease differs from the L4 reference by {error}.")
        samples.append({"sentences": len(scores), "matching_underlines": len(expected),
                        "max_error_on_displayed_ease": round(error, 6)})
    return {"server": server, "route": "/review_english", "reference_passages": samples}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=PUBLIC)
    args = parser.parse_args()
    print(json.dumps(verify(args.url), indent=2))
