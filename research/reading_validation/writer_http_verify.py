"""Replay hash-verified accelerator outputs through the live, localhost-only writer API."""

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import socket

from research.token_readout.app import CHANNELS, initial_position, parse_analysis, render_article, plot_rows
from research.token_readout.writing_review import STYLE, build_app, render
from research.reading_validation.writer_verify import DraftText
from research.token_readout.physical_view import render as render_physical


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def save_preview(dest, article, title, name, *, attribution=None):
    marked = render(article["text"], article["english"])
    parsed = DraftText()
    parsed.feed(marked)
    parsed.close()
    if parsed.depth or "".join(parsed.parts) != article["text"]:
        raise ValueError("Rendered preview lost text from the source passage")
    document = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
                f'content="width=device-width,initial-scale=1"><title>{title}</title><style>'
                'body{max-width:54rem;margin:6vh auto;padding:0 1rem;font:17px/1.5 system-ui;color:#23352f}'
                + STYLE + f'</style></head><body><main><h1>{title}</h1>{marked}{attribution or ""}</main></body></html>')
    (dest / name).write_text(document)
    return name


class RecordedOutput:
    def __init__(self, sources, physical=None):
        self.records = {}
        self.receipts = []
        self.calls = 0
        self.physical_records = {}
        self.physical_calls = 0
        self.physical_receipt = {}
        if physical is not None:
            receipt = json.loads((physical / "verification.json").read_bytes())
            if digest(physical / "responses.json") != receipt["files"]["responses.json"]:
                raise ValueError("The physical L4 writer recordings have changed")
            captures = json.loads((physical / "responses.json").read_bytes())
            self.physical_records = {r["text"]:r for r in captures.values()}
            if len(self.physical_records) != receipt["article_forwards"] or len(captures) != receipt["articles"]:
                raise ValueError("Each physical story must have one original encoder call")
            self.physical_receipt = receipt
        for source in sources:
            receipt = json.loads((source / "verification.json").read_bytes())
            for name, expected in receipt["files"].items():
                if digest(source / name) != expected:
                    raise ValueError(f"The real L4 verification archive has changed: {name}")
            captures = json.loads((source / "responses.json").read_bytes())
            forwards = receipt["encoder_forwards"] if "encoder_forwards" in receipt else receipt["forwards"]
            if "full_app_forwards" in receipt:
                if (receipt["full_app_forwards"] != len(captures) or
                        receipt["full_app_forwards"] + receipt["direct_serving_forwards"] != forwards):
                    raise ValueError("Unaccounted-for encoder calls in the summary audit")
            elif forwards != len(captures):
                raise ValueError("Required exactly one L4 forward for each captured article analysis")
            for record in captures.values():
                local = record["english"]
                key = (record["text"], "narrative_words" in local, "terminology" in local,
                       "references" in local, "expectations" in local, "summary" in local)
                if key in self.records:
                    raise ValueError("Two captures disagree on who owns the same input and display mode")
                self.records[key] = record
            self.receipts.append((source, receipt, captures))

    def review_english(self, text, *, narrative=False, terminology=False, references=False, expectations=False, summary=False):
        key = (text, narrative, terminology, references, expectations, summary)
        if key not in self.records:
            raise ValueError("This verification server accepts only the captured passages in their original display modes.")
        self.calls += 1
        return self.records[key]


    def physical_story(self, text):
        self.physical_calls += 1
        return self.physical_records[text]["reading"]


def run(source, dest, terms=None, references=None, physical=None, expectations=None, summaries=None, serve=None):
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    import gradio
    from gradio_client import Client
    import httpx

    replay = RecordedOutput([folder for folder in (source, terms, references, expectations, summaries) if folder is not None], physical)
    app = build_app(replay)
    if serve:
        app.launch(server_name="127.0.0.1", server_port=serve, share=False, quiet=True, css=STYLE)
        return
    with closing(socket.socket()) as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    server = "http://127.0.0.1:" + str(port)
    try:
        app.launch(server_name="127.0.0.1", server_port=port, share=False,
                   prevent_thread_lock=True, quiet=True, css=STYLE)
        page = httpx.get(server, trust_env=False).raise_for_status()
        if "CogLLM Writing Review" not in page.text:
            raise ValueError("Gradio returned an unexpected home page")
        client = Client(server, verbose=False)
        for (draft, narrative, terminology, references_on, expectation_on, summary_on), expected in replay.records.items():
            output, trace, plot = client.predict(draft, narrative, terminology, references_on, expectation_on, summary_on,
                                                api_name="/review_english")
            analysis = parse_analysis(draft, expected)
            position, _ = initial_position(analysis)
            if output != render(draft, expected["english"]) or trace != render_article(analysis, position):
                raise ValueError("Gradio changed the validated source markup or experimental trace")
            expected_rows = plot_rows(analysis)
            if (plot["columns"] != list(expected_rows.columns) or plot["mark"] != "line"
                    or len(plot["data"]) != len(analysis.tokens) * len(CHANNELS)
                    or plot["data"] != json.loads(expected_rows.to_json(orient="split", date_unit="ms"))["data"]):
                raise ValueError("Gradio dropped token measurements in transit")
        for text, record in replay.physical_records.items():
            html = client.predict(text, api_name="/physical_story")
            if html != record["html"] or html != render_physical(text, record["reading"]):
                raise ValueError("The physical review route changed the native article display")
    finally:
        app.close()
    if replay.calls != len(replay.records):
        raise ValueError("Gradio unexpectedly repeated an analysis")
    dest.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    previews = []
    for folder, receipt, captures in replay.receipts:
        is_terms = "terminology" in captures["1"]["english"]
        is_reference = "references" in captures["1"]["english"]
        is_expectation = "expectations" in captures["1"]["english"]
        is_summary = "summary" in captures["1"]["english"]
        stash = dest / "reference-source" if is_reference else dest / "terminology-source" if is_terms else dest
        if is_expectation:
            stash = dest / "expectation-source"
        if is_summary:
            stash = dest / "summary-source"
        stash.mkdir(exist_ok=True)
        for name in [*receipt["files"], "verification.json"]:
            target = stash / name
            if target.resolve() != (folder / name).resolve():
                target.write_bytes((folder / name).read_bytes())
            source_hashes[str(target.relative_to(dest))] = digest(target)
        if is_summary:
            previews.append(save_preview(dest,captures["1"],"Can a reader understand this summary?","summary-understanding.html"))
            credit=('<p>Study excerpt from the English WikiLingua test set, presented in '
                    '<a href="https://github.com/google-research-datasets/seahorse">SEAHORSE (Clark et al., 2023)</a>. '
                    'One professional rater answered Yes for this summary. Source identifier '
                    'wiki_lingua_english_en-test-5068, mt5_small. '
                    '<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.</p>')
            previews.append(save_preview(dest,captures["2"],"A summary from the reading study",
                                         "summary-study-example.html",attribution=credit))
            positive_credit=('<p>From <a href="https://github.com/google-research-datasets/seahorse">'
                             'SEAHORSE (Clark et al., 2023)</a>, XSum xsum-test-16, mt5_small. '
                             'Selected for the highest model estimate among the fixed writer examples. '
                             'The original professional rater answered No. '
                             '<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.</p>')
            previews.append(save_preview(dest,captures["3"],"A summary above the review threshold",
                                         "summary-review-example.html",attribution=positive_credit))
        elif is_expectation:
            previews.append(save_preview(dest,captures["1"],"Where readers’ next guesses change","word-expectations.html"))
        elif is_reference:
            examples = [r for r in captures.values() if r["english"]["references"]["links"]]
            if not examples:
                raise ValueError("No observed reference link to display in the verified preview")
            previews.append(save_preview(dest, examples[0], "CogLLM Reference Review", "reference.html"))
        elif is_terms:
            previews.append(save_preview(dest, captures["1"], "CogLLM Scientific Writing Review", "terminology.html"))
        else:
            previews.append(save_preview(dest, captures["1"], "CogLLM Writing Review", "review.html"))
            if "4" in captures and "narrative_words" in captures["4"]["english"]:
                previews.append(save_preview(dest, captures["4"], "CogLLM Narrative Review", "narrative.html"))
    if replay.physical_calls != len(replay.physical_records):
        raise ValueError("The app did not preserve one request per physical story")
    if replay.physical_records:
        from research.token_readout.physical_view import EXAMPLE
        record = replay.physical_records[EXAMPLE]
        doc = ('<!doctype html><html lang="en"><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width,initial-scale=1">'
               '<title>CogLLM Physical Continuity</title><style>'
               'body{max-width:48rem;margin:6vh auto;padding:0 1rem;font:17px/1.55 system-ui;color:#23352f}'
               +STYLE+'</style><main><h1>Do these events fit together?</h1>'+record["html"]+'</main></html>')
        (dest/"physical-story.html").write_text(doc)
        previews.append("physical-story.html")
    checks = dict(gradio=gradio.__version__, api="/review_english", http_status=page.status_code,
                  articles=len(replay.records), result_html_equals_direct_render=True, token_plot_complete=True,
                  model_forwards_on_l4=len(replay.records)+len(replay.physical_records),
                  physical_route="/physical_story", physical_articles=replay.physical_calls,
                  physical_native_receipt=replay.physical_receipt, local_model_loads=0, source_text_roundtrips=True,
                  display_files={name:digest(dest / name) for name in previews},
                  renderer_sha256=digest(Path("research/token_readout/writing_review.py")),
                  terminology_renderer_sha256=digest(Path("research/token_readout/terminology_view.py")),
                  reference_renderer_sha256=digest(Path("research/token_readout/reference_view.py")),
                  summary_renderer_sha256=digest(Path("research/token_readout/summary_view.py")),
                  expectation_renderer_sha256=digest(Path("research/token_readout/expectation_view.py")),
                  source_files=source_hashes)
    (dest/"http-verification.json").write_text(json.dumps(checks,indent=2)+"\n")
    print(json.dumps(checks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--terms",type=Path)
    parser.add_argument("--references",type=Path)
    parser.add_argument("--expectations",type=Path)
    parser.add_argument("--summaries",type=Path)
    parser.add_argument("--physical",type=Path)
    parser.add_argument("--dest",type=Path,default=Path("research/results/reading-validation-writer"))
    parser.add_argument("--serve",type=int)
    run(**vars(parser.parse_args()))
