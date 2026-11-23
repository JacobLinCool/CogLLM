"""Exercise learner display events against a verified live L4 capture through localhost."""

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import socket

from research.reading_validation.local_forecast import digest
from research.token_readout.learner_view import EXAMPLE, LEVELS, POOLED, TITLE, display
from research.token_readout.learner_view import STYLE as LEARNER_STYLE
from research.token_readout.writing_review import STYLE, build_app


def run(source, dest):
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    import gradio
    from gradio_client import Client
    import httpx

    note = json.loads((source / "verification.json").read_bytes())
    capture = source / "responses.json"
    if digest(capture) != note["files"]["responses.json"]:
        raise ValueError(
            "Serving verification accepts only hash-reproducible native outputs"
        )
    actual = json.loads(capture.read_bytes())
    by_text = {s["text"]: s for s in actual.values()}

    class Replay:
        def __init__(self):
            self.calls = 0

        def learner_sentence(self, t):
            if t not in by_text:
                raise ValueError(
                    "This replay accepts only its recorded native sentences"
                )
            self.calls += 1
            return by_text[t]

    runner = Replay()
    app = build_app(runner)
    with closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    try:
        app.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            prevent_thread_lock=True,
            quiet=True,
            css=STYLE,
        )
        page = httpx.get(origin, trust_env=False).raise_for_status()
        if TITLE not in page.text:
            raise ValueError("Missing learner control on original writing page")
        client = Client(origin, verbose=False)
        api = client.view_api(print_info=False, return_format="dict")["named_endpoints"]
        if not all(k in api for k in ("/review_english", "/learner_vocabulary")):
            raise ValueError("Unexpected loss of the writing or learner endpoint")
        count = 0
        for text, record in by_text.items():
            for level in (POOLED, "A2", "C1"):
                got = client.predict(text, level, api_name="/learner_vocabulary")
                if got != display(record, level):
                    raise AssertionError(
                        "HTTP endpoint did not return exact graded word probabilities"
                    )
                count += 1
        # A real browser's change action has an internal event rather than a public endpoint.
        # Exercise that actual registered callback with its stored response and confirm pool reuse.
        changes = [
            f
            for f in app.fns.values()
            if f.fn is not None
            and getattr(f.fn, "__name__", None) == "change"
            and getattr(f.fn, "__module__", None)
            == "research.token_readout.learner_view"
        ]
        if len(changes) != 1:
            raise ValueError("Unknown learner cohort-change callback")
        change = changes[0].fn
        switches = 0
        for text, record in by_text.items():
            for label in LEVELS:
                if change(label, text, record) != display(record, label):
                    raise ValueError(
                        "A cached CEFR change altered original source probabilities"
                    )
                switches += 1
        if (
            runner.calls != count
            or change("B2", "edited text", next(iter(by_text.values()))) != ""
        ):
            raise AssertionError(
                "A cached cohort display used a new forward or showed a stale sentence"
            )
    finally:
        app.close()
    dest.mkdir(exist_ok=True, parents=True)
    example = dest / "example.html"
    example.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{TITLE}</title><style>"
        "body{max-width:54rem;margin:6vh auto;padding:0 1rem;font:16px/1.55 system-ui;color:#24372e}"
        f"</style><style>{LEARNER_STYLE}</style><main><h1>{TITLE}</h1>"
        f"{display(by_text[EXAMPLE], POOLED)}</main></html>"
    )
    data = {
        "gradio": gradio.__version__,
        "local_http_status": 200,
        "native_receipt_sha256": digest(source / "verification.json"),
        "l4_encoder_forwards": note["encoder_forwards"],
        "worked_example_sha256": digest(example),
        "http_requests_replayed": count,
        "exact_semantic_renders": count,
        "cached_cefr_callback_views": switches,
        "extra_model_calls_on_level_change": 0,
        "original_article_endpoint_registered": True,
        "original_article_parameter_names": [
            q["parameter_name"] for q in api["/review_english"]["parameters"]
        ],
        "learner_endpoint": "/learner_vocabulary",
        "learner_parameter_names": [
            q["parameter_name"] for q in api["/learner_vocabulary"]["parameters"]
        ],
        "local_model_loads": 0,
    }
    (dest / "http-verification.json").write_text(json.dumps(data, indent=2) + "\n")
    print(data)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    a = p.parse_args()
    run(a.source, a.dest)
