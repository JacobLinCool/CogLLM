"""Verify the public Gradio route against original, fingerprinted L4 responses."""

from argparse import ArgumentParser
from contextlib import closing
from pathlib import Path
import socket

from gradio_client import Client

from research.reading_validation.writer_http_verify import RecordedOutput
from research.reading_validation.writer_verify import DraftText
from research.token_readout import experimental_view, writing_review
from research.token_readout.app import parse_analysis


def verify(source, serve=None):
    replay = RecordedOutput([source])
    app = writing_review.build_app(replay, study_tools=False, max_tokens=512)
    if serve is None:
        with closing(socket.socket()) as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
    else:
        port = serve
    app.launch(
        server_name="127.0.0.1", server_port=port, prevent_thread_lock=True,
        css=writing_review.STYLE, quiet=True,
    )
    try:
        client = Client(f"http://127.0.0.1:{port}", verbose=False)
        routes = client.view_api(print_info=False, return_format="dict")["named_endpoints"]
        if set(routes) != {"/review_english"}:
            raise AssertionError("The public writer exposes exactly one analysis endpoint.")
        articles = [r for (text, *extra), r in replay.records.items() if not any(extra)]
        for record in articles:
            text = record["text"]
            html = client.predict(text, api_name="/review_english")
            source_text = DraftText()
            source_text.feed(html)
            source_text.close()
            analysis = parse_analysis(text, record)
            expected = writing_review.render(text, record["english"]) + experimental_view.panel(analysis)
            if html != expected or "".join(source_text.parts) != text or source_text.depth:
                raise AssertionError("The HTTP review changed the source text or a recorded result.")
            if html.count("<tr>") - 1 != len(analysis.tokens):
                raise AssertionError("One table row per actual model token is required.")
        if not articles or replay.calls != len(articles):
            raise AssertionError("Every eligible article must invoke the runner exactly once.")
        print(f"Verified {replay.calls} recorded articles through {next(iter(routes))}.", flush=True)
        if serve is not None:
            input(f"Browser replay at http://127.0.0.1:{port}/ — press Return to stop. ")
    finally:
        app.close()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("research/results/reading-validation-writer"))
    parser.add_argument("--serve", type=int, help="Keep the verified local replay open for a browser check.")
    args = parser.parse_args()
    verify(args.source, args.serve)
