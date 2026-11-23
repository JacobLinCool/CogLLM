"""Reduce live nested-text contrast measurements to a traceable local accessibility receipt."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path("research/results/reading-validation-writer")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify(path):
    output=path.read_text().split("### Result\n",1)[1]
    observed,_=json.JSONDecoder().raw_decode(output)
    names={"wording","summary","expectation","narrative","terminology","reference","physical"}
    expected={(t,w,mode) for t,w in (("light",1280),("dark",390)) for mode in names}
    if len(observed)!=len(expected) or {(r["theme"],r["width"],r["mode"]) for r in observed}!=expected:
        raise ValueError("Missing an original live writer mode or viewport")
    ratios={};samples={}
    for theme,width in (("light",1280),("dark",390)):
        rows=[r for r in observed if r["theme"]==theme]
        values={role:[] for r in rows for role in r["readable"]}
        counts={}
        for row in rows:
            for role,stat in row["readable"].items():
                values[role].append(stat["min"])
                counts[role]=counts.get(role,0)+stat["nodes"]
        required={"underlined words","legend and caption","disclosure","link","source words","explanation","table","darkest scale endpoint"}
        if not required.issubset(values):
            raise ValueError("A required element was absent from the live readings")
        minimum={k:min(v) for k,v in values.items()}
        if min(minimum.values())<4.5:
            raise ValueError("A visible reading-surface contrast fell below 4.5:1")
        ratios[theme]=minimum;samples[theme]=counts
    checks={
        "method":"Foreground/background compositing from each visible DOM text node in the actual Gradio theme, including opened disclosures. Word-mark ink also compared against darkest fixed scale endpoint.",
        "required_ratio":4.5,"minimum_by_theme_and_text_role":ratios,"evaluated_text_node_presentations":samples,
        "original_signed_writer_modes":sorted(names),"captured_runs":observed,"browser_log_sha256":digest(path),
        "css_module_sha256":digest(Path("research/token_readout/writing_review.py")),
    }
    (ROOT/"contrast-browser-verification.json").write_text(json.dumps(checks,indent=2)+"\n")
    print({k:round(min(v.values()),3) for k,v in ratios.items()})


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("log",type=Path);verify(p.parse_args().log)
