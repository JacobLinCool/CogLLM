"""Check the visible English summary percentages and threshold decisions against actual L4 outputs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


RESULT = Path("research/results/reading-validation-writer")


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def main(log):
    body=log.read_text().split("### Result\n",1)[1]
    rows,_=json.JSONDecoder().raw_decode(body)
    captured=RESULT/"summary-source/responses.json"
    signed=json.loads(captured.read_text())
    if [(r["theme"],r["width"],r["id"]) for r in rows] != [(t,w,i) for t,w in (("light",1280),("dark",390)) for i in ("1","2","3")]:
        raise ValueError("Expected both full interface sizes and all three certified human-study examples")
    records=[]
    for row in rows:
        raw=signed[row["id"]];score=raw["english"]["summary"]
        expected=f"{score['probability_difficult']:.0%}"
        decision="Worth a second read." if score["review"] else "Below the study’s review threshold."
        if raw["text"]!=row["text"] or row["percent"]!=expected or not row["decision"].endswith(decision):
            raise ValueError("Visible word, percentage or review flag was not present in the native reading")
        if not row["fits"] or "short English summaries" not in row["caption"] or "whole draft" not in row["caption"]:
            raise ValueError("The screen did not state the intended use of the model estimate")
        records.append({"theme":row["theme"],"width":row["width"],"input":row["id"],"percent":expected,"review":score["review"],
                        "unchanged_original_text":True,"fits_viewport":True})
    imgs=[Path("output/playwright")/f"summary-{t}.png" for t in ("light","dark")]
    report={"native_responses_sha256":sha(captured),"visible_summaries_verified":records,
            "screenshots_sha256":{str(p):sha(p) for p in imgs},"original_log_sha256":sha(log)}
    (RESULT/"summary-browser-verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"Verified all {len(records)} visible summary predictions.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("log",type=Path);main(p.parse_args().log)
