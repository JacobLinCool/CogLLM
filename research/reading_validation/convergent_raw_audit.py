"""Rebuild item-level behavioral counts without storing any raw questionnaire or reader identifier."""
from collections import Counter, defaultdict
from pathlib import Path
import ast
import base64
import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
import re
import subprocess

import numpy as np
import openpyxl
import xlrd
from research.backbone_transfer.extract import digest, write_json
from research.reading_validation.convergent_counts import study
from research.reading_validation.convergent_rating_counts import counts
from research.reading_validation.convergent_word_prepare import ROOT, OUT

REF = "47b7f923a27111da428d3802d9ab52c6db835ba8"


def download(name):
    proc = subprocess.run(
        [
            "gh",
            "api",
            f"repos/Andrea-de-Varda/prediction-resource/contents/{name}?ref={REF}",
        ],
        capture_output=True,
    )
    if proc.returncode:
        raise OSError("Could not retrieve pinned public source " + name)
    result = json.loads(proc.stdout)
    return base64.b64decode(result["content"], validate=False)


def constants():
    tree = ast.parse((ROOT / "preprocess.py").read_text())
    return {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("trgt_compregension", "punct_words")
    }


def raw():
    code = constants()
    truth = code["trgt_compregension"]
    punct = code["punct_words"]
    material = defaultdict(list)
    for r in csv.DictReader((ROOT / "items.csv").open()):
        material[int(r["list"])].append(r)
    fitted, _ = study()
    rating, _ = counts()
    files = {}
    results = {}
    cohorts = {}
    for group in range(1, 9):
        hashes = {}
        numbers = {}

        def inspect(path):
            value = download(path)
            hashes[path] = sha256(value).hexdigest()
            return value

        correction = inspect(f"logs/logs{group}.csv").decode("utf-8-sig")
        correction = {
            (int(r["idx"]), r["Q_num"]): (r["word"], r["corrected"])
            for r in csv.DictReader(StringIO(correction))
        }
        # Read only experimental responses. Metadata columns and individual IDs never enter the output.
        sheet = xlrd.open_workbook(
            file_contents=inspect(f"cp/list{group}.xls")
        ).sheet_by_index(0)
        cloze_header = sheet.row_values(0, 28)
        cloze_questions = sheet.row_values(1, 28)
        data = []
        for rr in range(2, sheet.nrows):
            if sum(a == b for a, b in zip(sheet.row_values(rr, 18, 28), truth)) >= 8:
                data.append((rr - 1, sheet.row_values(rr, 28)))
        if len(cloze_header) != len(material[group]):
            raise ValueError("Source cloze item count changed")
        numbers["cloze_respondents"] = len(data)
        wb = openpyxl.load_workbook(
            BytesIO(inspect(f"ratings/list{group}.xlsx")),
            read_only=True,
            data_only=True,
        )
        rows = iter(wb.active.iter_rows(min_col=19, values_only=True))
        next(rows)
        next(rows)
        scores = []
        for row in rows:
            if sum(a == b for a, b in zip(row[:10], truth)) >= 8:
                scores.append(row[10:])
        wb.close()
        numbers["rating_respondents"] = len(scores)
        for j, item in enumerate(material[group]):
            iid = int(item["item_id"])
            target = re.sub(r"[.,?!:;]", "", item["word"].strip())
            target = punct.get(target, target)
            question = re.sub(
                r"\[Field-1\]\.\.\.|- Write the next word of the sentence:",
                "",
                cloze_questions[j],
            ).strip()
            if question != item["sentence"].strip():
                raise ValueError(
                    f"Original question order differs, list {group}, item {iid}"
                )
            cr = []
            blank = 0
            changes = 0
            for idx, row in data:
                word = row[j]
                if not isinstance(word, str) or not word.strip():
                    blank += int(not str(word).strip())
                    val = word
                else:
                    val = word.split(" ")[0].lower().replace("’", "'")
                    if (idx, cloze_header[j]) in correction:
                        old, new = correction[idx, cloze_header[j]]
                        if old != val:
                            raise ValueError(
                                "Correction log does not match original behavioral entry"
                            )
                        val = new
                        changes += 1
                cr.append(val)
            vals = [
                int(
                    str(row[j])
                    .replace("2 - Very much", "5" if group == 5 else "2")
                    .replace("- Very much", "")
                    .replace("- Not at all", "")
                    .strip()
                )
                for row in scores
            ]
            h = sum(x == target for x in cr)
            cn, hit = fitted[iid]
            rn, mu, sd = rating[iid]
            results[str(iid)] = dict(
                cloze_slots=cn,
                cloze_nonempty=cn - blank,
                cloze_source_hits=h,
                source_cloze_denominator=len(cr),
                reconstructed_cloze_hits=hit,
                source_ratings=len(vals),
                reconstructed_ratings=rn,
                raw_rating_moment_max_error=float(
                    max(abs(np.mean(vals) - mu), abs(np.std(vals, ddof=1) - sd))
                ),
                spelling_corrections=changes,
            )
        cohorts[str(group)] = numbers
        files.update(hashes)
    actual = [r for r in results.values()]
    result = dict(
        revision=REF,
        source_sha256=files,
        analysis_sha256=digest(Path(__file__)),
        word_units=len(results),
        cohorts=cohorts,
        cloze_matching_counts=sum(
            x["cloze_source_hits"] == x["reconstructed_cloze_hits"]
            and x["source_cloze_denominator"] == x["cloze_slots"]
            for x in actual
        ),
        ratings_matching_n=sum(
            x["source_ratings"] == x["reconstructed_ratings"] for x in actual
        ),
        max_abs_rating_moment_difference=max(
            x["raw_rating_moment_max_error"] for x in actual
        ),
        questionnaire_cloze_slots=sum(x["source_cloze_denominator"] for x in actual),
        empty_cloze_slots=sum(x["cloze_slots"] - x["cloze_nonempty"] for x in actual),
        items=results,
    )
    write_json(OUT / "independent-source-count-audit.json", result)
    print(
        {
            k: v
            for k, v in result.items()
            if k
            in [
                "word_units",
                "cloze_matching_counts",
                "ratings_matching_n",
                "max_abs_rating_moment_difference",
                "questionnaire_cloze_slots",
                "empty_cloze_slots",
            ]
        }
    )


if __name__ == "__main__":
    raw()
