"""Estimate how consistently two public annotation teams locate question changes."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from research.reading_validation.memory_data import sha
from research.reading_validation.narrative_course_data import OUT as NARRATIVE, book, original_path

PLAN = Path(__file__).with_name("narrative-agreement-protocol.md")
OUT = Path("research/results/reading-validation-narrative-agreement")
CURVES = Path("research/results/reading-validation-stock-process/human-and-estimated-questions.csv")


def changes(bounds):
    return np.column_stack((bounds[1:, 0] - bounds[:-1, 1], bounds[1:, 1] - bounds[:-1, 0]))


def direction(interval):
    return (interval[:, 0] > 0).astype(int) - (interval[:, 1] < 0)


def read_stories():
    source = json.loads((NARRATIVE / "input-seal.json").read_text())
    with CURVES.open() as f:
        release = list(csv.DictReader(f))
    result = {}
    for title, article in source["articles"].items():
        if set(article["questions"]) != {"AA", "JE"}:
            continue
        rows = [r for r in release if r["story"] == title]
        if [int(r["line"]) for r in rows] != list(range(1, article["lines"] + 1)):
            raise ValueError("Forecast lines differ from the registered article")
        teams = {}
        first = None
        for team in ("AA", "JE"):
            name, path = original_path(title, team)
            if sha(path) != source["files"][name]:
                raise ValueError("Original annotation workbook changed")
            lines, marks = book(path)
            if first is not None and first != lines:
                raise ValueError("Two annotation teams saw different lines")
            first = lines
            bounds = np.column_stack(((marks == 1).sum(1), (marks != 0).sum(1)))
            published = np.array([[float(r[team+"_confirmed"]), float(r[team+"_possible"])] for r in rows])
            if not np.array_equal(published, bounds):
                raise ValueError("Original labels disagree with the curve coordinates")
            teams[team] = bounds
        result[title] = (teams, {k: np.array([float(r[k]) for r in rows]) for k in ("count", "duration")})
    if len(result) != 3:
        raise ValueError("Expected the three originals with independent annotations")
    return result


def evaluate():
    stories = read_stories()
    results = []
    for title, (teams, forecasts) in stories.items():
        for target, peer in (("AA", "JE"), ("JE", "AA")):
            b = teams[target]
            y, other = changes(b), changes(teams[peer])
            direction_y, direction_peer = direction(y), direction(other)
            peer_level = teams[peer].mean(1)
            train_x = []; train_y = []; weights = []
            for train_title, (paired, _) in stories.items():
                if train_title == title:
                    continue
                n = len(paired[target])
                train_x.extend(paired[peer].mean(1))
                train_y.extend(paired[target].mean(1))
                weights.extend([1 / np.sqrt(n)] * n)
            w = np.array(weights)
            design = np.stack([np.ones_like(train_x), train_x], axis=1)
            fit = lsq_linear(w[:, None]*design, w*np.array(train_y), bounds=([-np.inf, 0], [np.inf, np.inf]))
            if not fit.success:
                raise ArithmeticError("Independent peer scaling failed")
            level = dict(**forecasts, peer=peer_level, crossfit_peer=np.maximum(0, fit.x[0]+fit.x[1]*peer_level))
            errors = {k: float(np.mean((np.maximum(b[:,0]-a,0)+np.maximum(a-b[:,1],0))**2)) for k,a in level.items()}
            determinate = direction_y != 0
            covered = determinate & (direction_peer != 0)
            d = {k: np.diff(v) for k,v in forecasts.items()}
            predictions = {"peer": direction_peer, **{k: np.sign(v) for k,v in d.items()}}
            by_direction = {}
            for label, mask in (("all", determinate), ("declines", direction_y<0), ("increases", direction_y>0)):
                n = int(mask.sum())
                by_direction[label] = dict(n=n, peer_determinate=int((mask & covered).sum()),
                    correct={k: int(((v==direction_y) & mask).sum()) for k,v in predictions.items()})
            decline = int(np.argmin(y[:,1]))
            if y[decline,1] >= 0:
                raise ValueError("Both teams were expected to record at least one decline")
            results.append(dict(story=title, target=target, peer=peer, lines=len(b),
                direction=by_direction, level_squared_error=errors, peer_affine=fit.x.tolist(),
                largest_decline=dict(line=decline+2, target=y[decline].tolist(), peer=other[decline].tolist(),
                  tied=int((y[:,1]==y[decline,1]).sum()), **{k:float(v[decline]) for k,v in d.items()})))
    OUT.mkdir(exist_ok=True)
    scores = dict(protocol_sha256=sha(PLAN), curve_sha256=sha(CURVES), shared_stories=len(stories), results=results)
    (OUT/"scores.json").write_text(json.dumps(scores,indent=2,allow_nan=False)+"\n")
    print(json.dumps([{k:r[k] for k in ("story","target","direction","largest_decline")} for r in results]))


if __name__ == "__main__":
    evaluate()
