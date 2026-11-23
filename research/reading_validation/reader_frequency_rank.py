"""Test whether personal lexical frequency preferences can reorder actual named responses."""
from collections import defaultdict
from argparse import ArgumentParser
from pathlib import Path
from hashlib import sha256

import numpy as np
from scipy.sparse import coo_matrix
from research.backbone_transfer.extract import read_json, write_json, digest
from research.reading_validation import muse_personal
from research.reading_validation.muse_materials import WORK as DE
from research.reading_validation.reader_history_external import base, DIR, OUT as EN_OUT
from research.reading_validation.reader_history_peer import (
    histories,
    population_weights,
    BETA,
)
from research.reading_validation.reader_history_mode import (
    online_priors,
    observed_paths,
)
from research.reading_validation.reader_history_model import match
from research.reading_validation.reader_history_sequence import timeline
from research.reading_validation.cloze_population_data import ROOT as POP

OUT = Path("research/results/reading-validation-reader-rank")
PLAN = Path(__file__).with_name("reader-history-rank.md")
GAMMA = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])


class LanguageModes:
    """Named-word distributions with separate contextual sharpness and frequency tilt."""

    def __init__(self, native, dictionary, language):
        self.p = native[:, :-1]
        self.V = self.p.shape[1]
        if len(dictionary) != self.V:
            raise ValueError("Different word inventory")
        frequency_field = {"en": "frequency", "de": "freq"}[language]
        freq = np.log([v[frequency_field] for v in dictionary])
        self.factor = np.exp(((freq - freq.mean()) / freq.std())[:, None] * GAMMA)
        self.logz = np.empty((len(self.p), len(BETA), len(GAMMA)))
        for k, b in enumerate(BETA):
            for start in range(0, len(self.p), 64):
                p = self.p[start : start + 64].astype(float) * b
                shift = p.max(axis=1, keepdims=True)
                self.logz[start : start + len(p), k] = (
                    np.log(np.exp(p - shift) @ self.factor) + shift
                )

    def likelihood(self, item, word, prior):
        # The caller masks the residual category out of every update. Point that
        # category at an in-bounds placeholder when evaluating a padded batch.
        word = np.minimum(word, self.V - 1)
        # Exposure-shape (person, position) is allowed; retain original shape.
        shape = item.shape
        ii = item.ravel()
        ww = word.ravel()
        result = np.empty((len(ii), len(GAMMA)), np.float64)
        for j in range(0, len(ii), 20000):
            i = ii[j : j + 20000]
            w = ww[j : j + 20000]
            a = np.exp(self.p[i, w, None, None] * BETA[None, :, None] - self.logz[i])
            result[j : j + len(i)] = np.einsum("nkg,k->ng", a, prior) * self.factor[w]
        return result.reshape((*shape, len(GAMMA)))

    def ranked(self, item, prior):
        """Return every possible winner for any convex combination of the five modes."""
        p = np.exp(self.p[item, :, None].astype(float) * BETA)
        coeff = np.exp(-self.logz[item]) * prior[:, None]
        q = (p @ coeff) * self.factor
        if abs(q.sum(0) - 1.0).max() > 5e-7:
            raise ValueError("Unnormalized external word distribution")
        # At least one word attains this bound under every preference profile.
        # A word that falls short even in its strongest mode cannot be the winner.
        lower = q.min(1).max()
        selected = np.flatnonzero(q.max(1) >= lower - 1e-14)
        return selected, q[selected].T


def datasets(en_path, de_path):
    ep, idx, _ = base(en_path)
    h, other = histories(idx)
    with np.load(DIR / "peer-future-geometry.npz") as f:
        sf = f["stem"]
        rf = dict(zip(f["subjects"], f["reader"], strict=True))
    eg = []
    for m in read_json(EN_OUT / "mode-priors.json"):
        a, b = m["reader_fold"], m["stem_fold"]
        tr = (
            np.array([rf[r] != a for r in h[:, 0]])
            & (sf[h[:, 1]] != b)
            & (h[:, 2] != 8192)
        )
        eg.append((a, b, np.array(m["name"]["prior"]), h[tr]))
    en = dict(
        law=LanguageModes(ep, read_json(POP / "dictionary.json"), "en"),
        index=idx,
        fold=sf,
        peoplefold=rf,
        items=eg,
        other=other,
    )
    p, di = muse_personal.native(de_path)
    gm = read_json(muse_personal.OUT / "auditory-fit.json")
    dp = muse_personal.categorical(p, np.array(gm["base"]))
    train = muse_personal.event_rows("training", di)
    de = dict(
        law=LanguageModes(dp, read_json(DE / "dictionary.json"), "de"),
        index=di,
        items=[
            (0, 0, np.array(gm["modes"]["name"]["prior"]), train[train[:, 2] != 4096])
        ],
    )
    return {"en": en, "de": de}


def fit(tables):
    if digest(PLAN) != read_json(OUT / "registration.json")["protocol"]:
        raise ValueError("Protocol changed")
    model = {}
    for language, d in tables.items():
        param = []
        for a, b, sharp, training in d["items"]:
            r, i, w = training.T
            q = d["law"].likelihood(i, w, sharp)
            lp = np.log(q)
            prior, info = population_weights(lp, np.arange(len(q)))
            lam, pairs = muse_personal.correlation_fit(lp, r, prior)
            param.append(
                dict(
                    reader_fold=a,
                    stem_fold=b,
                    prior=prior.tolist(),
                    intraclass=lam,
                    calibration_pairs=pairs,
                    kkt=info["stationarity"],
                    n=len(q),
                    sharp=sharp.tolist(),
                )
            )
        model[language] = param
    write_json(
        OUT / "frequency-source-model.json", dict(protocol=digest(PLAN), source=model)
    )
    print(
        {
            l: np.quantile([p["intraclass"] for p in v], [0, 0.5, 1]).tolist()
            for l, v in model.items()
        },
        flush=True,
    )


def bootstrap(e, loss, guesses):
    r, i, w = e.T
    persons, ri = np.unique(r, return_inverse=True)
    items, ix = np.unique(i, return_inverse=True)
    P, T = len(persons), len(items)
    B = 1800
    rng = np.random.default_rng(76391)
    R = rng.multinomial(P, np.full(P, 1 / P), B)
    I = rng.multinomial(T, np.full(T, 1 / T), B)

    def agg(x):
        A = coo_matrix((x, (ri, ix)), shape=(P, T)).tocsr()
        return np.concatenate(
            [
                np.sum(R[b : b + 60] * (A @ I[b : b + 60].T).T, axis=1)
                for b in range(0, B, 60)
            ]
        )

    units = agg(np.ones(len(r)))

    def estimate(x):
        v = agg(x) / units
        return dict(
            estimate=float(np.mean(x)), interval=np.quantile(v, [0.025, 0.975]).tolist()
        )

    out = {}
    for j, k in enumerate(["population", "matched_other", "cohort"], 1):
        no = guesses[:, 0] != guesses[:, j]
        win = (guesses[:, 0] == w).astype(float)
        control = (guesses[:, j] == w).astype(float)
        out[k] = {
            "proper_gain": estimate(loss[:, j] - loss[:, 0]),
            "top_accuracy_gain": estimate(win - control),
            "changed_tops": int(no.sum()),
            "fraction_changed": float(no.mean()),
            "personal_top_correct_when_changed": int(np.sum(win[no])),
            "control_top_correct_when_changed": int(np.sum(control[no])),
        }
    out["sharp_population"] = {"proper_gain": estimate(loss[:, 4] - loss[:, 0])}
    return dict(
        people=P,
        stems=T,
        named_choices=len(w),
        mean_named_nll=loss.mean(0).tolist(),
        named_top_accuracy=np.mean(guesses == w[:, None], axis=0).tolist(),
        own_versus=out,
    )


def evaluate_fold(
    law, obs, counts, hist_items, hist_words, length, people, other, params
):
    r, i, w = obs.T
    J = {p: j for j, p in enumerate(people)}
    fitted = np.array(params["prior"])
    sharp = np.array(params["sharp"])
    live = np.arange(hist_items.shape[1])[None, :] < length[:, None]
    available = law.likelihood(hist_items, hist_words, sharp)
    post = online_priors(
        available, live, hist_words < law.V, fitted, params["intraclass"]
    )
    del available
    loss = np.empty((len(r), 5))
    guess = np.empty((len(r), 4), int)
    states = np.empty((len(r), 4, len(GAMMA)))
    grouped = defaultdict(list)
    for j, (p, n) in enumerate(zip(r, counts, strict=True)):
        grouped[p, n].append(j)
    cache = {}
    for (p, n), ids in grouped.items():
        u = J[p]
        d = J[other[p]]
        if n not in cache:
            select = length >= n
            cache[n] = post[select, n].astype(float).sum(0), int(select.sum() - 1)
        total, k = cache[n]
        own = post[u, n].astype(float)
        states[ids] = [own, (total - own) / k, post[d, n], fitted]
    prob = law.likelihood(i, w, sharp)
    loss[:, :4] = -np.log(np.einsum("nik,nk->ni", states, prob))
    loss[:, 4] = -np.log(prob[:, 2])
    for item in np.unique(i):
        ids = np.flatnonzero(i == item)
        eligible, q = law.ranked(int(item), sharp)
        guess[ids] = eligible[np.argmax(states[ids] @ q, axis=2)]
    if not np.isfinite(loss).all():
        raise ValueError("Invalid named probability loss")
    return loss, guess


def english(d, params):
    law = d["law"]
    idx = d["index"]
    sf = d["fold"]
    rf = d["peoplefold"]
    _, table, other = timeline(idx)
    people = sorted(table)
    source = DIR / "sequential-forecast-scores.npz"
    if digest(source) != read_json(EN_OUT / "sequential-scores.json")["scored_arrays"]:
        raise ValueError("English trial list changed")
    with np.load(source) as f:
        ok = f["events"][:, 2] < law.V
        e = f["events"][ok]
        n = f["support"][ok]
    r, i, _ = e.T
    rs = np.array([rf[p] for p in r])
    L = np.full((len(e), 5), np.nan)
    G = np.full((len(e), 4), -1)
    for param in params:
        a, b = param["reader_fold"], param["stem_fold"]
        sel = (rs == a) & (sf[i] == b)
        t, len_, x, y = observed_paths(table, people, sf, b)
        L[sel], G[sel] = evaluate_fold(
            law, e[sel], n[sel], x, y, len_, people, other, param
        )
    if np.any(G < 0):
        raise ValueError("Incomplete English exposures")
    return bootstrap(e, L, G)


def german(d, param):
    law = d["law"]
    idx = d["index"]
    sealed = read_json(muse_personal.OUT / "human-forecast-lock.json")
    h = muse_personal.event_rows("history", idx)
    e = muse_personal.event_rows("target", idx)
    e = e[e[:, 2] < law.V]
    orig = sealed["studies"][0]
    people = orig["people"]
    sizes = dict(zip(people, orig["history_sizes"], strict=True))
    e = e[np.isin(e[:, 0], people)]
    n = np.array([sizes[r] for r in e[:, 0]])
    ids = read_json(DE / "cohorts.json")
    shared = set(ids["stems"]["a"]) & set(ids["stems"]["v"])
    geo = read_json(DE / "geometry.json")
    allowed = {
        r: {
            idx[k]
            for s in geo[str(r)]
            if (k := sha256(s.encode()).hexdigest()) in shared
        }
        for r in people
    }
    other, _ = match(people, allowed)
    keys = {
        i: sha256(("German shared calibration " + k).encode()).digest()
        for k, i in idx.items()
    }
    ordered = [
        sorted(h[h[:, 0] == r].tolist(), key=lambda z: keys[z[1]]) for r in people
    ]
    lengths = np.array(list(map(len, ordered)))
    x = np.zeros((len(people), max(lengths)), int)
    y = np.zeros_like(x)
    for j, b in enumerate(ordered):
        x[j, : len(b)] = [a[1] for a in b]
        y[j, : len(b)] = [a[2] for a in b]
    L, G = evaluate_fold(law, e, n, x, y, lengths, people, other, param)
    return bootstrap(e, L, G)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("action", choices=["fit", "score"])
    p.add_argument("en", type=Path)
    p.add_argument("de", type=Path)
    args = p.parse_args()
    tables = datasets(args.en, args.de)
    if args.action == "fit":
        fit(tables)
    else:
        file = OUT / "frequency-source-model.json"
        models = read_json(file)
        if models["protocol"] != digest(PLAN):
            raise ValueError("Different registered model")
        for k, study in [
            ("de", lambda: german(tables["de"], models["source"]["de"][0])),
            ("en", lambda: english(tables["en"], models["source"]["en"])),
        ]:
            result = study()
            write_json(
                OUT / (k + "-named.json"),
                dict(heldout=result, protocol=digest(PLAN), source=digest(file)),
            )
            print(k, result["own_versus"]["population"], flush=True)
