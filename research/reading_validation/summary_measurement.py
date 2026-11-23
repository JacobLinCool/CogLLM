"""Standalone numerical definition of the frozen English sentence and document measurements."""
from collections import Counter
import itertools
import re
import unicodedata

import numpy as np
from scipy.special import logsumexp

from research.reading_validation.argument_data import sentences

WORDS=re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*|\d+(?:[.,]\d+)*")
FORM_WORDS=re.compile(r"[^\W_]+",re.UNICODE)
PROJ=np.r_[0:8,32:40]
CONNECT=frozenset("and but so then however meanwhile therefore nevertheless consequently instead because although yet dunque ma poi però tuttavia quindi inoltre invece perciò e mentre sebbene".split())


def document_form(text,atoms):
    """Fourteen layout and token-string measurements used in the original news control."""
    normalize=lambda s: FORM_WORDS.findall(unicodedata.normalize("NFKC",s).casefold())
    words=normalize(text); parts=[normalize(text[a:b]) for a,b in atoms]
    hits=[];past=set()
    for part in parts:
        if past:
            hits.append(sum(w in past for w in part)/max(1,len(part)))
        past.update(part)
    n=max(1,len(words))
    count=lambda mark:text.count(mark)/n
    def repeated_grams(width):
        grams=Counter(tuple(words[j:j+width]) for j in range(len(words)-width+1))
        total=sum(grams.values())
        return 1-len(grams)/total if total else 0.
    return [np.log1p(len(words)),np.log1p(len(parts)),len(words)/max(1,len(parts)),sum(map(len,words))/n,
            len(set(words))/n,repeated_grams(2),repeated_grams(3),1-len(set(map(tuple,parts)))/max(1,len(parts)),
            np.mean(hits) if hits else 0.,*[count(mark) for mark in (",",":","'","?","!")]]


def ratio(k, n):
    return k / max(1, n)


def jaccard(a, b):
    return ratio(len(a & b), len(a | b))


def cosine(a, b):
    scale = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / scale) if scale > 1e-9 else 0.0


def form(text, words, coords, a, b, atoms):
    current = np.flatnonzero((coords[:, 0] >= a) & (coords[:, 1] <= b))
    if not len(current):
        return None
    before = np.flatnonzero(coords[:, 1] <= a)
    prev = None
    for s, e in reversed(atoms):
        if s >= a:
            continue
        hit = np.flatnonzero((coords[:, 0] >= s) & (coords[:, 1] <= min(a, e)))
        if len(hit):
            prev = hit
            break
    if prev is None:
        prev = np.empty(0, dtype=int)
    current_words = [words[i] for i in current]
    near = [words[i] for i in prev]
    history = [words[i] for i in before]
    c, p, all_prev = map(set, (current_words, near, history))
    n = len(current_words)
    nt = len(before)
    bigram = set(itertools.pairwise(near))
    seg = text[a:b]
    position = sum(1 for s, e in atoms if e <= a)
    first = current_words[0]
    x = np.array(
        [
            np.log1p(nt),
            np.log1p(position),
            np.log1p(n),
            np.log1p(len(prev)),
            np.mean([len(w) for w in current_words]),
            np.mean([len(w) >= 8 for w in current_words]),
            np.mean([text[coords[i, 0]].isupper() for i in current]),
            np.mean([w.isnumeric() for w in current_words]),
            len(c) / n,
            seg.count(",") / n,
            "?" in seg,
            first in CONNECT,
            jaccard(c, p),
            jaccard(c, all_prev),
            np.mean([w in p for w in current_words]),
            np.mean([w in all_prev for w in current_words]),
            ratio(sum(v in bigram for v in itertools.pairwise(current_words)), n - 1),
            ratio(len(c & all_prev), len(c)),
        ],
        float,
    )
    return x, current, prev, before


def neural(u, c, p, before, atoms, a):
    coords, valid, surprisal, word, attention = u
    c, p, before = (q[valid[q]] for q in (c, p, before))
    if not len(c):
        return None
    S = surprisal[c]
    pc = surprisal[p] if len(p) else np.empty(0)
    prevmean = pc.mean() if len(pc) else 0.0
    q = np.array(
        [
            S.mean(),
            S.std(),
            np.quantile(S, 0.9),
            S[0],
            S[:3].mean(),
            S[-1],
            S.mean() - prevmean,
            S[:3].mean() - prevmean,
        ]
    )
    v = word[c].mean(0)
    previous = word[p].mean(0) if len(p) else np.zeros_like(v)
    overall = word[before].mean(0) if len(before) else np.zeros_like(v)
    visible = [j for j, (s, e) in enumerate(atoms) if e <= a]
    relevance = []
    for j in visible:
        b, e = atoms[j]
        ix = before[(coords[before, 0] >= b) & (coords[before, 1] <= e)]
        if len(ix):
            relevance.append(cosine(v, word[ix].mean(0)))
    strong = max(relevance, default=0.0)
    recent = cosine(v, previous)
    links = np.zeros(5)
    if visible:
        observed = attention[c][:, visible, :]
        m = observed[:, :, 0].mean(0)
        den = m.sum()
        links[:3] = [m[-1], observed[0, -1, 0], observed[:, -1, 1].max()]
        if den > 1e-9:
            dist = m / den
            links[3:] = [
                dist @ dist,
                dist @ np.arange(len(m) - 1, -1, -1) / max(1, len(m) - 1),
            ]
    sem = np.r_[
        v[PROJ],
        word[c[-1], PROJ],
        np.abs(v[PROJ] - previous[PROJ]),
        recent,
        cosine(v, overall),
        cosine(word[c[0]], previous),
        cosine(word[c[-1]], previous),
        strong,
        recent - strong,
        links,
        bool(visible),
    ]
    return np.r_[q, sem]


def summarize(text, native, center, scale):
    atoms=sentences(text);raw=native["char"].astype(int)
    matches=list(WORDS.finditer(text));coords=raw[:,:2]
    if not np.array_equal(coords,np.array([w.span() for w in matches],dtype=int).reshape(-1,2)):
        raise ValueError("Printed words and original native word coordinates differ")
    words=[w.group().casefold() for w in matches]
    observable=raw[:,4].astype(bool)
    if not observable.any():return None,0,"no_observable_word_surprisal"
    word,nll,rel=(native[k] for k in ("words","surprisal","rel"))
    u=(coords,observable,nll.astype(float),word.astype(float),rel.astype(float))
    regions=[]
    for a,b in atoms[1:] if len(atoms)>1 else atoms:
        base=form(text,words,coords,a,b,atoms)
        if base is not None:
            value=neural(u,*base[1:],atoms,a)
            if value is not None:regions.append(np.r_[base[0],value])
    if not regions:return None,0,"no_certified_boundary"
    z=np.clip((np.stack(regions)-center)/scale,-8,8)
    doc=np.r_[document_form(text,atoms),float(nll[observable].mean()),z.mean(0),logsumexp(z,axis=0)-np.log(len(z))]
    if doc.shape!=(187,) or not np.isfinite(doc).all():raise ArithmeticError("Invalid English summary measurements")
    return doc,len(regions),None
