"""Causal late interaction between a supplied past anchor and each arriving token."""

import re

import numpy as np
import torch
from torch.nn import functional as F


STOP = frozenset('a an the and or who what when where which how why is was were are of from to in on for as that this their it paper study question following by at does do did has have had us we i they those these based described used using use can be will with'.split())
WORD = re.compile(r'[^\W_]+', re.UNICODE)
CHANNELS = ('exact', 'character', 'embedding', 'middle', 'final', 'mean_attention', 'peak_attention')
POOL = ('current_max', 'mean_best_6', 'mean_best_24', 'mean_recent_6')
NAMES = ['baseline_logit', *(f'{c}/{p}' for c in CHANNELS for p in POOL)]
LEXICAL = [i for i, name in enumerate(NAMES) if name.split('/')[0] in ('exact', 'character')]


def token_forms(text, offsets):
    forms = []
    for start, end in offsets:
        parts = WORD.findall(text[start:end].casefold())
        word = max(parts, key=len) if parts else ''
        forms.append(word if len(word) > 1 and word not in STOP else '')
    return forms


def char_grams(term):
    if not term:
        return frozenset()
    marked = '^' + term + '$'
    return frozenset(marked[i:i + 3] for i in range(len(marked) - 2))


def pools(matrix):
    # Rows are candidate anchor pieces and columns are visible cue positions.
    current = matrix.amax(0)
    features = [current]
    for width in (6, 24):
        values = F.max_pool1d(F.pad(matrix.unsqueeze(0), (width - 1, 0), value=-float('inf')),
                              kernel_size=width, stride=1).squeeze(0)
        features.append(values.mean(0))
    recent = F.avg_pool1d(F.pad(current[None, None], (5, 0)), kernel_size=6, stride=1).flatten()
    features.append(recent)
    return features


@torch.no_grad()
def pairs(runtime, reader, packet, row, encoded):
    text, anchors = row['text'], row['anchor_tokens']
    forms = token_forms(text, encoded.offsets)
    grams = {form:char_grams(form) for form in set(forms)}
    device = encoded.hidden.device
    words = [forms[i] != '' for i in range(len(forms))]
    valid_cue = torch.tensor(words, device=device)[None, :]
    token_ids = torch.tensor(encoded.token_ids, device=device, dtype=torch.long)
    embeddings = runtime.encoder.embed_tokens(token_ids).float()
    states = [F.normalize(v.float(), dim=1) for v in (embeddings, encoded.mid, encoded.hidden)]
    trace = reader(encoded).resolution
    scale, prior = packet['scale'], packet['priors']['resolution']
    output = []
    for label in sorted(anchors):
        start, end = anchors[label]
        if not 0 <= start < end <= len(forms):
            raise ValueError('An anchor is outside token boundaries')
        selected = [i for i in range(start, end) if forms[i]] or list(range(start, end))
        idx = torch.tensor(selected, device=device)
        # Hide the whole query until its final token is available. Earlier candidate
        # features are sentinels and never enter the question's risk set.
        base = trace[:, end - 1] * scale['resolution_gain'] + scale['resolution_bias'] + prior
        base = base.clamp(-24, 12)
        lexical = np.zeros((2,len(selected),len(forms)), dtype='float32')
        for j, i in enumerate(selected):
            term = forms[i]
            if not term:
                continue
            key = grams[term]
            similarities = {other:(len(key & value) / len(key | value) if other else 0.)
                            for other,value in grams.items()}
            lexical[0,j] = [bool(other) and other == term for other in forms]
            lexical[1,j] = [similarities[other] for other in forms]
        exact, character = torch.from_numpy(lexical).to(device)
        matrices = [exact, character]
        for state in states:
            # Mask uninformative cue tokens; clip negative cosine similarities to zero.
            matrices.append((state.index_select(0, idx) @ state.T).clamp_min(0) * valid_cue)
        for chan in range(2):
            matrices.append(encoded.relations[:, :, chan].index_select(1, idx).T)
        columns = [base]
        for matrix in matrices:
            columns.extend(pools(matrix))
        features = torch.stack(columns, dim=1)
        features[:end] = 0
        output.append(features.cpu().numpy())
    matrix = np.stack(output)
    if not np.isfinite(matrix).all() or matrix.shape != (len(anchors), len(forms), len(NAMES)):
        raise ArithmeticError('Nonfinite or unaligned late-interaction features')
    return matrix
