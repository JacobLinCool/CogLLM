"""Verify reader-visible proposals on randomly and intentionally truncated scientific text."""
from collections import Counter
from pathlib import Path
import numpy as np
from research.backbone_transfer.extract import digest,stream_rows,write_json
from research.reading_validation.symbol_geometry import ROOT,OUT
from research.reading_validation.symbol_span_geometry import math_position,mentions


def run():
    rand=np.random.default_rng(314895);stats=Counter()
    for i,r in enumerate(stream_rows(ROOT/'stimuli-native.jsonl')):
        if i%34:continue
        text=r['text'];all_mentions=mentions(text);math=math_position(text);positions=list(rand.integers(1,len(text)+1,4))
        positions+=([j+1 for j,w in enumerate(text) if w in '\\$}'][:12]);stats['documents']+=1
        for cut in sorted(set(positions)):
            prefix=text[:cut];orig={m for m in all_mentions if m.committed<=cut};new=set(mentions(prefix));stats['prefixes']+=1;stats['proposals']+=len(new)
            if orig!=new:
                sample=list(orig^new)[:2];raise ValueError(f'Uncommitted boundary at {cut}, delta={sample!r}')
            if not np.array_equal(math[:cut+1],math_position(prefix)):raise ValueError('Math mode depends on later text')
    write_json(OUT/'typed'/'prefix-certificate.json',dict(**stats,geometry=digest(Path(__file__).with_name('symbol_span_geometry.py')),original=digest(ROOT/'stimuli-native.jsonl')))
    print(stats)


if __name__=='__main__':run()
