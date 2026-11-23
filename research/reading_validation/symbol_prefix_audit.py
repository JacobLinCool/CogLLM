"""Exercise source proposals, membership and pointwise readouts against removal of all later text."""
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from research.backbone_transfer.extract import digest, write_json
from research.reading_validation.symbol_model import Bank,Page
from research.reading_validation.symbol_syntax import proposals
from research.reading_validation.symbol_geometry import OUT


def run():
    bank=Bank('.',synthetic=True);checks=0;locations=0;offsets=0;maxdiff=0.;bad=[];maxargs=''
    for page,d in bank.pages('fit',160):
        for fraction in (.26,.61,.87):
            n=int(page.n*fraction)
            if n<2 or page.end[n-1]>=len(page.text) or page.end[n-1]>=page.end[n]:continue
            end=page.end[n-1];text=page.text[:end];keys=page.keys[page.keys<n]
            row={'id':page.row['id'],'text':text,'keys':keys};view=Page(row,SimpleNamespace(offsets=page.words.offsets[:n]),page.state[:n],page.relations[:n,:len(keys)])
            ref={(m.start,m.end,m.committed):m for m in page.mentions if m.committed<=end};got={(m.start,m.end,m.committed):m for m in view.mentions}
            if ref.keys()!=got.keys():
                bad.append((d['id'],end,list(ref.keys()-got.keys())[:2],list(got.keys()-ref.keys())[:2],text[-30:]));continue
            locations+=len(ref);checks+=1
            for k in list(ref)[-2:]:
                x=np.hstack(page.features(ref[k],[n-1]));y=np.hstack(view.features(got[k],[n-1]));delta=float(np.abs(x-y).max());maxdiff=max(delta,maxdiff);offsets+=1
                if delta>1e-6 and len(bad)<6:bad.append((d['id'],end,'feature',np.argmax(np.abs(x-y)).item(),delta))
    print('prefix trials',checks,'positions',locations,'feature checks',offsets,'max',maxdiff,'failures',len(bad),bad[:8])
    if bad or maxdiff>1e-6:raise ArithmeticError('An article readout consulted unavailable suffix text')
    write_json(OUT/'prefix-geometry-audit.json',{'source':'Synthetic fixed native states on actual fit-paper coordinates; GPU suffix invariance is reported separately.','prefixes':checks,'source_proposals':locations,'features':offsets,'max_error':maxdiff,'features_sha256':digest(Path(__file__).with_name('symbol_features.py')),'syntax_sha256':digest(Path(__file__).with_name('symbol_syntax.py'))})

if __name__=='__main__':run()
