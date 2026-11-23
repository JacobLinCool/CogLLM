"""Check analytic interval gradients, causally masked inputs, and fit transfer on simulated labels."""

from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from research.reading_validation.identity_fit import ARMS, RiskSet, fit_models, predict


def main():
    rng=np.random.default_rng(189)
    data=SimpleNamespace(channels=['baseline_logit','exact/test','middle/test'],values={},anchors={})
    rows=[]
    for i in range(52):
        key=f'synthetic-{i}'; values=rng.normal(0,.6,(2,50,3)); windows=[[12,17],[32,37]]
        if i%2:
            windows.reverse()
        for j,(a,b) in enumerate(windows):
            values[j,a:b,2] += 2.3
        row={'id':key,'group_id':key,'source':'simulation','risk_start':5,'window_tokens':dict(zip('AB',windows))}
        rows.append(row); data.values[key]=values; data.anchors[key]=list('AB')
    risk=RiskSet.make(data,rows[:40],[0,1,2]); theta=rng.normal(size=4)
    numerical=approx_derivative(lambda w:np.array([risk.objective(w)[0]]),theta,method='3-point').reshape(-1)
    err=float(np.max(abs(numerical-risk.objective(theta)[1])))
    if err>1e-7:
        raise AssertionError(f'Incorrect analytic gradient: {err}')
    models=fit_models(data,rows[:40]); checks=[]
    for row in rows[40:]:
        logits=predict(models,data.values[row['id']])[ARMS.index('new_readout')]
        good=all(logits[j,a:b].mean()>logits[1-j,a:b].mean() for j,(a,b) in enumerate(row['window_tokens'].values()))
        checks.append(bool(good))
    if not all(checks):
        raise AssertionError('Known generating signal was not recovered on held-out simulated papers')
    print(json.dumps({'analytic_gradient_max_error':err,'held_out_both':sum(checks),'held_out_papers':len(checks),
                     'max_abs_fit_gradient':{k:v['max_abs_gradient'] for k,v in models.items()}},indent=2))


if __name__=='__main__':
    main()
