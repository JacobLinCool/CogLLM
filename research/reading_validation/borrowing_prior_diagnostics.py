"""Describe identification and export per-source held effects without reusing the fit optimizer."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


OUT=Path('research/results/reading-validation-borrowing-prior')
REPLICATE=Path('research/results/reading-validation-borrowing-replication')


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source_stats=read(OUT/'source-scores-and-intervals.json');aggregate=read(OUT/'scores.json');report={};errors=[];quadrature=[];grad=[];fd=[];counts={}
    for mode in ('historical','fresh'):
        for seed in (1729,2718):
            d=OUT/mode/f'seed-{seed}';c=read(d/'calibration-lock.json');known=read(d/'scores.json');err=[];shifts=[];med=[];width=[];cal_count=[];owners=c['source_hyperpriors'];centers=[z['mean'] for z in owners.values()];cross_mass=[];strong=0
            with np.load(d/'calibration-curves.npz',allow_pickle=False) as f:
                n=c['quadrature']['selected'];nodes=f['grid'+str(n)];mass=f['weights'+str(n)];losses=f['loss'+str(n)];r={v['endpoint']:v for v in c['calibration_rows_and_families']}
                for i,k in enumerate(f['endpoints']):
                    if k not in known['blocks']:continue
                    z=owners[r[k]['source']];a,b=z['alpha_shape'],z['beta_shape'];lp=np.log(mass)+(a-1)*np.log(nodes)+(b-1)*np.log1p(-nodes);lp-=logsumexp(lp);post=lp-losses[i];post-=logsumexp(post);pw=np.exp(post);our=pw@nodes;prs=c['coefficients'][k];err.append(abs(our-prs['supported']));med.append(our);shifts.append(our-prs['transported']);width.append(np.interp(.975,np.cumsum(pw),nodes)-np.interp(.025,np.cumsum(pw),nodes));cal_count.append(r[k]['families']);prob=pw[nodes>.5+1e-15].sum()+.5*pw[np.isclose(nodes,.5,atol=1e-15,rtol=0)].sum();strong+=bool(max(prob,1-prob)>.975);cross_mass.append(pw@(post-lp))
                quadrature.append(c['quadrature']['max_abs_coefficient_difference']);grad.extend(q['convergence']['projected_gradient'] for q in owners.values());fd.extend(q['convergence']['finite_difference_error'] for q in owners.values())
            with np.load(d/'family-statistics.npz',allow_pickle=False) as f:
                base=known['methods'];source={}
                for key,v in known['blocks'].items():
                    n=f[key+'.numerator'].sum(0)/f[key+'.denominator'].sum();e=np.array([v['scores'][k] for k in base]);err.extend(abs(n-e));source.setdefault(v['source'],[]).append(n)
                m=np.mean([np.array(arr).mean(0) for arr in source.values()],axis=0);public=np.array([known['macro_source_nll'][x] for x in base]);err.extend(abs(m-public))
            cinfo={'evaluable':len(med),'with_target_calibration':sum(n>0 for n in cal_count),'source_means_range':list(map(float,[np.min(centers),np.max(centers)])),'target_supported_coef_range':list(map(float,[np.min(med),np.max(med)])),'median_abs_target_update':float(np.median(abs(np.asarray(shifts)))),'median_posterior_mass_interval_width':float(np.median(width)),'median_target_information_nats':float(np.median(cross_mass)),'posterior_mass_one_side_of_half_above_0975':strong,'largest_replay_error':float(max(err)),'historical_control_parity':known['published_four_control_scores_replay_max_error']};report[mode+'/'+str(seed)]=cinfo;errors.extend(err)
    if max(errors)>3e-6:raise ArithmeticError('Independent aggregate or posterior replay failed')
    with (OUT/'all-study-results.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['source','material_split','endpoints','held_families','private_nll','shared_nll','midpoint_nll','endpoint_map_nll','hierarchical_supported_nll','endpoint_map_gain_vs_midpoint','gain_95_lower','gain_95_upper','supported_gain_vs_endpoint_map','gain_95_lower','gain_95_upper'])
        for mode in ('historical','fresh'):
            for src,rec in source_stats[mode].items():
                v=rec['nll_by_method'];u=rec['held_material_uncertainty']['intervals'];p=rec['paired_improvement'];w.writerow([src,mode,rec['endpoints'],rec['held_families'],*[v[k] for k in ['private','shared','midpoint','independent_mode','supported']],p['independent_mode_over_midpoint'],*(u['independent_mode_over_midpoint'] if u else ['','']),p['supported_over_independent_mode'],*(u['supported_over_independent_mode'] if u else ['',''])])
    result={'by_split_seed':report,'largest_quadrature_coefficient_error':max(quadrature),'largest_projected_hypergradient':max(grad),'largest_directional_derivative_error':max(fd),'largest_independent_replay_error':float(max(errors)),'source_table':digest(OUT/'all-study-results.csv'),'code':digest(Path(__file__))};(OUT/'parameter-diagnostics.json').write_text(json.dumps(result,indent=2)+'\n');print(result['largest_independent_replay_error'],report['historical/1729'])


if __name__=='__main__':main()
