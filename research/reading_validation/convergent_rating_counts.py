"""Verify the integer first and second rating moments for each of the eight respondent lists."""
from collections import defaultdict
from csv import DictReader
from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.convergent_word_prepare import ROOT,OUT


def counts():
    groups=defaultdict(list)
    for r in DictReader((ROOT/'outcomes.csv').open()):groups[int(r['list'])].append((int(r['item_id']),float(r['rating_mean']),float(r['rating_sd'])))
    coverage={};obs={}
    for k,rows in sorted(groups.items()):
        feasible=[]
        for n in range(35,80):
            for i,m,sd in rows:
                a=n*m;b=(n-1)*sd**2+n*m*m
                if abs(a-round(a))>1e-7 or abs(b-round(b))>1e-6 or not n<=round(a)<=5*n or not n<=round(b)<=25*n or (round(b)-round(a))%2:break
            else:feasible.append(n)
        if len(feasible)!=1:raise ValueError('The released rating moments did not identify a unique list sample size')
        n=feasible[0];coverage[k]={'n':n,'items':len(rows)}
        for item,mean,sd in rows:obs[item]=(n,mean,sd)
    return obs,coverage


if __name__=='__main__':
    o,c=counts();write_json(OUT/'rating-count-identification.json',dict(source=digest(ROOT/'outcomes.csv'),coverage=c,items=len(o),assumptions='One integer-valued 1–5 response per retained participant and item, sample SD uses N-1, common list denominator, possible list size between 35 and 79.'));print(c)
