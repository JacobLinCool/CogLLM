"""Reconcile cleaned individual completions with the authors' released rounded response tables."""
from collections import Counter, defaultdict
from pathlib import Path
import csv

import numpy as np

from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.reader_history_external import records, answer_normalizer, get_geometry
from research.reading_validation.reader_history_materials import DIR, OUT
from research.reading_validation.cloze_population_data import normalized


def run():
    parents=read_json(OUT/'external-lock.json')
    if digest(DIR/'output.tsv') != parents['inputs']['output.tsv']:
        raise ValueError('Original response table changed')
    _, number=get_geometry()
    normalize=answer_normalizer()
    observed=defaultdict(Counter)
    for _,_,row,fields in records():
        for a,b in fields:
            case=row[a];word=normalize(row[b],number[case])[1] or 'no response'
            observed[case][word]+=1
    input_audit=read_json(OUT/'verified-input-seal.json')
    fixed=dict(zip(input_audit['summary_table_field'],input_audit['observed_raw_field'],strict=True))
    unknown=0;count=0;outside_rounding=0;stems_agree=0;verified_top=0;observed_choices=0
    unreported_answers=0;by_case=[];uniques=0
    with (DIR/'output.tsv').open() as fp:
        for row in csv.DictReader(fp,delimiter='\t'):
            case=fixed.get(row['Question'],row['Question'])
            counts=observed[case];n=sum(counts.values())
            if not n:raise ValueError('Missing exposed stem')
            c=Counter();duplicates=Counter()
            for j in range(1,int(row['Number_of_Unique_Responses'])+1):
                text=normalized(row['Answer_'+str(j)])
                c[text]+=float(row['Percent_of_Responses_'+str(j)])
                duplicates[text]+=1
            diffs=[]
            for w in set(counts) | set(c):
                if w not in c:unknown+=counts[w]
                point=abs(c[w]-counts[w]/n)
                eps=0.0050001*max(1, duplicates[w])
                diffs.append(point)
                if point>eps:outside_rounding+=1
            unreported_answers+=sum(1 for w in counts if w not in c)
            agree=all(abs(c[w]-counts[w]/n)<=0.0050001*max(1,duplicates[w]) for w in set(counts)|set(c))
            uniques += len(c);observed_choices+=n;count+=1;stems_agree+=agree;by_case.append(max(diffs))
            top={w for w,v in counts.items() if v==max(counts.values())}
            verified_top+=int(any(c[w]>=max(c.values())-0.0050001 for w in top))
    if count != len(observed):raise ValueError('Question inventory changed')
    stats=dict(source='Author anonymized raw ballots vs published per-spelling proportions rounded to 0.01.',
        raw_observations_including_missing=observed_choices,stems=count,all_spelling_proportions_within_release_rounding=stems_agree,
        original_most_frequent_response_matches_reconstructed_top=verified_top,published_stem_spellings=uniques,
        absolute_difference_max_per_stem_percentiles=np.quantile(by_case,[.5,.9,.95,.99,1]).tolist(),
        reconstructed_choices_with_spelling_absent_from_summary=unknown,unexpected_unique_stem_spellings=unreported_answers,
        stem_spelling_cells_outside_published_rounding=outside_rounding,population_primary=digest(OUT/'external-scores.json'),
        chance_of_disagreement='The released percentages are rounded. Individual forecasts use the raw released, cleaned choices.')
    write_json(OUT/'external-response-count-audit.json',stats)
    print(stats)
if __name__=='__main__':run()
