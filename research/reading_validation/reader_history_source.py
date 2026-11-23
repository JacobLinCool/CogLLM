"""Recover anonymous repeated answers and verify that they reproduce all published aggregate rows."""
from collections import Counter, defaultdict
from hashlib import sha256
from io import StringIO
import csv,re
import numpy as np
import xlrd
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.cloze_population_data import ROOT as POP,normalized
from research.reading_validation.reader_history_materials import OUT,DIR,PLAN
from research.reading_validation.convergent_raw_audit import download,constants
from research.reading_validation.convergent_word_prepare import ROOT as UCL


def run():
    registered=read_json(OUT/'registration.json')
    if registered['protocol']!=digest(PLAN):raise ValueError("Different source protocol")
    audit=read_json(OUT.parent/'reading-validation-convergent/independent-source-count-audit.json')
    gold=read_json(POP/'source-counts.json')
    words={e['word']:i for i,e in enumerate(read_json(POP/'dictionary.json'))}
    groups=defaultdict(list)
    for r in csv.DictReader((UCL/'items.csv').open()):groups[int(r['list'])].append(r['item_id'])
    control=constants()['trgt_compregension']
    rows=[];cohorts={};record_sets=defaultdict(list);skipped=0
    for g,items in sorted(groups.items()):
        def fetch(name):
            value=download(name)
            if sha256(value).hexdigest()!=audit['source_sha256'][name]:raise ValueError("Different original source sheet")
            return value
        book=xlrd.open_workbook(file_contents=fetch(f'cp/list{g}.xls')).sheet_by_index(0)
        repair={(int(r['idx']),r['Q_num']):(r['word'],r['corrected']) for r in csv.DictReader(StringIO(fetch(f'logs/logs{g}.csv').decode('utf-8-sig')))}
        cols=book.row_values(0,28);recounts=defaultdict(Counter)
        assert 'Prolific ID' in str(book.cell_value(1,17))
        for row in range(2,book.nrows):
            if sum(a==b for a,b in zip(book.row_values(row,18,28),control))<8:continue
            source_key=str(book.cell_value(row,17)).strip().lower()
            # No recruitment identifier, free response, contact field or coordinate is written to disk.
            if not re.fullmatch(r'[0-9a-f]{24}',source_key):
                skipped+=1
                subject=('unknown',g,row)
            else:
                subject=('known',sha256(source_key.encode()).hexdigest())
            record_sets[subject].append(g)
            for j,(word,iid) in enumerate(zip(book.row_values(row,28),items,strict=True)):
                if not str(word).strip():continue
                value=word
                if isinstance(word,str):
                    value=word.split(' ')[0].lower().replace('’',"'")
                    if (row-1,cols[j]) in repair:
                        old,new=repair[row-1,cols[j]]
                        if value!=old:raise ValueError("Source correction differs")
                        value=new
                w=normalized(value)
                recounts[iid][w]+=1
                rows.append((subject,g,int(iid),words.get(w,8192)))
        if any(dict(c)!=gold[i] for i,c in recounts.items()):raise ValueError("Individual choices fail to reproduce registered aggregate response counts")
        cohorts[g]=len(recounts)
    if len(rows)!=135905:raise ValueError("Actual source choice counts changed")
    ids={k:i for i,k in enumerate(sorted(record_sets,key=str))}
    arr=np.array([[ids[s],g,i,w] for s,g,i,w in rows],np.int32)
    path=DIR/'source-observations.npz';np.savez_compressed(path,events=arr)
    describe=dict(protocol=digest(PLAN),source_records=sum(len(v) for v in record_sets.values()),verified_nonmissing_worker_keys=len(record_sets)-skipped,
          anonymous_records_without_worker_key=skipped,repeated_people_across_lists=sum(len(set(v))>1 for v in record_sets.values()),repeated_same_list_sessions=sum(len(v)-len(set(v)) for v in record_sets.values()),recruitment_identifiers_saved=False,
          unique_source_reader_units=len(ids),corrected_choice_responses=len(rows),source_aggregate_exact_match=True,canonical_source_file=digest(path),
          future_geometry=['local_integer_reader','questionnaire_list','word_item','fixed_lexicon_category'],published_word_counts=digest(POP/'source-counts.json'))
    write_json(OUT/'source-readers.json',describe);print(describe)


if __name__=='__main__':run()
