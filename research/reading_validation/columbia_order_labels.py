"""Recover every participant row from the authors' separately released actual-order tables."""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

from research.reading_validation.columbia_order_data import Cells, OUT, TOPICS, URL, WORK, units
from research.reading_validation.local_forecast import digest


def load(topic,inventory):
    dest=WORK/'actual-orders'/(topic+'.html')
    if not dest.exists():
        with urlopen(URL+topic+'-orderings.html',timeout=30) as stream:dest.write_bytes(stream.read())
    text=dest.read_bytes().decode('windows-1252')
    if 'Multiple Orderings' not in text or '['+topic+']' not in text:raise ValueError('Unknown original actual-order table')
    table=Cells();table.feed(text);observed=[];ids=[]
    for row in table.extracted():
        if len(row)!=2:raise ValueError('Unexpected original ordering row')
        m=re.fullmatch(r'Participant\s+(\d+)',row[0],re.I)
        if not m:raise ValueError('Missing original participant identifier')
        labels=row[1].split()
        if len(labels)!=len(inventory) or set(labels)!=set(inventory):raise ValueError(f'Missing or repeated unit for {topic}, participant {m.group(1)}: {labels}')
        ids.append(int(m.group(1)));observed.append(labels)
    if len(set(ids))!=len(ids) or ids!=sorted(ids) or len(ids)<2:raise ValueError('Unclear public participant row identities')
    return topic,observed,ids,digest(dest)


def main():
    seal=json.loads((OUT/'forecast-seal.json').read_bytes())
    if seal['forecasts']!=digest(OUT/'forecasts.npz') or not seal['individual_order_tables_opened']==False:raise ValueError('Seal model forecasts before opening participant choices')
    (WORK/'actual-orders').mkdir(exist_ok=True);inventory=units();rows=[]
    with ThreadPoolExecutor(max_workers=5) as pool:
        rows=list(pool.map(lambda k:load(k,inventory[k]),TOPICS))
    record={'source_sha256':{t:s for t,_,_,s in rows},'participant_numbers_by_event':{t:ids for t,_,ids,_ in rows},'orders':{t:a for t,a,_,_ in rows},'forecasts_frozen_first':digest(OUT/'forecast-seal.json')}
    (OUT/'observed-orders.json').write_text(json.dumps(record,indent=2)+'\n');print({t:len(a) for t,a,_,_ in rows})


if __name__=='__main__':main()
