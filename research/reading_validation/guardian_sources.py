"""Identify the reading experiment's stimulus windows from the publisher and two released anchors."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sqlite3
from urllib.error import HTTPError,URLError
from urllib.parse import urlparse
from urllib.request import Request,urlopen

from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.identity_prepare import DATA,sealed_rows

ROOT=DATA/'guardian'
OUT=Path('research/results/reading-validation-guardian')
WORD=re.compile(r"[\w]+(?:[’'][\w]+)*")


class PublisherArticle(HTMLParser):
    def __init__(self):
        super().__init__();self.scripts=[];self.capture=False;self.stack=[];self.content={'body':[],'standfirst':[]}
    def handle_starttag(self,tag,attrs):
        a=dict(attrs);name=None
        if a.get('id')=='maincontent':name='body'
        if 'standfirst' in (a.get('id'),a.get('data-gu-name'),a.get('data-component')):name='standfirst'
        if tag not in ('area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'):
            self.stack.append((tag,name))
        if tag=='script':self.capture=a.get('type')=='application/ld+json'
        if tag in ('div','p','h2','h3','blockquote','br','li'):self.add_text(' ')
    def handle_endtag(self,tag):
        if tag=='script':self.capture=False
        if tag in ('div','p','h2','h3','blockquote','li'):self.add_text(' ')
        for i in range(len(self.stack)-1,-1,-1):
            if self.stack[i][0]==tag:self.stack=self.stack[:i];break
    def add_text(self,s):
        if any(t in ('script','style','aside','svg','button','form','nav','figure','iframe') for t,_ in self.stack):return
        for _,n in reversed(self.stack):
            if n is not None:self.content[n].append(s);break
    def handle_data(self,data):
        if self.capture:self.scripts.append(data)
        self.add_text(data)


def article(url):
    if urlparse(url).hostname not in {'www.theguardian.com','www.guardian.co.uk'}:raise ValueError('Unexpected publisher domain')
    with urlopen(Request(url,headers={'User-Agent':'CogLLM public scientific stimulus verification'}),timeout=30) as response:
        target=response.geturl()
        if urlparse(target).hostname not in {'www.theguardian.com','www.guardian.co.uk'}:raise ValueError('Unexpected origin of redirected article')
        html=response.read(5000000).decode('utf-8')
    p=PublisherArticle();p.feed(html);stack=[json.loads(data) for data in p.scripts];found=[];titles=[]
    while stack:
        x=stack.pop()
        if isinstance(x,list):stack.extend(x);continue
        if not isinstance(x,dict):continue
        if x.get('headline') and 'Article' in str(x.get('@type','')):titles.append(unescape(x['headline']))
        if x.get('articleBody'):found.append(x)
        if '@graph' in x:stack.append(x['@graph'])
    if len(found)==1:
        d=found[0]
        return dict(body=unescape(d['articleBody']),standfirst=unescape(d.get('description','')),headline=unescape(d.get('headline',''))),target
    body=' '.join(''.join(p.content['body']).split());lead=' '.join(''.join(p.content['standfirst']).split())
    if not body or len(found)>1:return None,target
    return dict(body=body,standfirst=lead,headline=titles[0] if len(titles)==1 else ''),target


def tokens(text):return [(m.group().casefold().replace('’',"'"),m.start(),m.end()) for m in WORD.finditer(text)]


def occurrences(phrase,source):
    n=len(phrase);return [i for i in range(len(source)-n+1) if source[i:i+n]==phrase]


def reconstruct(row):
    identity,quotation,url=row;url=url.replace(' (broken)','').replace('http://www.guardian.co.uk','https://www.theguardian.com')
    try:d,target=article(url)
    except (HTTPError,URLError,ValueError,TimeoutError) as err:return {'id':identity,'status':type(err).__name__},None
    if not d:return {'id':identity,'status':'no_full_publisher_article','url':target},None
    parts=[v.strip() for v in quotation.split('...') if v.strip()]
    if len(parts)!=2:return {'id':identity,'status':'ambiguous_source_quotation','url':target},None
    left,right=[[m[0] for m in tokens(v)] for v in parts]
    seqs={'body':d['body'],'standfirst_body':d['standfirst']+' '+d['body'],
          'headline_standfirst_body':d['headline']+' '+d['standfirst']+' '+d['body']}
    candidates={};checks={}
    for name,text in seqs.items():
        w=tokens(text);words=[t[0] for t in w];la=occurrences(left,words);ra=occurrences(right,words);checks[name]=[len(la),len(ra)]
        for l in la:
            for r in ra:
                if r<l+len(left) or r+len(right)-l>1300:continue
                a=w[l][1];b=w[r+len(right)-1][2]
                # The released excerpt itself ends before an ellipsis, often mid-sentence.
                value=text[a:b];sig=' '.join(t[0] for t in tokens(value))
                candidates[sig]=(value,name)
    if len(candidates)!=1:return {'id':identity,'status':'unmatched_or_nonunique','url':target,'anchors':checks},None
    sig,(value,name)=next(iter(candidates.items()))
    return {'id':identity,'status':'both_anchors','url':target,'words':len(sig.split()),'route':name,
            'text_sha256':sha256(value.encode()).hexdigest()},(value,d['headline'])


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    with sqlite3.connect(ROOT/'guardian_stimulus.db') as db:
        rows=db.execute('select stimulus,text,url from guardian_stimuli order by stimulus').fetchall()
    with ThreadPoolExecutor(max_workers=3) as exe:results=list(exe.map(reconstruct,rows))
    report=[];texts=[]
    for audit,text in results:
        report.append(audit)
        if text is not None:
            value,title=text
            if not title:raise ValueError('Cannot identify the contemporary publisher headline')
            for name,article_text in [('body',value+' ...'),('current_title',title+'\n\n'+value+' ...')]:
                uid=sha256(article_text.encode()).hexdigest();audit[name+'_id']=uid
                texts.append({'id':uid,'text':article_text})
    sealed_rows(ROOT/'matched-stimuli.jsonl',texts)
    write_json(OUT/'source-recovery.json',{'source':digest(ROOT/'guardian_stimulus.db'),'outcomes_read':False,'release_url':'https://doi.org/10.6084/m9.figshare.27057958.v1',
        'publisher_texts_republished':False,'matched_stimuli':digest(ROOT/'matched-stimuli.jsonl'),'articles':report,'status':dict(Counter(r['status'] for r in report))})
    print('Recovery:',Counter(r['status'] for r in report),flush=True)


if __name__=='__main__':main()
