"""Acquire only the official ARIES artifacts used in the scientific-review transfer test."""

from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen
import xml.etree.ElementTree as ET

ROOT=Path('.cache/cogllm/reading-validation/aries')
BASE='https://ai2-s2-research-public.s3.us-west-2.amazonaws.com/aries/'


def fetch():
    inventory=ET.parse(ROOT/'inventory.xml')
    sizes={x.findtext('{*}Key').split('/')[-1]:int(x.findtext('{*}Size')) for x in inventory.findall('.//{*}Contents')}
    for name in ('LICENSE','edit_labels_test.jsonl','paper_edits.jsonl','review_comments.jsonl','s2orc.tar.gz'):
        path=ROOT/name
        if not path.exists():
            part=path.with_suffix(path.suffix+'.part')
            with urlopen(BASE+name,timeout=60) as source,part.open('wb') as output:
                while chunk:=source.read(1024*1024):output.write(chunk)
            if part.stat().st_size!=sizes[name]:raise ValueError('Incomplete artifact')
            part.replace(path)
        if path.stat().st_size!=sizes[name]:raise ValueError('Unexpected artifact size')
        print(name,sizes[name],sha256(path.read_bytes()).hexdigest(),flush=True)


if __name__=='__main__':fetch()
