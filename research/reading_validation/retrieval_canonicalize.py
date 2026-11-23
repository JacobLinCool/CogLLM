"""Normalize three literal array headers from the pinned L4 writer without altering data bytes."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

import numpy as np

from research.reading_validation.memory_data import sha


def normalize(original,dest):
    if dest.exists():
        raise FileExistsError(dest)
    receipt=json.loads((original/"receipt.json").read_bytes())
    if any(sha(original/k)!=v for k,v in receipt["files"].items()):
        raise ValueError("Original L4 archive no longer matches its seal")
    layout=json.loads((original/"layout.json").read_bytes())
    n=layout["words"][-1];pairs=layout["relations"][-1]
    if receipt["words"]!=n or receipt["articles"]!=len(layout["ids"]):
        raise ValueError("Native array geometry is inconsistent")
    schema={"signals":("<f4",n,3),"states":("<f2",n,96),"relations":("<f2",pairs,2)}
    dest.mkdir(parents=True)
    for name in ("coordinates.npy","layout.json"):
        shutil.copyfile(original/name,dest/name)
    evidence={}
    for name,(dtype,length,width) in schema.items():
        file=name+".npy"
        with (original/file).open("rb") as src, (dest/file).open("xb") as target:
            if src.read(8)!=b"\x93NUMPY\x01\x00":
                raise ValueError("Only the pinned NPY version-one headers are understood")
            size=struct.unpack("<H",src.read(2))[0]
            header=src.read(size).decode("ascii").strip()
            expected="{'descr': '%s', 'fortran_order': False, 'shape': (np.int64(%d), %d), }"%(dtype,length,width)
            if header!=expected or (original/file).stat().st_size-src.tell()!=length*width*np.dtype(dtype).itemsize:
                raise ValueError(f"Unexpected shape or stored payload size for {file}: {header}")
            np.lib.format.write_array_header_1_0(target,dict(descr=dtype,fortran_order=False,shape=(length,width)))
            before=hashlib.sha256()
            while chunk:=src.read(1<<20):
                before.update(chunk);target.write(chunk)
        reopened=np.load(dest/file,allow_pickle=False,mmap_mode="r")
        after=hashlib.sha256(memoryview(reopened).cast("B")).hexdigest()
        if reopened.shape!=(length,width) or after!=before.hexdigest():
            raise ValueError("Normalization changed a numeric array")
        evidence[file]=after
    receipt.update(original_receipt=sha(original/"receipt.json"),unchanged_numeric_payloads=evidence,
                   files={p.name:sha(p) for p in dest.iterdir()})
    (dest/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("Certified three normalized headers and identical numeric bytes",list(evidence))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("original",type=Path);p.add_argument("dest",type=Path)
    args=p.parse_args();normalize(args.original,args.dest)
