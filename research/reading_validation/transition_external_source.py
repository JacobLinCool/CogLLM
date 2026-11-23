"""Retrieve authors' exact public benchmark files for local evaluation."""
import hashlib
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

CACHE = Path(".cache/cogllm/reading-validation/cohesion")
COHE = "AviyaMn/CoheSentia/891891af4d38518df2f305fa20a12bc9e63fd2e3/data/"
DECOR = "BillyZhang24kobe/writing2coherence/382b4fa086351202f5ac514fcd2b86a5479882e1/data/test/"
FILES = {
    "TrainData.json": (COHE, "085829e31bb7cd2bdef7e96d9185099d94bafa82"),
    "TestData.json": (COHE, "d21356a0f248caaeae93f6703a34d8ec23de8dea"),
    "test_1355_clean.csv": (DECOR, "55081da692b33e2a25b8368774203429527dff30"),
    "test_rewrite_213_no_delete.csv": (DECOR, "3ec31cd8716ba62bbdf2e78532339db2f3082509"),
}


def open_verified(name):
    raw = (CACHE / name).read_bytes()
    digest = hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()
    if digest != FILES[name][1]:
        raise ValueError(f"Official public git blob mismatch: {name}")
    return raw


def download(name):
    path = CACHE / name
    if not path.exists():
        tmp = path.with_suffix(".part")
        subprocess.run(["curl","--fail","--silent","--show-error","--location","--retry","3",
                        "--max-time","120","https://raw.githubusercontent.com/"+FILES[name][0]+name,
                        "-o",str(tmp)],check=True)
        tmp.replace(path)
    open_verified(name)


if __name__=="__main__":
    CACHE.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download,FILES))
