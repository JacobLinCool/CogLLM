"""Download hash-verified public files for the Carter and Hoffman replication."""
import hashlib
import json
from pathlib import Path
import subprocess

CACHE = Path(".cache/cogllm/reading-validation/carter")
SOURCE = Path(__file__).with_name("transition-source.json")


def download():
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, source in json.loads(SOURCE.read_text())["files"].items():
        if not source["used"]:
            continue
        path = CACHE / name
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".part")
            subprocess.run(["curl", "--fail", "--location", "--show-error", "--silent", "--retry", "3",
                            "--max-time", "180", source["url"], "-o", str(tmp)], check=True)
            check(tmp, source)
            tmp.replace(path)
        check(path, source)


def check(path, source):
    if path.stat().st_size != source["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError(f"Public file does not match the OSF manifest: {path}")


if __name__ == "__main__":
    download()
