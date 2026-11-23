"""Build a condition-blind English sentence packet for the Leroy et al. intervention."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from shutil import copyfile

from research.reading_validation.local_forecast import digest


BASE = Path(".cache/cogllm/reading-validation")
DATA = BASE / "noun"
OUT = Path("research/results/reading-validation-noun")
SHA = "9039dd6da83a84407ca4726998d43ed16d174367bdb060e0e43575bd89f5662f"


def design():
    from openpyxl import load_workbook

    source = BASE / "leroy.xlsx"
    if digest(source) != SHA:
        raise ValueError("Unrecognized original workbook")
    workbook = load_workbook(source, read_only=True, data_only=True)
    grid = iter(workbook["Data"].values)
    header = next(grid)
    rows = [dict(zip(header, row, strict=True)) for row in grid]
    workbook.close()
    seen = Counter()
    groups = defaultdict(list)
    records = []
    for row in rows:
        gram = row["GRAM"]
        split = {"split":True,"unsplit":False}[row["compound"]]
        pseudo = {"YES":True,"NO":False}[row["pseudoword"]]
        cell = gram, split, pseudo
        family = f"{gram}-{seen[cell]:02d}"
        seen[cell] += 1
        text = row["Original Sentence"]
        ident = hashlib.sha256(text.encode()).hexdigest()
        groups[family].append((row["Head Noun"], split, pseudo, row["Noun Phrase"], text))
        records.append(dict(family=family, gram=gram, split=split, pseudo=pseudo, id=ident, text=text,
                            difficulty=row["avgPerceivedDif"], error=1.0-row["avgActualDif"]))
    if len(seen)!=12 or set(seen.values())!={30} or len(groups)!=90:
        raise ValueError("Expected the source's complete 3-by-2-by-2 design")
    for group in groups.values():
        if len({str(r[0]).casefold() for r in group})!=1 or len({(r[1],r[2]) for r in group})!=4:
            raise ValueError("Experimental variants disagree on source family")
        if len({(r[1], str(r[3]).casefold()) for r in group})!=2:
            raise ValueError("Pseudoword intervention unexpectedly changes the designated noun phrase")
    return records


def prepare():
    rows = design()
    stimuli = {r["id"]:r["text"] for r in rows}
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    packet = DATA / "local-stimuli.jsonl"
    with packet.open("w") as f:
        for ident, text in sorted(stimuli.items()):
            f.write(json.dumps(dict(id=ident, text=text),ensure_ascii=False)+"\n")
    copyfile(BASE / "local-lexical.npz", DATA / "lexical.npz")
    seal = dict(workbook_sha256=SHA, protocol=digest(Path(__file__).with_name("noun-protocol.md")),
                original_sentences=len(rows), unique_texts=len(stimuli), families=len({r["family"] for r in rows}),
                packet=digest(packet), lexical=digest(DATA / "lexical.npz"))
    (OUT / "input-seal.json").write_text(json.dumps(seal, indent=2)+"\n")
    print(json.dumps(seal))


if __name__ == "__main__":
    prepare()
