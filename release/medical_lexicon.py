"""Rebuild the published reader's lexical features from the original MedReadMe hosts."""

import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from research.reading_validation.local_features import Lexicon, medical_counts
from research.reading_validation.local_inference import LEXICON_DIGEST, LocalInference
from research.stage6.medreadme_span_audit import FILES, REVISION

FREQUENCY_DIGEST = "81b4f6de9ca44c4c1f80637a0c3e455a66db97b6efb94ef96fe2e9754749978a"
SOURCE = f"https://raw.githubusercontent.com/chaojiang06/medreadme/{REVISION}/dataset"


def reconstruct(directory, raw):
    """Return exactly the features used in the final English estimator."""
    frequencies = gzip.decompress((directory / "complex-frequencies.json.gz").read_bytes())
    if sha256(frequencies).hexdigest() != FREQUENCY_DIGEST:
        raise ValueError("The released CompLex frequencies have changed")
    medical, coverage = medical_counts(raw, exclude=lambda words: False)
    if coverage != {"used_sentences": 2559, "ambiguous_or_unaligned": 28}:
        raise ValueError("The training subset differs from the published MedReadMe feature fit")
    parameters = Lexicon(json.loads(frequencies), *medical).records()
    canonical = json.dumps(parameters, separators=(",", ":"), sort_keys=True).encode()
    if sha256(canonical).hexdigest() != LEXICON_DIGEST:
        raise ValueError("Rebuilt lexical features differ from the validated reading model")
    return canonical


def load_writing_head(directory, *, cache_directory=None):
    """Load the editor's verified wording and sentence weights, caching only fitted features."""
    directory = Path(directory)
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache = Path(cache_directory) if cache_directory is not None else cache_root / "cogllm"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"lexicon-{LEXICON_DIGEST}.json"
    if not target.exists():
        with TemporaryDirectory(dir=cache) as temp:
            work = Path(temp)
            for filename in FILES:
                with urlopen(f"{SOURCE}/{filename}", timeout=60) as source:
                    with (work / filename).open("wb") as output:
                        copyfileobj(source, output)
            built = work / "lexical-parameters.json"
            built.write_bytes(reconstruct(directory, work))
            built.replace(target)
    return LocalInference.from_lexicon(directory, target.read_bytes())
