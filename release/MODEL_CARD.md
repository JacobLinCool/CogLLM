---
language:
  - en
base_model: Qwen/Qwen3.5-2B
license: mit
tags:
  - reading-comprehension
  - readability
  - psycholinguistics
  - gradio
---

# CogLLM for English writing review

CogLLM highlights wording readers may find difficult and sentences that may be hard to connect to the preceding text. Paste an English passage into the writing app to see the estimates on your own words. All positions share one causal pass through the frozen Qwen3.5-2B text model. The readouts return numbers and character offsets without generating a response. [Open the writing app](https://huggingface.co/spaces/JacobLinCool/CogLLM)

## Reading the marks

| Output | What the reader did in the underlying study |
| --- | --- |
| Sentence ease, 1–5 | Readers rated the ease of reading biomedical sentences with the article available. A higher score means greater estimated ease. The readout uses the completed sentence. |
| Wording to review | Readers named difficult phrases in biomedical abstracts. The underline proposes the strongest candidate within a sentence. |
| Connection to the previous text | Readers evaluated a story sentence by sentence. An alert marks a completed sentence where the model estimates a higher chance of a coherence complaint. |
| Experimental token signals | The jointly trained checkpoint produces opening, unresolved, resolution and transition scores from each visible prefix. These values are in model units. The timing and identity of an individual reader’s confusion remain unvalidated. |

## What matched human judgments

In a retrospective evaluation on 94 phrases named by 42 readers, the strongest suggestion in the supplied sentence overlapped 45 of the 94 phrases. A model using word frequency, length and position recovered 25. Requiring exact boundaries reduced the released method’s result to 30. Readers could choose what to mention, so this test gives no false-alarm rate for unmentioned wording. On 265 aligned revisions the sentence readout estimated a mean ease gain of 0.497 points on the five-point scale. Readers reported 0.537 points. [PLABA-EVAL and its original observation instructions](https://github.com/SDS-NLP/PLABA-EVAL)

On 49 held-out short stories, the connection alert detected 77 of 123 reader-marked breaks and flagged 48 of 647 unmarked reading decisions. On 100 independent student essays, the unchanged rule found 114 of 446 expert-marked breaks and flagged 131 of 906 coherent sentences. An unflagged passage can still contain a difficult transition. On eight constructed examples with two concurrent questions, the experimental token trace correctly recovered both links in zero examples. [CoheSentia](https://aclanthology.org/2023.emnlp-main.324/) · [DECOR](https://aclanthology.org/2024.emnlp-main.639/)

## Download and run

The model download contains the trained joint token checkpoint, the wording and sentence-rating weights, the CompLex frequency resource, the sentence-connection coefficients and their integrity manifest. The original Qwen weights and tokenizer are downloaded separately from revision `15852e8c16360a2fea060d615a32b45270f8a8fc`. On first use the runtime obtains the two original MedReadMe files directly from their authors, verifies their SHA-256 fingerprints, reconstructs the validated lexical parameters and caches the result. [Inference source and installation](https://huggingface.co/spaces/JacobLinCool/CogLLM/tree/main) · [Qwen backbone](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc)

```python
from hashlib import sha256
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

repo = "JacobLinCool/CogLLM-Qwen3.5-2B"
revision = HfApi().model_info(repo).sha
folder = Path(snapshot_download(repo, revision=revision))
manifest = json.loads((folder / "manifest.json").read_text())
for name, expected in manifest["files"].items():
    assert sha256((folder / name).read_bytes()).hexdigest() == expected, name
print("Verified CogLLM revision", revision)
```

With the Space source checked out and its dependencies installed, load the same revision and inspect the native per-token and sentence outputs directly.

```python
from app import load_writer

writer = load_writer(repo, revision)
result = writer.review_english("The signal weakened. The team adjusted the sensor.")
print(result["english"]["sentences"])
```

## Training and attribution

The experimental token readout was fitted with 65 observation objectives from 26 sources. The selected checkpoint used fixed source weights and a constraint retaining earlier token behavior. The wording head combined biomedical expression statistics, a CompLex lexical readout and Qwen features. Sentence ratings used an ordinal model. Coherence estimation used 5,190 reader decisions on 346 short stories, with 87 separate stories used for calibration. The dataset-specific scales retain their original meanings.

The project supplies its own weights and software under the accompanying MIT license. The derived file `complex-frequencies.json.gz` contains case-folded aggregate word frequencies computed from CompLex fitting sentences by Matthew Shardlow and colleagues. CompLex is available under CC BY 4.0. PLABA-EVAL by Kexin Bian, Su-Youn Yoon and Mamoru Komachi uses the same source license. The MedReadMe inputs come directly from Chao Jiang and colleagues. Qwen publishes its original backbone under Apache 2.0. Training participants’ records and the source passages are excluded from this model download. [CompLex source and license](https://github.com/MMU-TDMLab/CompLex) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [PLABA-EVAL](https://github.com/SDS-NLP/PLABA-EVAL) · [MedReadMe](https://github.com/chaojiang06/medreadme) · [Qwen terms](https://huggingface.co/Qwen/Qwen3.5-2B)
