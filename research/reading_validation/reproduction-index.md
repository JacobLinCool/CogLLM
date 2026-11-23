# Reproducing the human-reading measurements

Start with the human response you want to reproduce. The experiments use different people, different texts and different units of observation. A reader's next-word choice, a keypress at a sentence and an expert's judgment about a complete summary each have their own scoring rule. The manuscript's comparisons and the forecasts used to draw its main figure are available separately. [Manuscript](manuscript.md) · [Figure data](../results/reading-validation-manuscript/figure-data.json) · [Figure program](reading_response_figure.py)

## Check the released numbers

Run these commands from the repository root. The first check uses Python's standard library. It verifies that the first-report scores refer to the published forecast and its protocol, checks the word-level marks against their released location table, and reads the actual sample counts. No model or human-response download is needed for this check.

```sh
python3 - <<'PY'
from hashlib import sha256
import json
from pathlib import Path

root = Path("research/results")
def read(path):
    return json.loads(path.read_text())
def fingerprint(path):
    return sha256(path.read_bytes()).hexdigest()

r = root / "reading-validation-reporting-propensity"
s = read(r / "scores.json")
lock = read(r / "fit-lock.json")
assert s["lock"] == fingerprint(r / "fit-lock.json")
assert lock["forecast"] == fingerprint(r / "hazards.npz")
assert lock["protocol"] == fingerprint(Path("research/reading_validation/reporting-propensity-protocol.md"))
print("First reports:", s["participants"], "readers,", s["reader_trials"], "trials,", s["stories"], "families")

w = root / "reading-validation-word-marks"
m = read(w / "scores.json")
assert m["word_outcomes_sha256"] == fingerprint(w / "word-updates.csv")
assert m["design_sha256"] == fingerprint(w / "forecast-lock.json")
print("Expectation marks:", m["adjacent_words"], "adjacent comparisons in", m["paragraphs"], "paragraphs")
PY
```

Expected counts are 233 readers, 1,398 trials and 12 story families for the first-report test, and 1,566 adjacent comparisons in 46 paragraphs for the word marks. To regenerate Figure 1 from the retained numerical predictions, use a Python environment with NumPy and Matplotlib. The program reads the retained word forecasts and the first-cue and narrative-agreement scores and writes PNG, SVG, PDF and figure source data. [Figure PDF](../results/reading-validation-manuscript/observed-reading-changes.pdf)

```sh
python3 -m research.reading_validation.reading_response_figure
```

## Recover the observations and rerun a score

Use a separate working copy for reanalysis. Several fit and scoring commands deliberately create files exclusively, and others rewrite a figure or score. For a grade-only replay, stage the original human data, registered protocol, sealed predictions and input receipts, and leave the destination score absent in the reproduction copy. Keep the published copy intact for comparison. The CPU first-report scorer imports NumPy, SciPy, PyTorch, NLTK, wordfreq and Hugging Face tokenizers. The replay environment used NumPy 2.5.5, SciPy 1.18.1, PyTorch 2.15.0, NLTK 3.10.4, wordfreq 3.1.1 and tokenizers 0.23.3. Source fitting also needs the fingerprinted English WordNet archive under `.cache/cogllm/reading-validation/nltk/corpora/wordnet.zip`. Native L4 model loading has its own package versions in the demo guide. The paths below use the original workspace layout. The hidden `.cache/cogllm` tree contains downloaded sources or derived representations and is not a portable public data bundle. The study protocols identify the original sources, required versions, and exclusion rules. Read each source's use conditions when obtaining it.

| Human observation and unit | Inputs and executable path | Result and boundary of inference |
|---|---|---|
| **Next-word expectations.** People supplied a word after a visible prefix. | UCL source logs and item map, Provo predictability norms, independent dictionary, and frozen native probability export. [Input builder](cloze_population_data.py) · [Fit](cloze_population_fit.py) · [Independent scoring](cloze_population_score.py) | [Population forecasts](cloze-population-results.md) · [Disagreement magnitude](word-dispersion-results.md) · [Large complete-spelling changes](word-mark-certification-results.md). Adjacent prefixes allow a change in the population distribution to be located at a printed word. |
| **Reading history and eye movements.** Later responses from an identified reader, or a regression from a fixated word. | Released response chronology and earlier choices for the reader analysis. The regression analysis uses Provo's eye-movement table alongside independently verified word-cost and cloze measurements. [Reader history](reader_history_mode.py) · [Rereading estimator](expectation_rereading.py) | [Reader history](reader-history-results.md) · [Mixture of reader response modes](reader-history-mode-mixture.md) · [Eye movements](expectation-rereading-results.md). Repeated responses by the same people and alternate scores of the same corpus share evidence. |
| **Earlier reference.** A participant chose an action and, for a backward link, clicked an earlier expression. | Original Phrase Detectives decisions, word ranges and work identities, plus one-pass word features. [Response restoration](nominal_data.py) · [Source fit](nominal_fit.py) · [Held works](nominal_score.py) | [Noun-reference responses](nominal-reference-results.md) · [Pronoun range retrieval](transition-pointer-results.md). An action prediction can be scored on all decisions. A location conditional on an actual backward click has a smaller denominator. |
| **Contradiction onset.** An altered earlier premise preceded an unchanged target sentence. | English Frick study materials, published workbooks for experiments 2 and 3, family definitions, original response paths and native word features. [Input preparation](memory_data.py) · [First-cue model](first_cue.py) · [Reader-reporting model](reporting_propensity.py) | [First-report position and magnitude](first-cue-results.md) · [Persistent reporting propensity](reporting-propensity-results.md). Measurement resolution is one sentence. The latest model comparison reused the previously examined third cohort. |
| **Questions that persist or resolve.** Literary-analysis teams recorded possible questions and opening/closing lines. | Complete German originals with their teams' separate workbooks, exact printed lines and matched story identities. [Independent-team scoring](narrative_agreement.py) · [Closure model](narrative_discharge.py) | [Agreement across teams](narrative-agreement-results.md) · [Simultaneous closures](narrative-discharge-results.md). Retain line or interval uncertainty and the small number of independent stories. |
| **Access to an earlier goal or character.** Humans completed a controlled reading task. | Complete source narratives, matched experimental conditions, registered target exposures and probe measurements. [Goal-change protocol](transition-goal-access-protocol.md) · [Character-access protocol](character-access-protocol.md) | [Goal change](transition-goal-access-results.md) · [Character accessibility](character-access-results.md). Directional agreement with a behavioral intervention and a calibrated effect size are reported separately. |
| **Passage coherence and understandability.** A professional judged a summary, or an annotator identified a specific break. | SEAHORSE's official English release and train/validation/test article groups for whole-summary responses. DECOR and related source annotations for localized judgments. [SEAHORSE preparation](transition_seahorse_data.py) · [Whole-summary scorer](transition_seahorse_score.py) · [Discourse reproduction guide](discourse-order-reproduction.md) | [Professional comprehension judgments](transition-seahorse-results.md) · [Repeated alternatives](transition-seahorse-repeated-results.md) · [English gating](transition-seahorse-gated-results.md) · [Official Spanish transfer](transition-seahorse-spanish-results.md) · [Expert judgments on news](summeval-results.md) · [Local boundaries](transition-external-results.md). An adverse whole-summary rating supplies no position at which understanding failed. |
| **Difficult wording, learner vocabulary, and scientific prose.** Readers named wording, marked a word, rated an article or answered a comprehension item. | Original biomedical sentence ratings and difficult expressions, English learner selections, and public source/rewrite judgments. Preserve readers, articles, rewrites and original item identities. [Material and reader independence](local-study-independence.md) | [Difficult scientific wording](local-results.md) · [Learner-marked words](nns-results.md) · [Definition requests](local-help-results.md) · [ReLay summaries](relay-summary-results.md) · [Rewrite comprehension](revision-comprehension-results.md). Each report gives the text visible to readers and how prior or self-reported knowledge entered the analysis. |
| **Changing the order of the same material.** Readers or editors judged original and reordered sequences. | Original source families, alternative orders, restored training and held interventions, and separate evidence on actual choices. [Discourse guide](discourse-order-reproduction.md) · [News-order guide](news-order-reproduction.md) | [Italian and English discourse](discourse-order-results.md) · [News alternatives](columbia-order-results.md). A historical author order is a distinct outcome from a person's expressed preference. |
| **Transfer across backbones or studies.** Compare predictions of the same observation under controlled changes. | The three pinned backbone revisions, same response and material splits, and separately fitted readouts for each. The 24-source borrowing experiment also requires distinct source likelihoods, structured response geometry and its pretrained 48-coordinate projection. [Backbone revisions](../backbone_transfer/models.json) · [Multi-source commands and file inventory](adaptive-family-reproduction.md) | [Three-backbone comparison](transition-direct-results.md) · [Mandarin localized judgments](transition-mandarin-results.md) · [Source-direction analysis](source-directions-results.md) · [Adaptive sharing](adaptive-family-results.md). The scope of each model comparison is defined by its eligible tasks and held material. |

### Example 1. Grade next-word forecasts from native features

Recover the public UCL cloze source and Provo norms specified in the input builder and study protocol. The original layout includes `.cache/cogllm/reading-validation/provo-predictability.csv` and the prepared `cloze-population` directory under the same root. Restore the native archive containing `lexicon.npz`, `log-probabilities.npz`, `prefix-states.npz` and `receipt.json`. Both source fits and their sealed forecasts are needed before the corresponding grade. The word-mark replay additionally needs the checked native-serving word export and the external scored-word stream. [Original design and data](cloze-population-protocol.md) · [Native-serving protocol](population-serving-protocol.md) · [Conservative word-mark rule](word-mark-certification-protocol.md)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
native=.cache/cogllm/reading-validation/l4/cogllm-reading-cloze-population-20261122T213600Z-830419/extraction/cloze-population-forecast
python3 -m research.reading_validation.cloze_population_score "$native"
python3 -m research.reading_validation.word_population_dispersion_score "$native"
python3 -m research.reading_validation.word_mark_certification
```

For exact spelling, the UCL fit distinguishes named words from unlisted spellings using the original individual answers. It consumes the same native archive above and the frozen source/transfer word-probability tables. The external scorer requires the Peelle anonymous response files, the released replacement and censor lists, and the registered later reader–stem assignments as well as the original Provo counts. The split-reader precision check reads the same verified assignments and frozen estimates. Its released per-stem table contains no worker identifiers. [Exact-spelling estimate](spelling-agreement-results.md) · [Precision across independent readers](spelling-agreement-human-precision.md) · [One-pass inference](spelling_inference.py)

```sh
python3 -m research.reading_validation.spelling_agreement "$native"
python3 -m research.reading_validation.spelling_agreement_score
python3 -m research.reading_validation.spelling_human_precision register
python3 -m research.reading_validation.spelling_human_precision score
python3 -m research.reading_validation.spelling_precision_lists
```

The signed native replay is a separate operation requiring an authorized Colab GPU session and the inputs and parameter files listed in the packet builder. The preparation command fingerprints and packages those inputs and prints the new job path. The worker checks that the number of backbone forwards equals the number of original articles and stores a SHA-256 fingerprint for the word-level output. The completed reference run used 54 articles and 54 forwards. [Signed packet builder](exact_replay_colab.py) · [Native worker](exact_word_l4.py) · [Source fit and forecast fingerprints](../results/reading-validation-spelling-agreement/fit-lock.json)

```sh
cogllm_replay_job=$(python3 -m research.reading_validation.exact_replay_colab prepare)
python3 -m research.reading_validation.exact_replay_colab run "$cogllm_replay_job"
python3 -m research.reading_validation.exact_word_verify "$cogllm_replay_job/extraction/exact-word-forecast"
```

The CPU verifier maps each reported word end to its original norming ID and checks the five available estimates against frozen predictions before writing its receipt. The archived run matched all 2,500 norming prefixes, including the 1,791 positions eligible for the human-outcome analysis. The maximum absolute error across all five outputs was below 0.00010 in probability units. [Word-alignment and numerical verification](exact_word_verify.py) · [Signed native comparison](../results/reading-validation-spelling-agreement/native-serving-verification.json)

### Example 2. Regrade the independent first-report cohort

The score uses the original experiment-3 workbook, prepared map and sealed sentence hazards. Source refitting additionally requires the earlier experimental materials, semantic opportunity tables, earlier first-cue forecast and source lexical resources used by the fit. The native directory below is the archive whose receipt matches the published fit lock. Use the first command to refit in a clean output directory, or restore the fitted forecast and use the second command alone. [Original materials and mapping](memory-protocol.md) · [Fitting and grading design](reporting-propensity-protocol.md)

```sh
native=.cache/cogllm/reading-validation/l4/cogllm-reading-scientific-clues-20261122T185857Z-0d9589/extraction/expectation-forecast
python3 -m research.reading_validation.reporting_propensity fit "$native"
python3 -m research.reading_validation.reporting_propensity score
```

### Example 3. Reproduce the joint calibration and use the trained observers

The joint data archive records the text support, likelihood, observed units and missingness for each measured response. The dedicated guide lists the fit, calibration and grading commands for the globally grouped partition and its separate confirmation partition, the input requirements, and the retained endpoint-level files. After restoring those materials, the verification command below replays source-specific forecast types and checks raw Qwen projections against archived native states. [Observation contracts](../../docs/research/union_training/observation-protocol.md) · [Reproduction guide](adaptive-family-reproduction.md) · [Typed inference API](borrowing_inference.py)

```sh
python3 -m research.reading_validation.borrowing_inference_verify
```

For example, an application that has already encoded a sentence can query Blott's sentence-end sense-response head at the annotated final token. The returned labels and probabilities describe the original response alternatives. The caller supplies the token index and the revision of its existing causal text encoding.

```python
from research.reading_validation.borrowing_inference import AdaptiveObservationReader

observer = AdaptiveObservationReader(seed=1729)
response = observer.from_article(
    "blott_2021_sentence_end_sense",
    features,
    [measured_sentence_end],
    revision=runtime.revision,
)
probability_by_response = dict(zip(response.labels, response.probabilities[0], strict=True))
```

For paired presentations, pass the two previously encoded exposure coordinates to `predict_projected`. For known-error Gaussian measurements, supply the study's sampling standard error in the original outcome units. Gaussian outputs contain a mean vector and covariance matrix in those units. Two independent fits reproduced 13,484 archived held rows apiece across 56 evaluable endpoints and ten likelihood types. The maximum difference in negative log likelihood per observed unit was below 4.5 × 10⁻⁸ for both fits. Seventy-one genuine native Qwen state projections reproduced their archived 48 components exactly. This verifies the implementation. Evidence that the models predict a new human response comes from the separate held-out comparisons. [Endpoint schema](../results/reading-validation-family-bootstrap/global/serving-schema.json) · [Numerical parity by endpoint](../results/reading-validation-family-bootstrap/global/serving-parity.json)

## Reproducing complete-spelling and study-calibration results

The complete-spelling observer predicts how often two people will write different words, including spellings outside the fixed vocabulary. The later reference-panel experiment additionally estimates one shift on 32 people and 64 sentences, holding out every other person and stem. The original no-adjustment predictions remain available for both Provo and Peelle. From a checkout with the two verified native archives and original public human files already staged, the following command fits and grades both experiments in an empty temporary directory and compares all eight resulting files by SHA-256. It does not load a language model locally. [Full-spelling estimates](spelling-agreement-results.md) · [Human sampling precision](spelling-agreement-human-precision.md) · [Disjoint reference-panel results](reading-population-calibration-results.md)

```sh
python -m research.reading_validation.replay_word_results /path/to/cloze-population-forecast /path/to/reader-history-forecast
python -m research.reading_validation.word_spelling_figure
```

On a deployed Qwen runtime, obtain the exact-spelling readout directly from the printed article. The returned rows contain the original character span and token index. The full-spelling probability, the grouped word-choice probabilities and the probability of a reader's first inconsistency report have distinct human observation units. The production writer's conservative underlines use the separately evaluated spelling bounds. [Research inference API](spelling_inference.py) · [Native numerical replay](../results/reading-validation-spelling-agreement/native-serving-verification.json) · [Full-spelling figure data](../results/reading-validation-manuscript/spelling-figure-words.csv)

```python
from research.reading_validation.spelling_inference import ReadingWordForecast

observer = ReadingWordForecast.load()
features = runtime.encode(article)
word_ends = observer.analyze(article, features, runtime)
```

## Section headings and scientific explanation

The deployed 2,048-dimensional sentence readout was evaluated on 24 raw abstracts with 362 matched body-sentence stops. One optional article-level coefficient was fitted within the separate Kerwer cohort. The whole-article scores, the comparison with a heading-only rule, and the nonrotating six-article test have distinct files. [Frozen model and checked native inputs](native-heading-results.md) · [Calibrated intervention and disjoint test](heading-adaptation-results.md) · [Numerical outcomes](../results/reading-validation-local/native-headings/independent-rectangle.json)

## Native inference, provenance and interpretation

The main Qwen experiments used `Qwen/Qwen3.5-2B` at revision `15852e8c16360a2fea060d615a32b45270f8a8fc`. The user guide provides the pinned Python packages, snapshot command, checkpoint path and Gradio invocation for an NVIDIA L4. A complete exposure supplies causal token states in a single forward. A study comparing two printed versions uses one pass for each version. CPU scoring of stored features and paths does not repeat this model call. Numerical parity of a serving adapter establishes implementation agreement with its measured research observer. [Demo setup](../../docs/try-cogllm.md) · [Backbone specifications](../backbone_transfer/models.json) · [Serving verification](../results/reading-validation-cloze-population/native-serving-verification.json)

Separate **what was withheld** from **what had already been examined**. Study-linked reports distinguish new readers, new material, frozen external transfers and analyses added after earlier results became known. Connected reader, article, story or source families determine the interval; a large count of words or repeated responses does not increase the number of independent articles. The corrected multi-source analysis documents one Frick family per newer evaluation that had appeared in an upstream supervised projection. Raw foundation-model pretraining exposure to public text remains unknown. The independent evidence supports specific reading responses at measured resolutions. Individual effort, unreported confusion and a common calibrated token-by-token load scale still require independent measurements. [Source exposure audit](exposure-results.md) · [Corrected family analysis](adaptive-family-results.md) · [Manuscript interpretation](manuscript.md)
