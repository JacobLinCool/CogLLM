# Reproducing human-order comparisons for news prose

## Public material

The Columbia study recruited ten participants to compose readable news by arranging released units. The first page for each event gives the randomized units. A separate page gives the observed arrangements. The source includes several multi-sentence units and has different content from the English essays and Italian talks used for initial fitting. The SummEval release records expert and crowd judgments on completed news summaries. It contains source IDs, original decoded text and eleven references per news story. The model reads each decoded draft as supplied. Local grade files can be retrieved from the original authors. These observations concern composing or judging news, rather than measuring online burden at individual words. [Columbia instructions and release](https://www.cs.columbia.edu/~noemie/ordering/coll/) · [SummEval paper](https://aclanthology.org/2021.tacl-1.24/) · [Official data inventory](https://github.com/Yale-LILY/SummEval/blob/81b59ad53d63cb6009764240853c91235a44e238/README.md#human-annotations)

The source-aware G-Eval comparator uses the released answers of GPT-4, aligned by news story, generating system and normalized summary wording. The result file includes other metadata. Only the `all_responses` field provides GPT-4 predictions in the present comparison. Individual human ratings are taken only from the original human release. [G-Eval forecasts](https://github.com/nlpyang/geval/blob/6f84404ce4503d2ba0cde0293c5d402ecf2976a8/results/gpt4_coh_detailed.json) · [Original evaluation parser](https://github.com/nlpyang/geval/blob/6f84404ce4503d2ba0cde0293c5d402ecf2976a8/meta_eval_summeval.py)

From the repository root, prepare the local copies of the releases. Split the original story families with the checked-in input manifest.

```bash
mkdir -p .cache/cogllm/reading-validation/summeval
curl -fL 'https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl' -o .cache/cogllm/reading-validation/summeval/original-model-annotations.jsonl
curl -fL 'https://raw.githubusercontent.com/nlpyang/geval/6f84404ce4503d2ba0cde0293c5d402ecf2976a8/results/gpt4_coh_detailed.json' -o .cache/cogllm/reading-validation/summeval/g-eval-coherence.json
python -m research.reading_validation.summeval_data
python -m research.reading_validation.summeval_controls
python -m research.reading_validation.columbia_order_data --fetch
python -m research.reading_validation.news_order_exposure
```

The exact SHA-256 digests of both remote news files are pinned by the two preparation modules. No public source story, output text, private reader profile or generated answer is added to repository artifacts. The hashed passage geometry identifies each item for independent reproduction. When reconstructing the one-pass features, use the original frozen Colab L4 recipe for the bilingual bank and the jointly verified `summeval_l4` observer for news. Each alternative continuation in Columbia has a separate text input. SummEval encodes each distinct draft once and reads both observers from the same token states. The methods share a model revision and unit representation. [Source fitting and environment](discourse-order-reproduction.md) · [Columbia design](columbia-order-protocol.md) · [SummEval design](summeval-protocol.md)

## Numerical reproduction

Given the two native archives, sequentially seal the independent forecasts, fit the calibration-only observers, and then score confirmation.

```bash
python -m research.reading_validation.columbia_order_forecast "$columbia_bank"
python -m research.reading_validation.columbia_order_labels
python -m research.reading_validation.columbia_order_score
python -m research.reading_validation.columbia_choice_training
python -m research.reading_validation.summeval_forecast "$summaries_bank"
python -m research.reading_validation.summeval_ballots --fit "$summaries_bank"
python -m research.reading_validation.summeval_score
python -m research.reading_validation.summeval_ballots --score
python -m research.reading_validation.news_preference_crossover
python -m research.reading_validation.news_strict_exposure_score
python -m research.reading_validation.news_order_figure
```

The source calibration uses the independent readers’ actual ballot distributions for within-story fitting. The ten future stories retain unopened outcomes. The published observer scores and the numeric verdicts are separate artifacts. Ordered hypotheses, optimizer convergence and cohort resampling are specified before consulting confirmation. [Conditional likelihood](summeval-ballot-objective.md) · [Fixed-story and fixed-producer controls](summeval-crossover.md) · [Historical exposure sensitivity](summeval-exposure-sensitivity.md) · [Cross-fitted actual order choices](columbia-choice-training.md)
