# Reproducing the coherence and sentence-order comparisons

The English estimates predict readers’ marks at published sentence boundaries. The Italian estimates predict the mean coherence grade after a four-sentence passage. The Danish estimates rank two naturally written passages within their original genre. The sentence-order experiment compares identical Italian wording in two positions. The annotation is at the paragraph level in Italy and Denmark. Any line plotted over individual Italian or Danish words needs a separate human measurement to validate where comprehension changed.

## Public observations and study identities

| Source | Released observation | Evaluation unit | Prior project exposure |
|---|---|---|---|
| CoheSentia incremental test | Actual sentence and reason indices, indexed separately per reader and story | Original English story, with its readers kept together | Previously graded for a different readout |
| DECOR | Experts’ published coherence decision for a target sentence in an English learner essay | Original essay | Previously graded for a different readout |
| DisCoTex Task 2 | Mean of Italian readers’ 1–5 ratings per passage | Original test passage; repeated source talks are clustered where text identifies them | New target cohort for the bilingual readout |
| DDisCo | A single 1–3 coherence rating per published Danish text | Original text, ranked against texts of the same domain | Independent transfer, with all Danish test grades excluded from fitting |
| Reconstructed DisCoTex | Four original sentence coordinates verified in a separate source task; variants retain every word and punctuation mark | Matched original/permuted text pair, clustered by original source document | Reuses Italian passages. Only one version in the paired comparison has a true human rating |

DDisCo describes two linguistic annotators on its test set. Its public TSV contains one rating per text. DisCoTex reports 10,567 Italian individual ratings in the study, while the public Task 2 table supplies their aggregate for each prompt. Neither table supports reconstructing individual reader variance, detection latency or the onset of a cognitive state. English reader labels distinguish the task design of CoheSentia from the editorial decisions in DECOR. [CoheSentia source](https://github.com/AviyaMn/CoheSentia/tree/891891af4d38518df2f305fa20a12bc9e63fd2e3/data) · [DECOR source](https://github.com/BillyZhang24kobe/writing2coherence/tree/382b4fa086351202f5ac514fcd2b86a5479882e1/data/test) · [DisCoTex human study](https://aclanthology.org/2023.clicit-1.40/) · [DDisCo human study](https://aclanthology.org/2022.lrec-1.260/)

## Public score replay

The numerical replay needs only Python and NumPy, the released prediction package, and three original published files. Use `--fetch` to retrieve the exact public Git revisions or omit it when verified local copies already exist. The package contains `public-observation-index.json`, `forecast/predictions.npz`, `forecast/seal.json` and `scores.json`. The script derives the English boundary votes from the two authors’ original files and reads the Italian mean rating directly from the published TSV. It recomputes 24 estimates, their 48 confidence limits and the maximum discrepancy from the stored table.

```bash
python research/reading_validation/discourse_public_replay.py \
  --package research/results/reading-validation-coherence-order \
  --public-data /tmp/cogllm-public-coherence --fetch
```

## Full fitting and registered comparisons

The private experiment cache is a local work directory for downloaded public raw files and outcome-blind L4 captures. Directory names below refer to already verified native capture directories. Each must contain `receipt.json` and `features.npz`. Use `state_transport_l4.run` with the unchanged original states input, the Danish test input, the 33 held Italian order alternatives, and the 113 Italian source alternatives. The latter three use independent receipts. Each capture contains a single article encoding per observed text. Three separate prefix calls per capture check numerical causality and do not contribute features to fit or test predictions. Exact raw file versions and SHA-256 values appear in the input and forecast receipts. [Native extraction](state_transport_l4.py) · [Main protocol](coherence_order_protocol.md) · [Genre and published sources](coherence_order_genre_protocol.md) · [Danish transfer](danish-coherence-protocol.md) · [Certified sentence order](italian-order-interventions.md) · [Auxiliary loss](italian-contrast-protocol.md)

In the DisCoTex authors’ Python environment, with NLTK 3.8.1 and its original English Punkt models, run the official comparator and its independent set-based reproduction. The English-default segmentation/tokenization of Italian is part of the cited reference. The original one-hot program yields no adjacent pair on `task_2_test_108`.
  
```bash
python -m research.reading_validation.coherence_order_reference
python -m research.reading_validation.coherence_order_reference_check
```

In the CogLLM analysis environment, after placing the three public source collections and native original bank in the documented cache folders, form the split before reading test outcomes. Build the genre control using the official Italian Task 1 TED and Wikipedia train and blind-test source files, alongside the verified Italian training and test text.

```bash
python -m research.reading_validation.coherence_order_data
python -m research.reading_validation.coherence_order_data --audit-history
python -m research.reading_validation.coherence_order_features
python -m research.reading_validation.coherence_order_fit --form-only
python -m research.reading_validation.coherence_order_genre
python -m research.reading_validation.coherence_order_genre_fit
python -m research.reading_validation.coherence_order_verify
python -m research.reading_validation.italian_order_data
python -m research.reading_validation.italian_contrast_data
python -m research.reading_validation.danish_coherence_data
python -m research.reading_validation.danish_coherence_controls
```

Use the original DisCoTex NLTK environment again for `python -m research.reading_validation.italian_order_reference`. The state bank contains hashes and source coordinates, with each text unchanged. No knowledge, reader profile, genre code, question or proposed intervention is supplied to Qwen. The Danish lexical/form references use ratings only from source groups independent of the tested materials. The augmented readout also receives each physically identified training passage in a second order as a separate example, preserving one forward per version.

```bash
python -m research.reading_validation.coherence_order_features --archive "$original_bank"
python -m research.reading_validation.coherence_order_fit
python -m research.reading_validation.danish_coherence_forecast "$danish_bank"
python -m research.reading_validation.italian_order_forecast "$original_bank" "$italian_held_order_bank"
python -m research.reading_validation.italian_contrast_fit "$italian_fit_order_bank" "$original_bank" "$danish_bank" "$italian_held_order_bank"
python -m research.reading_validation.coherence_order_score
python -m research.reading_validation.coherence_order_genre_score
python -m research.reading_validation.danish_coherence_score
python -m research.reading_validation.italian_order_score --authentic --structural
python -m research.reading_validation.italian_contrast_score
python -m research.reading_validation.discourse_family
```

Use document or recovered source clusters for cross-model comparisons, maintain the actual observation unit, and distinguish the original closed estimates from the additional structural objective. The material audit detects identical and near-identical wording. Topics translated between Italian and Danish could still overlap without sharing twelve words. The historical release and current fit share backbone pretraining whose source membership is unknown. [Prespecified ten-test family](discourse-family-protocol.md) · [Verified Italian baseline](https://github.com/davidecolla/DisCoTex/blob/e113774f1b5f9c1eb4ea5581767429241ebb8cc9/src/run/run_baseline_t2.py) · [Danish release](https://github.com/alexandrainst/danlp/blob/master/docs/docs/datasets.md#ddisco)

A second standalone NumPy replay checks the Danish ordering directly from the authors’ test TSV inside the original archive. It recalculates every informative pair within each domain and gives both texts in a pair their bootstrap multiplicities. This independent quadratic calculation checks 45 estimates and their intervals against the production rank-based implementation. The required package contains `forecasts.npz`, `prediction-seal.json` and `scores.json`. [Original DDisCo archive](https://danlp-downloads.alexandra.dk/datasets/ddisco.zip)

```bash
python research/reading_validation/discourse_danish_replay.py \
  --package research/results/reading-validation-danish-coherence \
  --original-zip /path/to/ddisco.zip
```
