# CogLLM Protocol for Reproducing and Analyzing Human Reading Behavior

CogLLM asks whether a language model can reproduce **how human reading effort and comprehension change with the position reached in a text**. The model makes one causal forward pass over the article. Readout heads estimate reading responses at each token in parallel. Each position uses the text available up to and including that token, with optional reader background. Training and validation use existing human observations and published experimental effects. The analysis code computes differences between text conditions from model outputs and compares them with the same endpoints in the original studies. Core endpoints include reader reports of ease, comprehension, questions, and changes in interpretation. Where matched data exist, the analysis can also compare answers, recall, eye movements, and reading times. [Empirical reading literature](reading-evidence.md)

Evaluation on research papers and expository writing includes passages that explain throughout. Different versions of the same content measure the effects of explanation order, sentence construction, and coherence cues. Human responses with defined locations validate within- and across-sentence estimates. Corresponding post-passage responses validate global comprehension. The author-facing comparison of rewrites draws on both forms of evidence. [Endpoints and contrasts for expository reading](expository-reading-plan.md)

The starting model is `Qwen/Qwen3.5-2B`. **Completed work includes separate readouts for sentence ratings, reference choice, and open-book answers, plus a first comparison of three-task shared training with and without published effect supervision.** A Zhan single-task readout also scored an external reference dataset. An obsolete index affected the early PLABA localization analysis of open-book responses. With a complete evidence inventory and the newer causal sentence-end readout, the candidate's ranking of incorrect responses fell short of two prespecified baselines. [First shared results](joint-first-run.md) · [Joint effect comparison](joint-effects-run.md) · [Corrected evidence scores](../../research/results/plaba-corrected-evidence-scores.json)

The [research execution plan](execution-plan.md) records agent responsibilities, stage deliverables, acceptance criteria, PI decisions, and resource estimates.

Formal training uses **Colab L4** and official Qwen weights pinned to a Hugging Face revision. Public releases on Hugging Face are preferred for data. The initial PLABA-EVAL and Mandarin continuation data come from verified author releases. The Apple GPU measurements below are existing experimental baselines. [Data sources](reading-evidence.md#hugging-face-source-audit) · [L4 execution and verification](execution-plan.md#colab-l4-的執行與驗收)

<a id="逐-token-的閱讀反應"></a>

## Token-Level Reading Responses

At article token `t`, the simulation uses only the text visible to the reader at that point, plus any reader background and task information then available. Held-out tests check that two continuations with the same prefix produce the same response distribution before they diverge. Qwen can return an estimate at every token. Human experiments typically sample at word or phrase boundaries where material can be shown and responses collected. The protocol aligns observed intervals to the source text and then to tokenizer offsets. Reported precision follows the human sampling points.

For the same potentially unresolved reference, concept, or causal relation, record these locations and their supporting evidence separately.

| Location | Question |
| --- | --- |
| When an unresolved issue appears | Where does reading first leave an interpretation open, such as a reference with two candidates? |
| When a difficulty becomes apparent | Which later cue shows that the current interpretation no longer fits or first produces a detectable difference in answers? This may be the first informative phrase of the next sentence. |
| When the interpretation changes | Where does later text resolve the question, favor a different interpretation, or temporarily make earlier information less accessible? An old detail may affect later responses again when it is mentioned. |

For example, “that card” may raise the probability of a report of unresolved reference. At “the access card,” the model can assign a clarification score at the current location and link it to the earlier “that card.” Analysis code compares responses across reading stops. An explicit report of no question, evidence that an existing question was resolved, and a missing observation are separate outcomes.

A stack-like state is a testable hypothesis. Several questions may remain unresolved at once, and an older question can resolve first. Sentence boundaries, informative cues, and topic changes are distinct candidate triggers. A question may resolve or temporarily cease to affect reading. Experiments have found that time shifts in narratives change access to earlier information, and that older information can reactivate after interference during reading declines. [Ditman et al. 2008, Methods and Fig. 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC2667942/) · [Kendeou et al. 2013, Experiments 1–2 and 6–7](https://experts.umn.edu/en/publications/updating-during-reading-comprehension-why-causality-matters/)

<a id="第一個前綴對照"></a>

### First Shared-Prefix Test

Engineering tests use alternative continuations after a common opening, including immediate clarification, delayed clarification, and a topic change. First verify that changing later text preserves the readouts over the shared token prefix. After divergence, the analysis can compare responses and record onset, trigger, and referent span separately. Highlights use character spans in the source text to render a mask.

Human validation uses existing data. INQUISITIVE stops at sentence boundaries and can test which phrase a reader selected when reporting a question. TED-Q rates the same question every two sentences and stops tracking it once the rating reaches 3. Ratings of 3–4 constrain an interval for partial answering. Positions without later ratings retain an observation mask. The data include fully answered older questions alongside newer questions still below the partial-answer threshold. State transitions therefore allow an older question to resolve first and preserve timing uncertainty between stops. The TED-Q text currently serves as a methodological precedent. Interval updating can use it once model use is authorized. An authored test that introduces A and then B and resolves only A can check implementation for this case. [INQUISITIVE, §3.2](https://aclanthology.org/2020.emnlp-main.530.pdf#page=3) · [Original TED-Q design](https://aclanthology.org/2020.lrec-1.141.pdf#page=3) · [Missingness bounds for full answers](semantic_tracking/tedq-full-answer-bounds.md)

<a id="研究驗證"></a>

## Research Validation

1. **Check known effects on development data.** Use the original materials, reader conditions, version assignment, and task to obtain model responses. Compare them using the study's measurement units and scoring rules. Any material or effect used to train the model, design prompts or losses, select a model, or set thresholds is development evidence.
2. **Test effect transfer in held-out experiments.** Fix the validation experiment families before development. Group source texts, rewrites, studies that reuse materials, and reanalyses into the same family. Report new readers, new texts, and complete held-out experiments separately. Across experiments, use the same fixed model weights and the predefined head or scoring rule for each endpoint. The holdout applies to CogLLM development. Exposure during backbone pretraining may be unknown. Under the same conditions, compare hierarchical statistical models, text-feature models, and frozen language-model baselines.
3. **Test model hypotheses with existing contrasts.** Once the model and analysis rules are fixed, prioritize held-out studies with matching manipulations, tasks, and endpoints, such as premise distance or coherence. If two explanations fit development results, look for an existing contrast on which their predictions differ. Model responses to new texts can generate hypotheses. Record those without matching observations as currently untestable. [Mandarin premise-distance experiment](https://journal.psych.ac.cn/xlxb/CN/abstract/abstract210.shtml) · [Prior knowledge and coherence experiment](https://doi.org/10.1207/s1532690xci1401_1)

Before inspecting model test results, define the primary endpoint, scoring rule, unit of analysis, estimand, acceptable error, and treatment of uncertainty. For error rates, the rewrite effect is the difference between text conditions for the same reader population. The difference in rewrite effects across reader groups tests prior knowledge as a moderator. Original assignment and manipulation constrain causal interpretation. Agreement with these contrasts supports behavioral agreement. Distinguishing mechanisms requires an existing manipulation on which the explanations predict different results.

Centaur demonstrates training on trial-level responses from multiple psychology experiments and testing unseen task variants. Reading has two direct precedents. AMoC v4.0 uses ChatGPT to build concept maps updated sentence by sentence and compares inconsistency effects in specified regions with existing experiments. Bai et al. use hierarchical reinforcement learning to model reading strategies and compare proposition recall across prior-knowledge and text-coherence conditions, plus timed comprehension observations from their study. CogLLM's primary test asks whether effects calculated from the same small model's responses reproduce observations in complete held-out experiments. [Centaur, Figs. 1 and 3](https://doi.org/10.1038/s41586-025-09215-4) · [AMoC v4.0, Studies 1–3](https://files.eric.ed.gov/fulltext/ED642779.pdf) · [Bai et al., Fig. 3](https://doi.org/10.1038/s41562-026-02534-0)

The model can help select materials that distinguish competing hypotheses. In a preprint on multiattribute decisions, Jagadish et al. use Centaur to generate simulated responses, update hypotheses, and then test them in held-out human experiments. CogLLM proposes the same validation sequence for reading. [Jagadish et al., Fig. 1](https://arxiv.org/html/2609.08003)

<a id="模型輸入閱讀反應與分析"></a>

## Model Inputs, Reading Responses, and Analysis

**Article interface.** The full text and optional reader background are the inputs. One forward pass returns a response score at each token and scores for spans already read. The available human labels define response types, such as unresolved reference, failed cross-sentence connection, and explicit clarification. The agent converts existing free-text reports into training and validation labels under predefined rules. The analysis code builds sentence-level summaries and highlights from source-text offsets.

**Experiments and analysis.** Fix visible text, lookback rules, reading stops, and reporting procedure for each study. Questions, candidate answers, human difficulty labels, and version labels stay in the research workflow. Question-based contrasts can test particular interpretations, recall, and error rates separately. Estimate uncertainty about the human population from the human data.

- **Reader background.** `knowledge_context=None` indicates that the caller supplied no personal data. Any population score must identify the study-defined reference readers and task. The PLABA-EVAL baseline applies to that study's readers and English biomedical abstracts. Individual conditioning uses available prereading information, such as `reader_background` or a concept pretest. Demonstrated unfamiliarity and an unmeasured concept are separate states. Conditioning fields record reader knowledge. Explanations, cues, and visible text supplied by an experiment belong to its reading procedure.
- **Locations.** Record what a span represents and where it comes from. Reader-marked difficulty, researcher-defined inference targets (`target_span`), and evidence or premises (`support_spans`) each use their own validation data. A sentence can contain several relations, including reference and causal inference. Source spans are half-open Unicode code-point intervals `[start,end)`. Token masks are computed by intersecting the spans with tokenizer offsets. The mapping layer converts to UTF-16 for frontends that need it. Record an evaluated span with no mark separately from a missing evaluation. Passage-level answers support reporting at passage or item level.
- **Intensity.** State the estimand and measurement. Difficulty marking rate is the proportion of responses in the same condition that mark the span. `comprehension_error_risk` is the probability that the specified reader population answers a defined local item incorrectly and needs calibration on human answers. `mental_effort` uses a separate effort scale. PLABA-EVAL provides 1–5 scales for processing ease and self-rated comprehension. Where a corresponding measure is missing, the score is `null`. Endpoints retain their own scales.

<a id="已完成的首輪基線"></a>

## Completed First-Round Baselines

The PLABA-EVAL release contains 39 pairs of English biomedical abstracts and plain-language rewrites. Averaging the reader ratings included in the study analysis gave 609 sentence means from 1,826 dual-scale ratings. Each focus sentence was appended in sentence order to the same full abstract. The baseline took final-layer representations at the corresponding `Assessment:` locations, so later focus states also saw focus sentences appended earlier. With the language model frozen, each sentence representation received L2 normalization and ridge regression fit the two ratings. Five folds were split by source-abstract pair, keeping the original and rewrite together. Length standardization was estimated within each training fold. This full-context baseline tests mean group ratings on held-out articles. Changes in token-level responses and individual trajectories require separate validation. [PLABA-EVAL study](https://aclanthology.org/2026.readi-1.4/) · [Data version and license](https://github.com/SDS-NLP/PLABA-EVAL/tree/153082e7d58b833d5e2f481b92fdcf9248c60d11)

Scoring uses held-out sentence-level mean absolute error (MAE) on the 1–5 scale. The length baseline takes `log1p` of character count and whitespace-separated word count. The version baseline also uses an indicator for the original version. PLABA-EVAL provides this field, which is generally unavailable for arbitrary articles.

The historical open-book analysis used a single version-specific sentence index in the response table to retrieve held-out predictions. Of 935 responses, 932 linked to a source sentence by that index, including 133 wrong answers. Stage 2 found **213 linked indices outside the evidence range listed in the question table** and 3 indices with no matching sentence. The table retains scores from that historical index for reference. Using the complete evidence range and newer causal ordinal predictions gives a local error-ranking AUROC of 0.557. [Stage 2 audit and corrected results](stage-2-data.md#plaba-eval-evidence-sentence-correction) · [Original historical metrics](../../research/results/plaba-eval-metrics.json)

| Sentence rating baseline | Processing ease MAE ↓ | Self-rated comprehension MAE ↓ |
| --- | ---: | ---: |
| Training-fold mean | 0.739 | 0.805 |
| Length | 0.635 | 0.724 |
| Frozen Qwen | 0.678 | 0.723 |
| Frozen Qwen + length | 0.626 | 0.681 |
| Length + version | 0.615 | 0.678 |

| Historical error ranking | AUROC ↑ | Average precision ↑ |
| --- | ---: | ---: |
| Length at the old sentence index | 0.530 | 0.177 |
| Length + version at the old sentence index | 0.555 | 0.212 |
| Qwen score at the old sentence index | 0.577 | 0.183 |
| Qwen + length score at the old sentence index | 0.563 | 0.190 |
| Mean Qwen sentence score for the whole article | 0.581 | 0.185 |
| Qwen score at the old index relative to the article mean | 0.532 | 0.163 |

Qwen + length lowered self-rated comprehension MAE from 0.724 for length to 0.681. With version provenance available, length + version reached 0.678. This is an experimental comparison within the same corpus.

The sentence-level rating MAEs use independent rating data. The current interpretation of open-book answers uses paired validation with the corrected evidence inventory. [Valid comparisons](stage6/plaba-corrected-evidence.md)

Reproduction used Apple Silicon, Python 3.11.14, MLX 0.32.2, MLX-LM 0.31.3, NumPy 2.4.6, and Transformers 5.17.0. The data directory needs the pinned versions of `sentences.csv`, `annotations.csv`, and `mcq_responses.csv`. The model directory uses official revision `15852e8c16360a2fea060d615a32b45270f8a8fc`. The output JSON records input hashes and each fold's pairs. The NPZ contains Qwen and Qwen + length weights refit on all data for later external validation. For Qwen + length, standardize length by the NPZ means and standard deviations, divide by √2, and append the result to the normalized Qwen vector. Clip output to 1–5. Holdout evaluation uses a head fit separately within each fold. [Training script](../../research/train_plaba_probe.py) · [Readout weights](../../research/results/plaba-eval-head.npz)

```bash
python research/train_plaba_probe.py \
  --model "$MODEL_SNAPSHOT" \
  --data "$PLABA_EVAL_DIR" \
  --output research/results
```

<a id="把文獻效應變成可反駁的測試"></a>

## Falsifiable Tests of Published Effects

| Reading phenomenon | Existing experimental contrast | Observable outcome |
| --- | --- | --- |
| Prior knowledge and coherence | Use the original high- and low-coherence texts and prior-knowledge groups. | Compare within-group condition differences by question type, then test the interaction using a difference in differences. [McNamara et al. 1996, Experiment 2](https://doi.org/10.1207/s1532690xci1401_1) |
| Providing a necessary premise | Compare a technical definition with a plain-language explanation. | Gu et al. observed a 9.5 percentage-point difference in content-question accuracy on their materials. Use the same endpoint when the original materials and conditions can be reconstructed. [Gu et al. 2017, Study 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC5977593/) |
| Cross-sentence monitoring and localization | The original study substituted one word to make a target sentence consistent or inconsistent with prior text, then asked readers to detect and select the sentence. | Released data provide combined detection and sentence-selection accuracy. With complete stimuli, compare the same endpoint. Individual sentence-selection distributions lack a released field. [Tibken & Tiffin-Richards 2025](https://doi.org/10.1007/s11409-025-09440-2) · [Released data](reading-evidence.md#data-available-for-training-and-evaluation) |
| Within-sentence ambiguity and lingering errors | The original study used semantic-role questions about the subordinate and main clauses of temporarily ambiguous sentences. | Align ambiguous and disambiguating regions to the source text. Compare each semantic relation on which answers are available. [Christianson et al. 2001, Experiment 2](https://ferreiralab.faculty.ucdavis.edu/wp-content/uploads/sites/222/2015/05/Christianson-et-al.-2001_LingeringInterpretationsGP_Cog-Psych.pdf) |
| Mandarin reference and causal cues | Use the original reference candidates and paired causal or concessive connective manipulations. | Continuations test reference distributions. Self-paced studies provide reading-time endpoints. [Zhan et al. 2020](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237012) · [Scholman et al. 2026](https://link.springer.com/article/10.1007/s10936-026-10269-2) |

Record the original study, language, readers, materials, conditions, measurement unit, estimated value, uncertainty, and data availability for each effect. Report sample sizes, estimated contrasts, and intervals by study. Retain uncertainty for nonsignificant results. To assess equivalence, define a substantively meaningful range first and test whether the estimate is precise enough. Compare passage effects at passage level. Group original data and later reanalyses into one family. The literature defines research targets and comparators. Direct training depends on what studies release and how those data may be used.

<a id="全資料聯合訓練與關鍵效應封存"></a>

## Joint Training on Eligible Data and Sealed Key Effects

Select the required human contrasts by phenomenon coverage, manipulation informativeness, endpoint precision, and data reconstructability. Where multiple independent sources exist, prioritize sealing at least one entire study or shared-material family unused in training and design selection. Hold out a portion of large development studies by material group for model and weight selection. Also hold out new readers where the design permits. Split originals, rewrites, materials reused in replications, and reanalyses of the same responses together. Final held-out responses and effect estimates stay outside the training targets. Kerwer's subheading effect and Paape and Vasishth's rereading effect have already informed diagnostics and count as development evidence. [Material and reader folds](stage-2-data.md#split-and-validation-scope) · [Process-specific contrasts](expository-reading-plan.md#從句級預測到閱讀歷程)

After removing validation and model-selection data, include all human observations with suitable usage terms, stimulus alignment, and measurement definitions. Fit trial-level observations with endpoint-specific likelihoods or interval-event likelihoods. For published aggregate estimates with comparable stimuli and conditions, supervise differentiable condition differences or interactions computed from model outputs. Weight effects by study precision and account for dependence. Treat the same trials and their summaries as one evidence source. Findings that establish a direction only, lack reconstructable conditions, or lack a corresponding output stay in the effect registry until suitable data can connect them. Keep the interval for a nonsignificant effect. [Effect registry](stage-1-observations.md) · [Weak supervision with expectation constraints](https://jmlr.org/papers/v11/mann10a.html)

Connect a small shared state module and endpoint readouts to one frozen Qwen3.5-2B causal forward pass. During development, balance sampling by study family and task, control gradient weights, and report every endpoint to check for negative transfer. On fixed development splits, compare separate supervised models, joint trial supervision, and joint trial plus effect supervision. Check prefix invariance, trial-level prediction, condition-effect magnitude, and timing for their respective endpoints. Compare backbone LoRA with the same data and acceptance criteria. If a later study fits reading actions that change what becomes visible, compare behavior cloning and reinforcement learning and validate against studies that record actions and subsequent visible text. [Earlier joint and effect comparisons](joint-effects-run.md) · [Three-task sharing results](semantic-joint-first-run.md) · [Expository LoRA results](expository_lora/results.md)

<a id="訓練目標的數學形式"></a>

## Mathematical Training Objective

Send the article and optional reader background through Qwen once. Parallel readout heads estimate the probability of a response at the current location and the conditional probability of a link to a span already read. Estimate intensity on the corresponding human scale. Render highlights from response or linked-span scores. The model delivers these outputs without text generation.

The full specification defines feature extraction from Qwen layers, readout equations, losses with missing labels, training on published effects, reader-specific state tracking, and reproducible criteria for human validity. Training and validation follow the stop positions and scales of human observations. [Full mathematical specification](mathematical-formulation.md)

Later joint training derives each observation likelihood from the original study's presentation, sampling, and reporting procedure and updates a shared reading state. Data integration identifies what the reader had seen, which text the response referred to, how the event was recorded, and whether other fields are derived from the same response. Sentence-end ratings, postreading ratings, event intervals, and study aggregates use their own observation processes. [Deriving training signals from human observations](observation-model.md)

The 24 integrated datasets have been aligned to their original studies. The updated training entry point uses measurement-specific likelihoods, coordinates, and shared projections. [Source alignment and implementation specification for 24 datasets](union_training/observation-protocol.md)

Full L4 feature extraction is checked and saved in document batches. A resumed run processes only batches that have yet to pass local validation. After merging all batches, run the planned joint training. [Batch saving and resumption](union_training/observation-resume.md)

<a id="訓練順序"></a>

## Training Sequence

<a id="1-登錄效應與刊出的證據"></a>

### 1. Register Effects and Published Evidence

Each entry records one verifiable observation from the original study. Required fields are the study and experiment, shared-material or trial family, language and reader population, visible text and task, two or more conditions, endpoint and scale, aggregation denominator and pairing, effect size and uncertainty, material and data availability, development or holdout status, and validation tolerance. One paper can have multiple entries. Separately record measurements taken after reading, at sentence end, or around a phrase. Group source texts, rewrites, reanalyses, and shared materials by family. Fix development and whole-family holdouts first. Previously used PLABA-EVAL is development data.

| Available evidence | Training or validation |
| --- | --- |
| Original stimuli and trial-level human responses | Train on the likelihood for that response type. Use effects aggregated from those trials to check whether the model learned the original condition difference. |
| Original stimuli and aligned aggregate estimates, such as condition means or between-group differences | Compute the expectation or contrast of model probabilities under the same conditions. Weight effect loss by sampling precision and record covariance for related endpoints. |
| Direction or description only, or incomplete stimuli or statistical precision | Record the conditions to which the finding applies. If the contrast can be reconstructed, use it to diagnose direction. If comparable materials or responses are missing, mark it untestable and continue with other endpoints. |

For free continuations in the Mandarin study, compute each reader's proportion of continuations about the first person by verb condition from the published CSV. Averaging equally across 50 readers gives **77.7% versus 11.7%**, a difference of about **66.0 percentage points**. Pooling individual trials gives 79.1% versus 11.9%. The pronoun-provided condition has its own effect. Trial-level continuations can train the reference head and the original contrast can check it. In a later study publishing only a comparable contrast, recompute it from model probabilities for weak supervision. [Zhan et al., Experiment 1 and Fig. 1](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237012#sec005) · [Trial-level CSV](https://osf.io/download/8zk57/)

<a id="2-先用逐次觀測訓練平行讀出頭"></a>

### 2. Train Parallel Readouts on Trial-Level Observations

Freeze Qwen3.5-2B. Make one forward pass per article and available reader background. At the observation locations, collect predefined features from hidden states, attention, or vocabulary logits. Train small parallel heads. The first sources are as follows.

- **PLABA-EVAL** uses ratings with the full text visible to train sentence-level ease, self-rated comprehension, and difficulty types. The existing regression on mean sentence ratings provides a comparison. Its 858 structured difficulty records include explicit reports of no difficulty. [Original measurements and data](https://github.com/SDS-NLP/PLABA-EVAL/tree/153082e7d58b833d5e2f481b92fdcf9248c60d11)
- **Mandarin continuations**, if assigned to development, use the reference selected in human continuations as a category, conditioned on the visible text and whether a pronoun has appeared. Source spans and known manipulations stay in the research workflow. [Zhan et al., Experiments 1–3](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237012)
- **Reader reports and selected spans** use INQUISITIVE sentence-end questions and the phrases they refer to once aligned prefixes are available under suitable usage terms. Calibrate question rates only with a known denominator and matching observation. Estimate resolution or reactivation intervals only when the data connect the same reader and the same question. [INQUISITIVE, §3.2](https://aclanthology.org/2020.emnlp-main.530.pdf#page=3) · [First shared-prefix test](#第一個前綴對照)

This is supervised training. If Qwen's weights need adjustment, use LoRA for supervised fine-tuning within the same single-forward architecture. Centaur provides a precedent for fine-tuning on trial-level human data. [Centaur fine-tuning method](https://arxiv.org/html/2410.20268)

<a id="3-加入跨條件的效應損失"></a>

### 3. Add a Cross-Condition Effect Loss

For each effect step, use the versions, reference populations, and fixed readout locations in one study. Make one forward pass per input to obtain probabilities or rating distributions. Apply the study design to aggregate them into a condition difference or interaction. The error between model and human estimates gives a differentiable loss. With a frozen backbone, cache condition representations and update the head on complete contrasts. Keep the trial-level loss in training to check absolute probabilities by condition and local annotations. Match model capacity, data splits, and features when comparing trial supervision against trial plus effect supervision. [Condition contrasts and effect loss](mathematical-formulation.md#從文獻效應回傳梯度) · [Weak supervision with expectation constraints](https://jmlr.org/papers/v11/mann10a.html)

The first executable comparison can use development data with original trials. Group by material. Assign some development groups aggregate condition differences only and give the others trial-level labels. With identical trial labels, test whether including the first group's condition differences improves responses and effects on independent held-out materials. Also train with complete trial labels across development as a comparison. This evaluates how much information the effect loss can use and remains separate from transfer to unseen research families.

Alignment depends on the conditions available to the model and the outputs it produces. For high- and low-knowledge readers, use the original study's measurable background. Empty background represents the specified reference population. If different questions test separate semantic relations, the analysis maps those questions to the model's existing relation outputs using fixed rules before evaluating each endpoint. Record missing outputs or human measures in the effect table. If published effects point in opposite directions, first check conditions, scales, question types, and uncertainty, then retain comparable discrepancies.

The single-forward design computes distributions and expectations directly. Evaluating lookback or skimming policies requires existing data that record actions and the information visible. Bai et al. train reading actions with deep reinforcement learning and use numerical optimization to fit aggregate human effects. [Bai et al., Training, Parameters and fitting](https://arxiv.org/html/2603.06006)

<a id="4-固定模型後逐個檢驗留出效應"></a>

### 4. Freeze the Model and Test Each Held-Out Effect

First check that replacing later text preserves outputs on the shared token prefix. Use one fixed checkpoint to estimate response distributions, effect directions, and effect sizes by condition. Test linked spans and onset intervals where matching annotations exist. Derive intervals for model–human effect differences from sampling uncertainty across existing readers and material families. Prespecify a meaningful tolerance in the original units. Mark each required effect as matching, deviating, underpowered, or untestable. Full reproduction applies to a prespecified scope only when every required effect passes. If an effect has only one family and it was used during development, record the absence of independent validation. [Human validity specification](mathematical-formulation.md#真人效度)

<a id="資料的輔助用途"></a>

### Additional Uses of the Data

The [inventory of human observations](reading-evidence.md#data-inventory-for-the-initial-readout-heads) lists fields, availability, and the timing each source can supervise.

- **PLABA-EVAL and OneStop** provide full-text ratings and postreading comprehension, respectively. **Jones & Bergen** use two-alternative reference choices to test candidate interpretations, with reading times from a separate group for later processing effects. Trial-level data and stimulus fields have been retrieved. The study exclusions and data reuse terms still need checking. [PLABA-EVAL](https://aclanthology.org/2026.readi-1.4/) · [OneStop variables](https://lacclab.github.io/OneStop-Eye-Movements/variables) · [Jones & Bergen effect entry](stage-1-observations.md#e06-jones--bergen-later-onset-reading-delay)
- **Mandarin continuations and German inconsistency detection** supply further contrasts. The first records the reference chosen in readers' continuations. The second releases combined detection and sentence-selection accuracy, with separate tracking of full stimulus availability. [Zhan et al., Experiments 1–3](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237012) · [German materials and responses](https://osf.io/ekf4t/?view_only=1cd488d47c254163bfc032fbbc7fa0cc)

<a id="qasper-之後的訓練對照"></a>

## Training Comparisons After QASPER

The next round tests whether the model distinguishes when text evidence appears for different questions about the same article. Start with the frozen Qwen3.5-2B backbone, one causal forward pass, and separate endpoint readouts. In earlier same-paper QASPER pairs, the body-text readout reduced evidence-interval NLL by 0.1610 nats relative to position. The original versus swapped question contrast was −0.0234 nats with 95% interval [−0.0822, +0.0343]. Same-stop ranking agreed for 21 of 41 pairs. The results suggest two separable hypotheses. Finer question–text interaction could supply useful information, or the existing representation could carry the information but the interval likelihood may supply too little ranking supervision. [QASPER holdout results](qasper-evidence-run.md#held-out-results)

<a id="第一個實驗"></a>

### First Experiment

Use 151 questions from the same 60 training papers and 121 within-paper pairs with different evidence locations. All four arms use per-question interval likelihood. Compare changes from the readout and from adding the loss. Fix visible prefixes, training materials, data weights, and seeds. Report differing readout capacity and retain the two matched comparisons to estimate the contribution of paired supervision. [Original data and alignment](../../research/results/qasper-evidence-first-run/data-audit.json)

| Question–article readout | Interval likelihood | Interval likelihood + same-paper, same-stop ranking |
| --- | --- | --- |
| Original interaction with paragraph-end and mean paragraph representations | Original baseline | Whether paired supervision improves the original readout |
| Question representation interacting with already-read article tokens | Finer readout alone | Paired supervision with the finer readout |

The finer readout scores visible tokens with a known question representation, using small prefix-masked cross-attention if needed. Make one backbone pass per article and use the same causal states to score all questions and stops in parallel. For two questions from the same paper at the same stop, let \(s_i(t)\) be the log-odds that annotated evidence for question \(i\) had appeared by stop \(t\). If evidence for A has appeared and B's annotated evidence comes later, add \(\mathrm{softplus}(s_B(t)-s_A(t))\). A different annotator group retrospectively chose QASPER evidence after seeing the full article. The ranking supervises relative order of annotated evidence. B lacks an absolute zero-probability label. Ranking and likelihood derive from the same annotations. Record their weights separately and use development materials to measure the effect of adding ranking. [Interval likelihood](semantic-joint-first-run.md#interval-loss-and-one-pass-forward) · [Paired ranking method](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/) · [QASPER annotation design](https://aclanthology.org/2021.naacl-main.365.pdf#page=3)

The 23 previously scored QASPER papers are development diagnostics. The formal comparison uses papers this project has never evaluated. Fix materials and scoring before inspecting new results. Primary checks are ranking within the same paper and stop, interval error for correct versus swapped questions, and calibration at stops where annotations establish whether evidence has appeared. Retain position-only and question-only baselines to estimate variation explained by position or question type. Question swaps and equal-length evidence replacements test the model's response to a particular question. Evaluate human endpoints where source-text evidence has human annotations. Use the replacements as model-behavior controls. Prespecify acceptance criteria for ranking and interval error and tolerable calibration degradation. Estimate uncertainty of differences by paper. Include the readout in joint semantic training only if it meets these criteria on new papers. If question specificity remains low, inspect pairing precision and question representations before comparing supervised LoRA on the same target. [Original three comparisons](qasper-evidence-run.md#held-out-results) · [LoRA method](https://arxiv.org/abs/2106.09685)

<a id="新-qasper-論文的固定評分"></a>

#### Fixed Scoring on New QASPER Papers

The formal candidate is `token_rank`, the tokenwise interaction readout with ranking. Under identical initialization and regularization, all four arms have the same number of readout coefficients. The finer readout computes question interactions for tokens in the paragraph just read, then pools with log-mean-exp. The paragraph-end representation supplies context. Prior paragraphs contribute to the cumulative arrival probability. Aggregation and scoring remain at the annotated paragraph stops. The ranking weight is 0.5 times the mean loss over all strict within-paper pairs, added to the average likelihood per question. All four arms and both baselines use the original 60 training papers. The new holdout is the 27 papers in official dev that meet fixed criteria, with 65 questions and 49 within-paper question pairs. [Holdout construction and rules](../../research/semantic_tracking/qasper_prepare.py)

Use 10,000 paired resamples grouped by paper. The candidate's same-stop ranking must outperform both position-only and question-only controls, and its negative log likelihood for the evidence interval must be lower for original questions than for within-paper swaps. For each of these three tests, the corresponding bound of the paired 95% interval must pass zero in the intended direction. Average the Brier error of cumulative arrival probability over paragraph stops within a question, then weight questions equally. Relative to the paragraph readout with interval likelihood, the upper bound on the increase must be at most 0.01. The two-sided interval for mean predicted probability minus annotated proportion must lie within ±0.05, a tolerance of five percentage points. Report paired intervals for the four-arm decomposition and its interaction separately. These tolerances are prespecified design choices. [Sealing and scoring script](../../research/semantic_tracking/qasper_factorial_score.py)

<a id="用文獻校準聯合模型"></a>

### Calibrate Joint Models Against Published Results

Trial data supervise each observable response. Rereading events, selected prior spans, self-rated comprehension, and answer intervals use their own likelihoods and original scales. To integrate published effects, start with one or two that have original materials, conditions, endpoints, and statistics. For Zhan's verb-type by pronoun-presence interaction, calculate the difference in differences using the model's response rate in each original condition, then compare with the human estimate and its uncertainty. Effects recalculated from trials already used for direct supervision serve primarily as acceptance checks. Add aggregate effects to the loss when they supply independent supervision or when dependence is modeled. Registry and acceptance tables retain directional findings without identified materials or conditions. [Effect-loss equations](mathematical-formulation.md#從文獻效應回傳梯度) · [Weak supervision with expectation constraints](https://jmlr.org/papers/v11/mann10a.html)

On Stage 4's within-study material holdout, aggregate supervision moved the Zhan interaction from −4.09 to +1.12 percentage points, compared with +24.46 for humans. Adding effect supervision to the later shared model improved trial-level scores. The improvement intervals for both held-out effect errors crossed zero. Joint training therefore checks trial-level scores, response distributions in each original condition, and effect magnitudes individually. Separate heads provide endpoint baselines. First compare partially shared designs of similar capacity, with predefined acceptable degradation for each primary readout. Test transfer on new materials or whole-study holdouts. [Stage 4 comparison](stage-4-zhan-weak-supervision.md#held-out-results) · [Shared-effect results](joint-effects-run.md#the-magnitude-of-the-publication-bias-effect) · [Three-task semantic comparison](semantic-joint-first-run.md#held-out-results)

Differentiable trial, ranking, and effect losses suit supervised training. Freeze the backbone to isolate changes in supervision and readout. If fine-tuning improves results, determine which Qwen parameters to update. Reinforcement learning applies to separate questions about reading policies with explicit actions and outcomes, for example when lookback changes the text visible at the next step and public trajectories support policy evaluation. The article interface still makes one causal forward pass. After known-question experiments pass, test whether text alone can support constructing and tracking the same question from the visible prefix. [Single-forward specification](mathematical-formulation.md#2-一次因果前向運算) · [Reading-action study](https://arxiv.org/html/2603.06006)

<a id="iirc-純正文片語定位"></a>

## IIRC Phrase Localization From Article Text Alone

This contrast tests whether article text alone can localize multiple sources of questions. In IIRC, readers read a linked English Wikipedia paragraph, ask a question, and select the link expected to supply the answer. Keep questions that match uniquely to a source link and retain original Unicode phrase boundaries. Each training observation is the selected phrase for a question already asked. Unselected phrases have no negative “no question” label. The model sees article text alone and ranks phrases conditional on this class of question being present. Different people answered the questions in the original study. This analysis evaluates the material to which the question refers. [IIRC §2](https://aclanthology.org/2020.emnlp-main.86.pdf#page=3) · [Alignment and license](../../research/results/iirc-preparation-audit.json)

Both arms return the five highest-ranking nonoverlapping spans from the same plain-text candidates. The surface baseline uses phrase length, relative position, capitalization, numbers, and punctuation. Qwen adds phrase-start and phrase-end states from one causal forward pass. Normalize scores for the article's candidates at paragraph end, and train a conditional multinomial model on each selected human target. Use 4,754 official training articles and their 7,195 uniquely alignable observations. Record any observation outside the fixed candidate set separately. Evaluate all 430 official dev articles. Human target selections, original links, and question text remain local until predictions and weights are sealed. [Data splits](../../research/semantic_tracking/iirc_prepare.py) · [Scoring rules](../../research/semantic_tracking/iirc_anchor_score.py)

The primary measure is the proportion of targets covered by at least one predicted phrase at character intersection over union of 0.5. The second primary measure is the proportion of pairs of different targets at the same stop covered by **different** predicted phrases. Both must beat the surface baseline on new official dev articles. Use 10,000 paired resamples by source article family and require a paired 95% interval lower bound above zero. Report exact-boundary coverage and the difference between top-five probability mass and exact hits as conditional-score diagnostics. The observation granularity is paragraph end. The original study did not measure first detection or later resolution. [Sealing and scoring script](../../research/semantic_tracking/iirc_anchor_score.py)

<a id="首輪運算前的資料確認"></a>

## Data Checks Before the First Run

- Check original publications, trial files, and material usage terms, then assign development and holdout materials.
- Set required endpoints and reproduction tolerances using the reader population, task, stop position, and scale of each existing dataset. Use published or recomputed uncertainty to determine achievable precision. Retain insufficiently measured endpoints in the effect table.

<a id="plaba-eval-回歸基線的工程驗收"></a>

## Engineering Acceptance for the PLABA-EVAL Regression Baseline

**Verdict**: PASS

### Preference Fit

- The sentence-level regression head trains on original human ratings using one pinned data version and model path.

### Correctness

- Each source pair belongs to a single holdout fold. Matrix shapes and values were checked for both heads.

### Evidence Run

- The Apple GPU encoded all 78 abstracts and produced held-out metrics and weights. Checks covered 609 sentences, 1,826 ratings, and the input SHA-256 hashes.

### Scope/Architecture

- Acceptance covers the reproducible sentence-rating baseline. Localizing comprehension and generating phrase masks requires direct human span and local-comprehension annotations.

### Required Fixes

- None.

[Original research record](../../research/training-protocol.md)
