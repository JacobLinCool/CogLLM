# Research document catalog

[Documentation home](README.md) · [Try the demo](try-cogllm.md)

Each English document links to the original research record and to any supporting code, predictions, or data manifests.

## Research stages and study reports

- [Anglin Study 1 · Repeated belief judgments and confidence](research/anglin-study1-audit.md)
- [CogLLM App Design](research/app-concept.md)
- [Reader background and English lexical difficulty](research/core-features-background-transfer.md)
- [First experiments on lexical difficulty, rereading, and reader background](research/core-features-first-run.md)
- [CogLLM Research Execution Plan](research/execution-plan.md)
- [Reading Load in Research Papers and Expository Text](research/expository-reading-plan.md)
- [CogLLM Interim Research Review](research/findings-and-gaps.md)
- [IIRC phrase localization on held-out passages](research/iirc-anchor-run.md)
- [IIRC right-censored ranking results](research/iirc-censored-contrast-results.md)
- [IIRC ranking supervision with right censoring](research/iirc-censored-contrast.md)
- [IIRC evidence matching within the same passage](research/iirc-identity-holdout.md)
- [IIRC linked-evidence preparation and verification](research/iirc-linked-evidence-protocol.md)
- [IIRC shared-readout results](research/iirc-linked-shared-head-results.md)
- [IIRC shared-readout validation](research/iirc-linked-shared-head.md)
- [IIRC protocol for later support in the same paragraph](research/iirc-seed-later-protocol.md)
- [IIRC localization of earlier anchors](research/iirc-seed-pointer-run.md)
- [Joint training with supervision from published effects](research/joint-effects-run.md)
- [OneStopQA baseline and the first shared-readout experiment](research/joint-first-run.md)
- [CogLLM research progress and the first joint-training experiment](research/joint-training-plan.md)
- [CogLLM Mathematical Specification](research/mathematical-formulation.md)
- [TECO document ratings, comprehension, and reader background](research/nahatame-teco-data-audit.md)
- [Defining CogLLM's Training Signals from Human Observations](research/observation-model.md)
- [Cross-corpus diagnostics for PLABA-EVAL issue spans](research/plaba-span-transfer-protocol.md)
- [Locating textual evidence in QASPER](research/qasper-evidence-run.md)
- [QASPER comparisons of token interactions and ranking supervision](research/qasper-factorial-run.md)
- [Human reading comprehension and processing cost](research/reading-evidence.md)
- [Joint training for three semantic tracking tasks](research/semantic-joint-first-run.md)
- [Locating questions, earlier references, and later updates](research/semantic-tracking-first-run.md)
- [Stage 1 · Evidence from human reading studies](research/stage-1-observations.md)
- [Stage 2 human observation data](research/stage-2-data.md)
- [Stage 3 · Feature ablations for sentence-level difficulty](research/stage-3-feature-ablation.md)
- [Stage 3 initial Colab L4 experiment](research/stage-3-first-run.md)
- [Stage 3 Chinese sentence-initial pronoun contrast](research/stage-3-zhan-landmark.md)
- [Stage 4 · Held-out validation of the reference readout](research/stage-4-zhan-next-validation.md)
- [Stage 4 · State changes around sentence-initial pronouns](research/stage-4-zhan-update.md)
- [Stage 4: Supervised effect of Chinese sentence-initial pronouns](research/stage-4-zhan-weak-supervision.md)
- [Stage 5 · CompLex context and training data comparisons](research/stage-5-complex-run.md)
- [Stage 5 · Scope for localization and dataset expansion](research/stage-5-scope.md)
- [Stage 5 · Semantic localization, question updates, and cross-sentence links](research/stage-5-semantic-run.md)
- [Stage 6 held-out validation](research/stage-6-holdout-plan.md)
- [Stage 6 · RACE comprehension validation](research/stage-6-race-run.md)
- [CogLLM Protocol for Reproducing and Analyzing Human Reading Behavior](research/training-protocol.md)

## Reading research and source selection

- [Stage 1 · Text Coherence and Reader Knowledge](research/stage-1/coherence-and-knowledge.md)
- [Stage 1 Training and Validation Signal Selection](research/stage-1/data-selection.md)
- [Stage 1 · Linguistic Cues, Text Integration, and Comprehension Updating](research/stage-1/language-and-updating.md)
- [Stage 1 literature extraction · Measures and boundaries of the effects](research/stage-1/measurement-and-boundaries.md)

## Semantic tracking and resolution

### First reports of inconsistency in Frick

- [Frick's condition of intra-sentence antecedent similarity](research/semantic_tracking/frick-conditional-protocol.md)
- [Pairwise sentence matching training slightly increases the target sentence effect](research/semantic_tracking/frick-conditional-results.md)
- [Sentence-by-sentence contradiction report and first-position verification for Frick and Schüler](research/semantic_tracking/frick-consistency-protocol.md)
- [Sentence-by-sentence contradictory reports of the held-out results by Frick and Schüler](research/semantic_tracking/frick-consistency-results.md)
- [Sentence-by-sentence contradiction judgments between Frick and Schüler](research/semantic_tracking/frick-consistency-source.md)
- [Frick's cross-sentence causal feature contrast](research/semantic_tracking/frick-context-protocol.md)
- [cross-sentence word form and representation features did not improve, contradicting the first report](research/semantic_tracking/frick-context-results.md)
- [The first-order calibration of the location probability for the Frick contradiction report](research/semantic_tracking/frick-mass-calibration-protocol.md)
- [The first contradiction report was less early, and the target sentence effect remained small](research/semantic_tracking/frick-mass-calibration-results.md)
- [Calibrate Frick's sentence-by-sentence report using the paired-context effect](research/semantic_tracking/frick-pair-protocol.md)
- [Frick's prior matching-supervised story held-out results](research/semantic_tracking/frick-pair-results.md)
- [The next sentence is judged consistent. does this imply the original query has been resolved?](research/semantic_tracking/frick-resolution-negative-control.md)
- [Can Qwen's per-token expected value and prefix attention improve first-pass contradiction report prediction?](research/semantic_tracking/frick-signals-protocol.md)
- [per-token vocabulary and attention signal end-of-sentence contradiction report verification](research/semantic_tracking/frick-signals-results.md)
- [Frick first reported that the primary difference stems from the target sentence at hand](research/semantic_tracking/frick-target-hazard-diagnostic.md)
- [Frick causal peak calibration specification](research/semantic_tracking/frick_peak_calibration_protocol.md)
- [Frick causal peak calibration's held-out results](research/semantic_tracking/frick_peak_calibration_results.md)
- [Frick Contradiction Source Sentence Localization Specification](research/semantic_tracking/frick_reference_protocol.md)
- [Frick's earlier text manipulation sentence localization results](research/semantic_tracking/frick_reference_results.md)

### Interpretation and reader knowledge

- [Readers' practice history and story-line held-out results](research/semantic_tracking/beukers-background-results.md)
- [Reader practice history and story clues](research/semantic_tracking/beukers-background.md)
- [Independent transfer validation of practice order](research/semantic_tracking/beukers-inserted-protocol.md)
- [Independent transfer results of practice order](research/semantic_tracking/beukers-inserted-results.md)
- [Cross-study held-out results for Blott's re-interpretation of the term "semantic blott"](research/semantic_tracking/blott-format-results.md)
- [Blott et al. (2023) proposed a re-interpretation of the final word meaning in English sentences](research/semantic_tracking/blott-format-source.md)
- [Supervised training of disambiguation cue representations and cross-study validation](research/semantic_tracking/blott-update-results.md)
- [Locate using disambiguation cues from the body text only](research/semantic_tracking/blott-word-results.md)
- [Cross-study held-out specifications for Blott's re-interpretation of the term "semantic blott"](research/semantic_tracking/blott_format_transfer_protocol.md)
- [Representational training before and after disambiguation cues for end-consequence judgment](research/semantic_tracking/blott_update_protocol.md)
- [Use only in-text positioning disambiguation cues](research/semantic_tracking/blott_word_protocol.md)
- [The first disambiguation cue at the readout when encountering the term](research/semantic_tracking/blott_word_stream_protocol.md)
- [End-of-sentence interpretation of Czech garden-path sentences](research/semantic_tracking/czech-garden-path-source.md)
- [Czech disambiguation readout with held-out results](research/semantic_tracking/czech-garden-results.md)
- [Independent validation of Czech disambiguation and residual misunderstanding](research/semantic_tracking/czech_garden_protocol.md)
- [Segment-wise judgment updates by Qiao and Lagnado](research/semantic_tracking/qiao-evidence-source.md)
- [When only the main text is input, Qiao's two scoring timelines share the same prediction](research/semantic_tracking/qiao-mode-identifiability.md)
- [Rodger Study 1: Two-reading stop judgment update fixed comparison](research/semantic_tracking/rodger-belief-protocol.md)
- [Rodger Study 1 material held-out judgment update results](research/semantic_tracking/rodger-belief-results.md)
- [Rodger et al.'s two reading stop written belief update data](research/semantic_tracking/rodger-belief-source.md)

### Rereading and links to earlier text

- [ARAU 2.1 bridging prefix held-out data](research/semantic_tracking/arrau-bridge-intake.md)
- [The ARRAU independent held-out comparison of the preceding context anchor span localization](research/semantic_tracking/arrau-bridge-protocol.md)
- [ARAU prior positioning held-out results](research/semantic_tracking/arrau-bridge-results.md)
- [Cross-sentence prefix readout matching](research/semantic_tracking/bridge-rerank-protocol.md)
- [Cross-span context-aware readout alignment results](research/semantic_tracking/bridge-rerank-results.md)
- [The first independent English readout effect relative to the preceding text](research/semantic_tracking/context-readback-transfer-results.md)
- [Independent English validation of prior context clues and first-pass readout](research/semantic_tracking/context-readback-transfer.md)
- [Only input the first readout held-out results](research/semantic_tracking/first-readback-transfer-results.md)
- [First-pass active readout of the main text input validation](research/semantic_tracking/first-readback-transfer.md)
- [GUMBridge cross-sentence context localization](research/semantic_tracking/gum-bridge-protocol.md)
- [GUMBridge cross-sentence prefix positioning results](research/semantic_tracking/gum-bridge-results.md)
- [Incremental update of the human reference during stepwise text revelation](research/semantic_tracking/written-reference-update-source.md)

### Locating supporting evidence

- [How much of the manual evidence segment is retained when pasting the body into the App?](research/semantic_tracking/contractnli-app-geometry.md)
- [ContractNLI hypothesis evidence independent held-out validation](research/semantic_tracking/contractnli-evidence-protocol.md)
- [ContractNLI's independent evidence localization results](research/semantic_tracking/contractnli-evidence-results.md)
- [ContractNLI's hypothesis evidence localization](research/semantic_tracking/contractnli-evidence-source.md)
- [Vocabulary and Qwen proposition reordering evidence](research/semantic_tracking/contractnli-rerank-protocol.md)
- [Evidence ranking through vocabulary fusion with Qwen is validated via held-out testing](research/semantic_tracking/contractnli-rerank-results.md)
- [Comparison of evidence intervals after symmetric calibration](research/semantic_tracking/evidence-calibration-comparison-results.md)
- [Evidence-to-readout symmetric calibration control](research/semantic_tracking/evidence-calibration-comparison.md)
- [Cross-domain evidence localization for Evidence Inference 2.0](research/semantic_tracking/evidence-inference-source.md)
- [Can the body text evidence for a specified question be located across genres?](research/semantic_tracking/evidence-inference-transfer-protocol.md)
- [The evidence fusion readout of the contract did not pass the held-out paper](research/semantic_tracking/evidence-inference-transfer-results.md)
- [ExpMRC's back-calibration results for the arrival rate](research/semantic_tracking/expmrc-calibration-results.md)
- [ExpMRC achieves the calibration of the reachability rate](research/semantic_tracking/expmrc-calibration.md)
- [Evidence location held-out in ExpMRC new papers](research/semantic_tracking/expmrc-transfer-results.md)
- [Freezing readout transfer of ExpMRC RACE+](research/semantic_tracking/expmrc-transfer.md)
- [Acquisition and verification of Natural Questions held-out data](research/semantic_tracking/nq-evidence-intake.md)
- [Independent evidence interval validation for Natural Questions](research/semantic_tracking/nq-evidence-source-audit.md)
- [Results from the independent hold-out interval of Natural Questions](research/semantic_tracking/nq-evidence-transfer-results.md)
- [Evidence at readout in the Natural Questions independent validation](research/semantic_tracking/nq-evidence-transfer.md)
- [QASPER has evidence of joint calibration with unanswerable questions](research/semantic_tracking/qasper-evidence-tail-results.md)
- [QASPER has evidence of joint acceptance with unanswerable](research/semantic_tracking/qasper-evidence-tail.md)

### When readers retrieve or recognize an answer

- [The first correct response position of ACT's human aligns with the clues](research/semantic_tracking/act-source.md)
- [The first correct response position of the German anchor span](research/semantic_tracking/act-transfer-protocol.md)
- [The English word-guess readout failed to reproduce the first correct response curve of the German ACT](research/semantic_tracking/act-transfer-results.md)
- [Sensitivity of the word-guessing result to the path sampling of the clue path](research/semantic_tracking/core-vocabulary-disjoint-protocol.md)
- [The new resampling path reproduces the group prediction gain of sentence-by-sentence guessing](research/semantic_tracking/core-vocabulary-disjoint-results.md)
- [Core Vocabulary marginal first-correct protocol](research/semantic_tracking/core-vocabulary-marginal-protocol.md)
- [Core Vocabulary archived-path sensitivity](research/semantic_tracking/core-vocabulary-path-sensitivity.md)
- [Qwen prefix representations and the response distribution of the group achieving first correct answers per sentence](research/semantic_tracking/core-vocabulary-qwen-protocol.md)
- [Qwen end-of-sentence representation improvements to group prediction for sentence-by-sentence word guessing](research/semantic_tracking/core-vocabulary-qwen-results.md)
- [Core Vocabulary source audit](research/semantic_tracking/core-vocabulary-source.md)
- [Comparison of nCloze options within the same article and the same span](research/semantic_tracking/ncloze-choice-contrast.md)
- [nCloze's token-by-token generation and original span coordinates](research/semantic_tracking/ncloze-source.md)
- [Vocabulary extraction responses for the same reader held-out](research/semantic_tracking/oliver-retrieval-results.md)
- [Vocabulary extraction co-occurrence trajectory and held-out specification](research/semantic_tracking/oliver_retrieval_protocol.md)
- [QANTA25 human buzz source audit](research/semantic_tracking/qanta25-source.md)
- [Sentence-by-sentence contradiction judgments by Singer et al](research/semantic_tracking/singer-validation-source.md)

### Question tracking, load curves, and source audits

- [Public hypothesis questioning, response, and original questioner confirmation](research/semantic_tracking/hypothesis-clarification-source.md)
- [Joint learning of two sequential word-guessing observation processes](research/semantic_tracking/joint-cue-hazard-protocol.md)
- [The shared word embedding coefficient did not improve held-out prediction for both tasks](research/semantic_tracking/joint-cue-hazard-results.md)
- [Frick readout at the end of Currie et al](research/semantic_tracking/larrc-consistency-protocol.md)
- [Cross-study held-out results for Currie's end-of-paper consistency judgment](research/semantic_tracking/larrc-consistency-results.md)
- [End-of-paper consistency judgment by Currie et al](research/semantic_tracking/larrc-consistency-source.md)
- [No resolution of the question, the continuation is a contrast](research/semantic_tracking/load-curve-controls.md)
- [Onea/PITQ concurrent re-opening and parallel state](research/semantic_tracking/onea-reopening-source.md)
- [Post-forward same-question state computation](research/semantic_tracking/question-states-run.md)
- [human oral and read-back data verification](research/semantic_tracking/reader-report-source-audit.md)
- [Validation of answer cues for the same query](research/semantic_tracking/same-question-protocol.md)
- [Locating the answer to the same question](research/semantic_tracking/same-question-results.md)
- [Self-contradictory word-level localization and annotation degree held-out](research/semantic_tracking/scarecrow-detection-results.md)
- [# Body Localization and Severity Specifications for Self-Contradictory Fragments](research/semantic_tracking/scarecrow_detection_protocol.md)
- [per-token curve first-round results](research/semantic_tracking/semantic-load-curve-pilot-results.md)
- [per-token unresolved question curve's first-round control](research/semantic_tracking/semantic-load-curve-pilot.md)
- [per-token unresolved semantic load training and validation](research/semantic_tracking/semantic-load-curve-plan.md)
- [The identifiable scope of TED-Q fully answered](research/semantic_tracking/tedq-full-answer-bounds.md)

## Held-out studies

- [Observation granularity of the Abel text-order experiment](research/stage6/abel-interleaving-audit.md)
- [AmbiStory's held-out results for semantic update](research/stage6/ambistory-context-update-results.md)
- [The semantic update of English narrative](research/stage6/ambistory-context-update.md)
- [AmbiStory's supervised held-out results show a significantly worse ending performance](research/stage6/ambistory-effect-supervision-results.md)
- [AmbiStory's end-difference supervised](research/stage6/ambistory-effect-supervision.md)
- [AmbiStory upper bound correction](research/stage6/ambistory-numerical-amendment.md)
- [Blott end-sentence coherence held-out and query-tracking data verification](research/stage6/blott-coherence-holdout.md)
- [CLEAR English readability held-out specification](research/stage6/clear-reading-ease-protocol.md)
- [CLEAR English readability held-out results](research/stage6/clear-reading-ease-results.md)
- [CommonLit CLEAR English discourse readability source audit](research/stage6/clear-reading-ease-source.md)
- [Cross-domain context retrieval and cross-lingual sentence-level scoring](research/stage6/external-projection-protocol.md)
- [Cross-domain and cross-lingual held-out results](research/stage6/external-projection-results.md)
- [French conjunction selection and reader background](research/stage6/french-connectives-audit.md)
- [English abstract rewriting held-out](research/stage6/kerwer-article-holdout.md)
- [Kerwer's English abstract reader background verification](research/stage6/kerwer-reader-background-protocol.md)
- [Kerwer English abstract reader background held-out results](research/stage6/kerwer-reader-background-results.md)
- [English word-by-word colored MedReadMe held-out](research/stage6/medreadme-lexical-holdout.md)
- [Gradio word-by-word color-coded MedReadMe held-out results](research/stage6/medreadme-lexical-results.md)
- [MedReadMe alignment of original text for difficult words and phrases](research/stage6/medreadme-span-audit.md)
- [OneStop: Initial and Repeated Reading](research/stage6/onestop-rereading-audit.md)
- [OneStop re-read state baseline](research/stage6/onestop-rereading-baseline.md)
- [Causal readout under OneStop re-reading conditions](research/stage6/onestop-rereading-model.md)
- [OneStop main text and rereading condition held-out results](research/stage6/onestop-rereading-results.md)
- [PLABA open-book understanding correction evidence segments](research/stage6/plaba-corrected-evidence.md)
- [PoTeC reader background held-out feasibility](research/stage6/potec-background-readiness.md)
- [The educational attainment of English open-book responses, held-out specification](research/stage6/race-education-holdout-protocol.md)
- [The educational attainment of English open-book responses in the held-out set](research/stage6/race-education-holdout-results.md)
- [discourse event sequence interpretation update](research/stage6/salt-discourse-audit.md)
- [StudentAnnotations reader background held-out results](research/stage6/student-annotations-background-results.md)
- [German sentence-level reader background data audit](research/stage6/student-annotations-background.md)
- [TECO English reading ability and discourse difficulty alignment](research/stage6/teco-proficiency-protocol.md)
- [TextComplexityDE secondary fixed readout](research/stage6/textcomplexityde-transfer-amendment.md)
- [PLABA end-of-sentence readout for external ranking validation of German self-assessment](research/stage6/textcomplexityde-transfer.md)
- [Turkish sentence-level reading difficulty data](research/stage6/turkish-sentence-source.md)
- [Yadav et al. German and Dutch sentence-level difficulty transfer diagnosis](research/stage6/yadav-transfer-audit.md)

## Lexical difficulty and reader background

- [Select the human-selected fit Gradio vocabulary to read out](research/core_features/cwi-lexical-readout-protocol.md)
- [Human selection-based fitting improves WikiNews ranking. the additional gain from CompLex remains unestablished](research/core_features/cwi-lexical-readout-results.md)
- [Gradio vocabulary score did not pass the professional news held-out](research/core_features/cwi-news-holdout-results.md)
- [Gradio vocabulary coloring professional news held-out](research/core_features/cwi-news-holdout.md)
- [Selection of human-level vocabulary from English textbooks](research/core_features/kangoo-source-audit.md)

## Expository reading and LoRA

- [Explainable sentence-level readout versus last-layer LoRA](research/expository_lora/protocol.md)
- [Frozen Backbone and LoRA for Sentence-Level Expository Reading Responses](research/expository_lora/results.md)

## Joint training and observation-specific losses

- [# First-Day Belief Shift Acceptance in the Anglin Study 1](research/union_training/formal-anglin-protocol.md)
- [Anglin's first-day belief shift results](research/union_training/formal-anglin-results.md)
- [Post-training verification of MECO's passage-level understanding](research/union_training/formal-meco-plan.md)
- [Post-training verification of the MECO passage](research/union_training/formal-meco-results.md)
- [SAP word-level self-paced reading results](research/union_training/formal-sap-spr-results.md)
- [Position-wise self-paced reading transfer validation](research/union_training/formal-sap-spr.md)
- [Timkey word-by-word eye-tracking results](research/union_training/formal-timkey-eye-results.md)
- [Timkey word-by-word eye-tracking transfer validation](research/union_training/formal-timkey-eye.md)
- [Jointly trained ensemble human-validated sealed snapshot](research/union_training/holdout-plan.md)
- [In Round 31, on materials with location markers observed by humans, was the location correctly identified?](research/union_training/localization-results.md)
- [Human observation and joint training specifications (first 12 groups)](research/union_training/observation-contracts-a.md)
- [Joint training human observation specifications (II)](research/union_training/observation-contracts-b.md)
- [Full Joint Training and L4 Resume Verification](research/union_training/observation-full-results.md)
- [Joint readout training on human observations](research/union_training/observation-protocol.md)
- [Observation Models for 24 Human Data Sources](research/union_training/observation-results.md)
- [Full joint training representation continuation](research/union_training/observation-resume.md)
- [Public human-observed joint training round one](research/union_training/protocol.md)
- [Report the joint training results of publicly observed human observations](research/union_training/results.md)
- [Subsequent training and formal held-out validation specifications](research/union_training/settling-and-heldout.md)
- [Full joint training fine-tuning and whole held-out validation](research/union_training/settling-results.md)

## Adaptive source weighting

- [Adaptive joint training on public human reading data](research/adaptive_joint/protocol.md)
- [Adaptive joint training results](research/adaptive_joint/results.md)
- [Fixed-weight control under retention constraints](research/adaptive_joint/retention-control.md)
- [Adaptive joint training with retention constraints](research/adaptive_joint/retention.md)

## Token-level readouts

- [Initial evaluation of the token-level reading readout](research/token_readout/design.md)
- [Initial token-level readout and Gradio validation](research/token_readout/results.md)
- [IIRC supervision for locating cues in the original text](research/token_readout/transfer-iirc-protocol.md)
- [Comparing token-level readouts across backbones](research/token_readout/transfer-protocol.md)
- [Token-level readouts across backbones](research/token_readout/transfer-results.md)

## Backbone comparisons

- [The weights and text forward pass of LFM3-1.7B-Base](research/backbone_transfer/lfm3-source.md)
- [CogLLM's three backbones single forward pass comparison](research/backbone_transfer/protocol.md)
- [A comparison of the three backbones on human reading data held-out](research/backbone_transfer/results.md)

## Gradio interfaces

- [CogLLM Gradio Interfaces](research/gradio_app/README.md)

## Interface verification records

- [Checkpoint](research/results/gradio-overview-regressions/checkpoint.md)
