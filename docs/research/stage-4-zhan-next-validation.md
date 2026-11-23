# Stage 4 · Held-out validation of the reference readout

## At which reading stop does Stage 4 error occur?

On held-out stimuli from Zhan Experiment 1, the weakly supervised readout already underestimated the difference between the two stimulus types before the sentence-initial pronoun appeared. Within each verb type, give stimuli equal weight, then subtract the IC-2 probability of choosing the previous sentence's subject from that probability for IC-1. This diagnostic was defined after reviewing the first results. The model probabilities average the held-out predictions over 12 samples. [Held-out probabilities by stimulus](../../research/results/stage4-zhan-weak/predictions.jsonl) · [Decomposition](../../research/results/stage4-zhan-weak/posthoc-condition-diagnostic.json)

| reading stop | human two-class gap | Weak-supervised readout gap between two categories |
| --- | ---: | ---: |
| The preceding sentence number | 67.29 percentage points | 15.90 percentage points |
| The next pronoun | 42.83 percentage points | 14.78 percentage points |
| The reduction in the gap between the two reading stops. | 24.46 percentage points | 1.12 percentage points |

The five held-out folds' weak-supervised readout all underestimated the reduction in gap. Since each fold contains at least one set of a specific verb material, the merged held-out results are used as the primary basis. This decomposition locates the error in the magnitude of the two reading stops. The roles of the readout architecture, representation, and material distribution still require further comparison. [Number of materials per fold and probabilities](../../research/results/stage4-zhan-weak/posthoc-condition-diagnostic.json)

## The next readout experiment

Using the same causal forward two normalized states, denote the pre-intervention as $h_0$ and the post-pronoun read as $h_1$. The prior reading stop still outputs $b+w^\top h_0$. At the subsequent reading stop, compare the original $b+w^\top h_1$ with $b+w^\top h_1+v^\top(h_1-h_0)$ augmented by an independent weight $v$. This allows checking whether the readout can separately estimate the prior contextual bias and the change after pronoun reading. The additive term is obtained at the two states only at the reading stop following the pronoun. It can be retrieved from the existing L4 cache.

Freeze Qwen and fix the five folds, label sampling, effect definition, and loss weights. Use the same mean loss and per-parameter L2 penalty for the held-out comparison. Initialize both heads at zero. Compare against both full trial-level supervision and the same smaller set of labels with weak supervision. The primary outcomes are trial-level Brier score and error in the stimulus interaction effect. The difference between the two groups before the intervention serves as a diagnostic. This comparison uses development data from the same study. After locking the model and endpoints, score the entire external stimulus set once. [Fixed Stage 4 endpoints and splits](stage-4-zhan-weak-supervision.md)

## Accessible held-out materials

| candidate | Verification results | Purpose |
| --- | --- | --- |
| Zhan et al., Experiments 2 and 3 | Reader IDs do not overlap with Experiment 1. Of the 72 prompts from Experiment 1, Experiment 2 contains 70 identical active prompts, and Experiment 3 contains 68 identical canonical-word-order prompts. Prompts with a different verb retain the original characters. | Contrast the cross-reading, passive, and active constructions of the study. Each material was held out from the same source family as its own experiment. [Original Stimulus](https://osf.io/download/q7bwn/) · [Original Study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237012#sec009) |
| Lam & Hwang (2025), Experiment 1 | The public OSF reaction table includes 1,702 published entries covering verb bias, prime type, subject, and referent. The CC BY 4.0 publisher's Appendix A contains 24 sets of implicit causality and 24 sets of transfer of ownership original stimuli. The implicit causality condition adds an explicit pronoun after "because". | Specify the implied causal group as the first external IC × prompt type control. With the model and scoring rules fixed, report the difference in the two-prompt subject re-mention rate versus the material equal weighting condition. [Paper](https://doi.org/10.1080/23273798.2024.2427073) · [Open Response Table](https://osf.io/download/ak25j/) · [Original Material Appendix A](https://doi.org/10.6084/m9.figshare.27766995.v1) |
| Lam & Hwang (2022) and Ng et al. (2025) | We publicly released the original stimulus files and response tables. It was confirmed that the main clause was identical, with free prompts ending in "so," and pronoun prompts including "she did" or the omitted pronoun "did." | Form a candidate research family. Ng et al.'s data can be compared to explicit and null pronoun subject selection, with human responses prompted to stop at "just". Additionally, verify cross-study material keys and reused information. [Free continuation materials and data](https://osf.io/sehn4/) · [Pronoun stimuli and data](https://osf.io/hgbtr/) · [Research methods](https://doi.org/10.1080/0163853X.2025.2485872) |

The set of preliminary examples reported by Lam & Hwang (2025) also appears in the materials of Lam & Hwang (2022) and Ng et al. (2025). Prior to reconciling the complete list, the two candidates should not be regarded as independent validations. [Preliminary Examples](https://www.cuny2021.io/wp-content/uploads/2021/02/CUNY_2021_abstract_126.pdf) · [Continuation Stimuli](https://osf.io/download/y7uc3/) · [Pronoun Stimuli](https://osf.io/download/fy8za/)

The publisher appendix and experimental response table for Lam & Hwang (2025) have been downloaded to the local data cache, with SHA-256 hashes `ecafa119adba221af01b413cf7f4edb722c09b14737efd478d8d502f789ef89a` and `d308975fcecdc4f8ae3b39ef9cc4febb9a3424d0031ca0d8cfe2987ee12f1557` respectively. The CSV source does not specify project-level authorization. raw responses are stored locally. Data validation employed materials, prompt fields, and column counts, without applying external research human-selected readout or scoring thresholds. This round's reading stop corresponds to reference selection. Questions arising, pointing to earlier text and subsequent resolution still require energy measurement of those responses' existing studies.

All 24 implicit-causality items have free-continuation and pronoun conditions. Match the appendix against `Item_ID` and `CN_Verb` for `Experiment == LamHwangLCPE1` in the authors' `simple_connectives` worksheet. Appendix items 1–12 correspond to the odd-numbered `I01–I23`, and items 13–24 to the even-numbered `I02–I24`. After removing final connectives and punctuation, no main clause is word-for-word identical to a Zhan Experiment 1 main clause. The free prompt ends at the Mandarin connective for “, because.” The original prompts are still needed to verify which pronoun character each item uses. The publisher's appendix reproduces only the full free prompts. If the characters cannot be verified, report sensitivity separately for the Mandarin written forms of “he” and “she” and include the corresponding range for the external effect estimate. [Author material keys](https://osf.io/download/e5z6h/) · [Publisher appendix](https://doi.org/10.6084/m9.figshare.27766995.v1)

## Empirical evaluation with the same prompt and new readers

Match the active and canonical-word-order prompts from Experiments 2 and 3 verbatim to Experiment 1. Use predictions from the fold that held out the original stimulus family. Keep all Stage 4 weights fixed. Readers in Experiments 2 and 3 received free-continuation instructions. Readers in Experiment 1 were asked to continue with an explanation. Use the Stage 4 Brier and interaction scores, averaging scores across samples. [Experiment 2 data](https://osf.io/download/z48qy/) · [Experiment 3 data](https://osf.io/download/yajhg/) · [Scoring program](../../research/stage4/zhan_exact_match.py)

| Data | Valid responses | human interaction effect | A small number of direct | Weakly supervised |
| --- | ---: | ---: | ---: | ---: |
| Experiment 2 · Identical active sentences | 1,117 | +12.94 percentage points | −4.60 percentage points | +0.63 percentage points |
| Experiment 3 · Identical canonical-word-order sentences | 1,283 | +5.07 percentage points | −4.43 percentage points | +0.86 percentage points |

In both samples, paired bootstrap 95% intervals crossed zero for the weak-supervision minus limited direct-supervision differences in effect error and Brier score. The comparison across readers and instructions leaves the benefit unresolved. The inputs were identical, so this diagnostic leaves the Mandarin bei-passive and ba constructions untested. [Replay and denominators](../../research/results/stage4-zhan-weak/exact-repeat-new-readers.json)

[Original research record](../../research/stage-4-zhan-next-validation.md)
