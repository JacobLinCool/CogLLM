# Frick readout at the end of Currie et al

Does training a frozen readout on sentence-by-sentence judgments of English stories enable estimation of the probability of post-reading "incoherence" on English children's reading data, as well as the report difference between narrative and expository genres? This design treats the model probability of "previously reported" as the end-of-document answer probability, serving as the cross-task hypothesis to be tested. Currie et al. report that sixth-grade readers are more likely to correctly identify contradictions in narrative texts under both conflicting distance conditions. The genre difference is defined within this corpus. [Original study Figure 1c, Measures and procedure](https://doi.org/10.1080/10888438.2020.1831503) · [Author profile and alignment](larrc-consistency-source.md)

## Frozen models and held-out ranges

All 24 contradictory topics and two distance variants, totaling 48 items, were fully held-out. The previously published group results and denominators have been verified by the source family. Our model and comparisons were fixed prior to generating any scores for this batch of texts. Since 12 consistent texts originate from separately written materials and the public archive lacks independently verifiable item IDs, the trial-by-trial validation was restricted to near and far contradictory variants. With 12 groups each for the two styles, all were included in the primary comparison. [Data Mapping and Constraints](larrc-consistency-source.md)

Each input consists of eight sentences, processed through a single causal forward pass using the same Gradio candidate Qwen3.5-2B backbone. Predictions are restricted to the original text and the end position of each sentence. The experiment style field, contradictory reading stop, distance version, alternative version text, reader data, and answer are provided solely for scoring. All 384 sentence ends must pass the source coordinate and tokenizer prefix check. The first sentence provides only the prefix representation, while the subsequent seven sentences generate hazards according to the original Frick readout definition. [Original Frick model input and training](frick-pair-protocol.md) · [backbone candidate](../../../research/results/gradio-candidate-checkpoint/checkpoint.json)

For the Frick three groups, each had 20-fold sealed snapshot first-report readouts. All 20 directions and coefficients per group were frozen. For each paper, using the same forward Qwen state, the first-event hazard \(h_{f,t}\) was computed sequentially for each fold at the seven reading stops. The end-of-paper prediction was

$$
$p(\text{incoherent}) = \frac{1}{20} \sum_{f=1}^{20}$
\left(1-\prod_{t=2}^{8}(1-h_{f,t})\right).
$$

We average event probabilities across folds with equal weights. The same rule applies to the models trained with uniform observation weighting, target-sentence weighting, and the paired-effect objective. The paired-effect model is the evaluation candidate. The uniform and target-weighted variants are controls using the same features. A constant baseline uses the proportion of Frick story trials with at least one human inconsistency report, \(754/1800\). No model is fitted or recalibrated on Currie data. The prediction step reads only held-out stimuli, backbone representations, and Frick weights. It accesses the held-out answers only after those predictions are sealed. [Denominators from the original trajectories](../../../research/results/frick-pair/results.json) · [Sealed forecasts for the three readouts](../../../research/results/frick-pair/forecast-seal.json)

## effect and tolerance

In contradictory texts, `Question.ACC=1` represents the post-answer article being incoherent. The primary endpoint is the NLL of the final binary report, and the narrative report rate minus the expository report rate. For each version, first average across available readers, then apply equal weighting to near and far versions of the same topic, and finally apply equal weighting across genres for each topic. The overall NLL averages across the two genres. The NLL unit is nats/trial, with a minimum of zero. The matching effect readout must satisfy all of the following criteria.

- The model's narrative-to-explanatory report ratio is poor, with a 95% confidence interval lower bound greater than zero.
- The model's effect difference relative to human is smaller, with its 95% confidence interval falling entirely within ±10 percentage points.
- Relative to all observations, equal weighting, and relative to a constant, the two NLL differences have 95% confidence interval upper bounds both below +0.02 nats/trial.

In the 20 near-far pairs with eight complete sentences each, we describe the difference between human and model reports for far-to-near transitions, estimate these differences separately for each style, and average them using equal weighting. Within the corpus, only sentence-order differences are retained for the main analysis. materials with lexical or punctuation differences in the four groups are preserved. The two effect types and their condition proportions are estimated using 10,000 cross-validation bootstrap iterations with seed 735972. In each iteration, sentences are resampled within the corpus, retaining both near and far versions in pairs. Readers are sampled jointly across the entire corpus, assigning equal sampling weights to all occurrences of a given sentence. Related comparisons use the same resampling scheme. The tolerance threshold extends Frick's magnitude scale, with the NLL upper bound set as a fixed criterion at the end of this section. [Original Frick tolerance](frick-pair-protocol.md#comparison-and-scope-of-application)

End-of-text consistency answers support whole-text estimation across materials. Even when end-of-text probabilities pass acceptance, humans may first detect at a later point or alter judgments before completion. Currie's answers are recorded only after the final sentence. therefore, in-sentence highlighting, the first report position, and the timing of resolution must correspond to human observations. [Original study Measures and procedure](https://doi.org/10.1080/10888438.2020.1831503)

[Original research record](../../../research/semantic_tracking/larrc-consistency-protocol.md)
