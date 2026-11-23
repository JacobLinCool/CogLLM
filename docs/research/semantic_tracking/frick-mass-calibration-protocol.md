# The first-order calibration of the location probability for the Frick contradiction report

Building on the Frick experiment, we utilize 1,800 text-only reading trajectories, 20 paired stories, and the existing Qwen3.5-2B causal end-state. The prior round of paired supervision estimated the condition difference for the target sentence's first report at +23.30 percentage points, with human at +48.27. Since the model placed too many first reports before the target sentence, this round simultaneously checks the full position distribution and the difference magnitude. These stories have already been used for prior development and error diagnosis. this round treats them as held-out material for the same experiment. [Existing Results](frick-pair-results.md) · [Stimuli and Human Observations](frick-consistency-source.md)

## Readout and selection modeling

For each of the 20 outer folds, one set of stories was held out, while the original seven features were retained. Within each fold, the pairing direction and feature scaling were recomputed using the training family of that fold. The inner folds were similarly split by story. The 19 outer training stories were ordered and sequentially cycled into four inner held-out folds. For each story, a logistic hazard was independently predicted for every reading stop. the cumulative product of the unreported probabilities yielded the first-report probability for each sentence and the final-unreported probability for the story. Predictions relied solely on the single source text and causal representations from the current or earlier time points. The target sentence position, conditions, and pairings were used exclusively for constructing the training direction and training/scoring endpoints.

The original sequential reading stop risk set negative log-likelihood maintains the same normalization (0.10\lVert\theta\rVert^2/2) regularization term. An additional four-class cross-entropy with weight (\lambda) is added. For each training story version, the first event is aggregated into pre-target sentence, target sentence, post-target sentence, and end-of-document no-report categories. The cross-entropy is computed based on the four-class proportions observed by the human in that version, then averaged with equal weighting across stories and versions. The four-class probabilities are drawn directly from the same set of hazard models. no additional four-class classifier is fitted.

\[
\mathcal{L}=\mathcal{L}_{\mathrm{hazard}}+
\frac{0.10}{2}\lVert\theta\rVert^2
-\lambda\;\operatorname{mean}_{s,c}\sum_{j=1}^{4}q_{scj}\log \widehat{p}_{scj}.
\]

Among fixed candidates (\lambda\in\{0,0.25,1,4\}), select the one with the minimum four-fold inner four-class cross-entropy. The candidate's inner full-text NLL must increase by at most 0.05 nats/document relative to (\lambda=0) on out-of-fold samples. Both inner scores are first averaged by version, then equal-weighted across all inner held-out stories. In case of a tie in cross-entropy, select the smaller coefficient. After selection, refit the 19 outer training stories and seal the snapshot of the held-out prediction for that fold.

## Acceptance

Primary scoring used outer prediction versus human first report. Retained the original fixed 10,000 story-reader cross-resampling, seed 735972, and 95% interval. The original seven-feature equal-weighting readout and matching effect readout both used sealed snapshot prediction. Also retained the initial prefix Qwen control. Through the requirement of simultaneously satisfying the following conditions.

- The condition difference for the first report of the target sentence was positive, and the model's effect relative to human was entirely contained within ±10 percentage points.
- The entire upper bound of the difference interval for the first report NLL relative to the seven-feature equal weighting and the original prefix Qwen was below +0.05 nats/piece. Additionally, the difference relative to the existing matching readout was reported for diagnostic purposes.
- In the consistent and inconsistent contexts, the model's human-aligned goal report difference decreased, with each interval fully contained within ±10 percentage points.

Report the full four-class distributions for both conditions and the coefficients for each fold selection. Human observations encompass the first sentence-by-sentence contradiction report elicited by prompting. Subsequent sentences' "consistent" response judgments pertain to the current sentence, without tracking whether the original contradiction has been resolved.

[Original research record](../../../research/semantic_tracking/frick-mass-calibration-protocol.md)
