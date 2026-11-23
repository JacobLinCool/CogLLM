# OneStop re-read state baseline

After completing the descriptive checks on the public data, we fixed the readout comparison below. We fit on batches 1–2 of the ordinary-reading group and held out the 10 articles and 60 readers in batch 3. Using the checked rules, we paired first and second readings of the same paragraph version by the same reader and excluded missing answers and truncations reported by the authors. Post-reading questions provide human correctness and group alignment for scoring. [Stimulus and control audit](onestop-rereading-audit.md)

- **Shared priors** are used to estimate a correct-response probability from paired responses across two rounds.
- **Re-read** with the same development split, estimating a hit probability for both first-read and re-read separately.

Probabilities are smoothed using ((x+0.5)/(n+1)), where (x) denotes the number of correct responses and (n) the number of trials. Both metrics serve as baselines for subsequent text models. This comparison quantifies the information gain provided by incorporating the reread state.

Scoring is restricted to held-out paragraph-item pairs that include both first-pass and second-pass responses. First, the Brier score is estimated for two rounds per group and averaged with equal weighting. second, the group-level averages for the same article are averaged with equal weighting. finally, the 10 articles are averaged with equal weighting. The Brier score represents the squared error between the probability of a correct answer and 0 or 1, ranging from 0–1, where lower values indicate better performance. The difference between the second-pass prior and the shared prior is reported per article, using 10,000 paired resamples with fixed readers and predictions, and a random seed of 20261121. Additionally, the magnitude of human second-pass gains is compared against second-pass prior gains using the same weighting scheme. The 95% confidence interval upper bound is checked to assess whether the second-pass prior improves the Brier score. This analysis serves as a public-data conditional baseline, noting the small number of articles and the variability of effect sizes across batches.

[Original research record](../../../research/stage6/onestop-rereading-baseline.md)
