# When only the main text is input, Qiao's two scoring timelines share the same prediction

In the single case of Qiao and Lagnado, after fixing the evidence order, the cumulative body text readings at the same temporal point are identical across the timelines. Targeting the actual sample mean from Study 1, any model that reads only the body text and shares a single deterministic readout across the three timelines would incur an error of at least **7.03 points** on the final order effect of at least one timeline. This represents a lower bound for the sample mean. Sample uncertainty still permits the two timeline effects of the population to be equal. [Results and Source Hash](../../../research/results/qiao-mode-identifiability.json)

## Verify the body text and human scores

The three schedules are item-by-item ratings in Study 1, ratings by group in Study 1, and item-by-item ratings followed by whole-group ratings in Study 2. Within each evidence order, all three versions have identical prefix strings, UTF-8 bytes, and character endpoints after paragraphs 4 and 8. At the end of paragraph 4, D→P has 2,239 characters and P→D has 2,323. Both reach character 2,937 after paragraph 8, and the full-text hashes differ by order. Version identifiers encode the experimental conditions. Content identifiers derived from prefix hashes and endpoints are shared across schedules. This check uses body text reconstructed from the authors' materials. The reconstructed text excludes screen layout, questions, and earlier responses. [Materials and alignment scope](qiao-evidence-source.md) · [Content identifiers](../../../research/results/qiao-mode-identifiability.json)

| Final assessment timeline | D→P, mean and reader-recorded counts | P→D, mean and reader-recorded counts | The sequential effect D→P is reduced compared to P→D, with a p-value and 95% confidence interval. |
| --- | ---: | ---: | ---: |
| Study 1 item-by-item evaluation | 67.33，43 | 48.29，38 | +19.04 [8.57, 29.52] |
| Study 1 per-group evaluation | 54.95，38 | 49.98，40 | +4.97 [−5.19, 15.24] |
| Study 2: Item-by-item followed by grouping | 60.67，45 | 55.35，48 | +5.31 [−4.10, 14.78] |

The score represents a 0–100 scale for the final probability of guilt. The mean retains all final scores from the author analysis. Subtracting the final model score for the P→D passage from that of the D→P passage within the same passage yields the model's ordering effect $d$. For the same input, $d$ remains identical across three time points. In Study 1, the two human effects differ by $19.0361-4.9724=14.0637$ points, thus

\[
\max\{|d-19.0361|,\,|d-4.9724|\}
\;\ge\; \tfrac12|19.0361-4.9724|=7.0319.
\]

Taking the midpoint of two human effect estimates $d=12.0042$ yields this boundary. The effect in Study 2 lies between the two, and the boundary remains identical when this sample is added. If pre-known rating schedules or earlier reader ratings are included as model inputs, the same text can have different input states, rendering the premise of a shared score invalid. [Full point estimation and minimax solution](../../../research/results/qiao-mode-identifiability.json)

## Sample uncertainty

Resample 20,000 times with seed 735972. In each resample, resample the entire possible same-account group within each study and recalculate the mean based on the records of each group in that resample. Study 1's 159 final ratings are partitioned into 157 groups. One group spans two time points, so each resample applies the same weight to the two records of that group. Study 2's 93 ratings are each in their own group and resampled independently. The time point effect difference in Study 1 is +14.06 points, with a 95% percentile interval of **[−0.73, +28.74]**. Taking this signed interval first and then mapping it to half the absolute difference yields **[0, 14.37] points**. In this sample, there is no positive population lower bound. [Group resampling setup and interval](../../../research/results/qiao-mode-identifiability.json)

Remove the last four scores from two potentially overlapping account groups combined, with a sample boundary of 6.67 points. Remove the last score that cannot be verified from a single text trajectory alone, with a boundary of 6.70 points. These two checks use the same case. the values in this paper do not estimate the effect of new cases, nor do they include word-by-word uncertainty or resolution endpoints. [Sensitivity and Scope](../../../research/results/qiao-mode-identifiability.json) · [Original Study](https://escholarship.org/uc/item/1zj2p06s)

[Original research record](../../../research/semantic_tracking/qiao-mode-identifiability.md)
