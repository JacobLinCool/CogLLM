# SAP word-level self-paced reading results

On Huang SAP's disambiguation words, CogLLM's direction of the disambiguation effect aligns with human at **1/3** of the constructions. Including the two subsequent words pre-specified, the direction aligns with human in **6/9** of the constructions. All nine human control intervals exceed zero. The model predicts the opposite direction for NP/S disambiguation words, as well as NP/Z disambiguation words and the following word. [Formal Results](../../../research/results/union-observations-formal-l4/sap/scores.json) · [Pre-specified validation specifications](formal-sap-spr.md)

| Construction | Relative position | human log time difference [95% interval] | The model's predictions are poor. |
| --- | --- | ---: | ---: |
| MV/RR | disambiguation term | +0.0522［+0.0280, +0.0784］ | +0.0185 |
| MV/RR | The next word | +0.2532［+0.2278, +0.2798］ | +0.0276 |
| MV/RR | the second word | +0.1670［+0.1422, +0.1921］ | +0.0266 |
| NP/S | disambiguation term | +0.0512［+0.0317, +0.0731］ | −0.0013 |
| NP/S | The next word | +0.0859［+0.0587, +0.1131］ | +0.0029 |
| NP/S | the second word | +0.0355［+0.0144, +0.0555］ | +0.0085 |
| NP/Z | disambiguation term | +0.1289［+0.1097, +0.1475］ | −0.0059 |
| NP/Z | The next word | +0.1868［+0.1599, +0.2140］ | −0.0019 |
| NP/Z | the second word | +0.0824［+0.0634, +0.1004］ | +0.0053 |

human difference was calculated as the same-material ambiguous version minus the control version, averaged by material using equal weighting. Scoring was based on 143,120 valid position reads. Following NP/Z, the second word used 23 materials due to a single surface word difference across one group, while all other conditions used 24 materials. Intervals reflect material sampling, with conditions at observed readers. The model column adopted the log time scale from Paape's developed data, with formal comparisons using direction. [Position-by-position results](../../../research/results/union-observations-formal-l4/sap/scores.json) · [Huang original study](https://tallinzen.net/media/papers/huang_et_al_2024_jml.pdf)

Freeze the joint state at epoch 31, perform a forward pass on each of the 360 inputs in L4 to obtain 1,078 verified prefix reading stops, with batched forward passes taking 8.14 seconds. Readouts are fitted to 628 development positions in Paape, covering 31,933 bidirectional segment readtimes. After predicting 432 SAP sealed snapshots, these are merged into the formal readtime. Compared against 27,167 joint development texts and 216 calibration texts, no SAP full-text or consecutive 12-word overlap was found. [Freeze model](../../../research/results/union-observations-settled-l4/seal.json) · [Predict sealed snapshot](../../../research/results/union-observations-formal-l4/sap/forecast/seal.json) · [Material audit](../../../research/results/union-observations-formal-l4/sap/training-text-overlap.json)

Results support only partial cross-data transfer of construction and position. Paape's calibration covers key look back and some single-pass multi-word reading times. Huang's validation targets unidirectional word-by-word reading times. Here, we can determine whether the frozen-state replication of the ambiguity effect's location and direction generalizes to new data, but we cannot distinguish whether differences stem from shared states or measurement transfer. Eye-tracking and sentence-final comprehension must be measured via their respective response metrics. [Calibration and Validation Scope](formal-sap-spr.md)

[Original research record](../../../research/union_training/formal-sap-spr-results.md)
