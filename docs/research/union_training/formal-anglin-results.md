# Anglin's first-day belief shift results

**The Anglin transfer in CogLLM failed this acceptance.** Of the six comparisons, three aligned with the original research hypothesis direction, and zero models had their human difference intervals fully contained within the pre-specified ±0.5 tolerance. The maximum estimated change in the model was only 0.028 points, while the mean change across the 312 readers ranged from 0.279 to 1.043 points. [pre-sealed snapshot scoring rubric](formal-anglin-protocol.md) · [itemized scores](../../../research/results/union-observations-formal-l4/anglin/scores.json)

| Evidence condition | First-day comparison | human average change | CogLLM variation | Model reduction of the human 95% interval |
| --- | --- | ---: | ---: | ---: |
| Both times opposed. | T3 − T1 | −0.279 | +0.010 | [−0.137, +0.687] |
| Both support | T3 − T1 | +1.043 | +0.013 | [−1.418, −0.665] |
| First oppose, then support. | T2 − T1 | −0.292 | −0.011 | [−0.025, +0.600] |
| First oppose, then support. | T3 − T2 | +0.528 | +0.028 | [−0.805, −0.250] |
| First support, then oppose | T2 − T1 | +0.696 | −0.010 | [−0.985, −0.440] |
| First support, then oppose | T3 − T2 | −0.468 | +0.021 | [+0.262, +0.743] |

Eight processes each performed a forward pass on the NVIDIA L4, yielding 24 reading stops confirmed by independent prefix splitting. Additionally, 12 truncation or post-text replacement causal verifications were conducted, with a maximum relative root mean square error of 0.0000273, below the threshold of 0.001. For the 27,167 text items in the training and development packets, 2,278 Anglin n-grams (normalized consecutive 12-grams) showed no matches. Weights for round 31 were selected from the development set. Eight predictions were sealed snapshots taken by the scoring program before reading individual data. [Selected weights](../../../research/results/union-observations-settled-l4/seal.json) · [Predicted sealed snapshot](../../../research/results/union-observations-formal-l4/anglin/forecast/seal.json) · [Material overlap verification](../../../research/results/union-observations-formal-l4/anglin/training-text-overlap.json)

The present results apply to text transcription materials and fixed Rodger readout scales. It projects the policy validity scale onto Anglin's eight-point belief scale. This design cannot separately estimate the effects of representation, scale transformation, and chart presentation on error. There are distinct issues regarding initial certainty, scale, and reading stop. Qiao's data include a post-test 0–100 confidence field. however, the original item stems and anchors for this field are not provided in the open methods and materials, and pre-test and next-day measurements are missing. therefore, the present scoring range is limited to initial belief. [Anglin original study and scale](https://doi.org/10.1525/collabra.133914) · [Qiao original study](https://escholarship.org/uc/item/1zj2p06s) · [Predictions of eight-process sealed snapshot](../../../research/results/union-observations-formal-l4/anglin/forecast/predictions.json)

[Original research record](../../../research/union_training/formal-anglin-results.md)
