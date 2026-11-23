# Kerwer English abstract reader background held-out results

Retain the English abstract scores from the current Gradio checkpoint. Across the original study's 166 readers, 12 source families, and 1,947 trials, complete 20 cross held-out grids. Each grid has no reader or DOI overlap in either fitting or testing. Raw scores and reader controls remain on the local machine. Colab uses the count of rating categories from each training grid and the no-rating held-out features to reproduce the likelihood of the sequential eight-level odds ratio. Fitting takes 3.0 seconds. L4 is released. [Specification of the pre-sealed snapshot](kerwer-reader-background-protocol.md) · [Fitting and release manifest](../../../research/results/kerwer-reader-background-l4/run-verification.json) · [Complete held-out scores](../../../research/results/kerwer-reader-background-l4/scores.json)

| Primary comparison | RPS improvement (positive values are better) | 95% readers × DOI matching interval |
| --- | ---: | ---: |
| Text readout with pre-read E‑PA | +0.00035 | [−0.00125, +0.00201] |
| Length, version, and E‑PA added to Gradio comprehension score | +0.00545 | [−0.00306, +0.01374] |
| Text with the main effect of E‑PA, including interaction terms. | +0.00020 | [−0.00035, +0.00072] |

The additive and interaction background readouts both met the prespecified noninferiority threshold for cumulative calibration. All three RPS improvement intervals crossed zero. For blank background, compute the marginal mean over the training cell's E‑PA distribution. Both blank-background readouts met the RPS and cumulative-calibration noninferiority margins against the text-only readout. The report archives the sensitivity analysis that leaves out each of the 12 DOIs in turn.

This result did not confirm that early PA improves the sequential rating distribution on unseen English abstracts and unseen readers. The experiment followed the original study's reader and abstract families validated by prior rewriting, with the evaluation metric being post-reading whole-document self-reported comprehension. This set of outputs is retained as a baseline for subsequent reader background comparisons. [Original Study Methods, Table 6](https://doi.org/10.1525/collabra.18898) · [Data Source and Dash Manifest](../../../research/results/kerwer-reader-background-preparation.json)

[Original research record](../../../research/stage6/kerwer-reader-background-results.md)
