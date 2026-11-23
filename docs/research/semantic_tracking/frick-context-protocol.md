# Frick's cross-sentence causal feature contrast

The original seven-feature readout significantly underestimated the difference in human first reports of the same sentence under consistent versus inconsistent contexts in Frick's target sentence. In this round, with the same frozen Qwen3.5-2B sealed snapshot sentence representations, the expanded readout captured lexical overlap and representational relationships between the current sentence and all previously read sentences. Both models employed the same nested split, loss, selection, and acceptance rules. The materials were used in prior development and error analysis, constituting a retrospective holdout. [Original condition likelihood](frick-conditional-protocol.md) · [Target sentence diagnosis](frick-target-hazard-diagnostic.md)

## The seven additional features computed from single-document readout

At the end of sentence $t$, let $W_t$ denote the normalized lemma set of the sentence. Follow the original readout Unicode NFKC, case-folding, and at least four-character lemma rules. Compute $n_t(w)$ as the count of sentences within the first $t-1$ containing lemma $w$. For each earlier sentence $j<t$, compute

\[
\begin{aligned}
\operatorname{idf}_t(w) &= 1+\log\frac{t}{1+n_t(w)},\\
a_{tj} &= \frac{\sum_{w\in W_t\cap W_j}\operatorname{idf}_t(w)}
{\sum_{w\in W_t}\operatorname{idf}_t(w)},\\
\pi_{tj} &=\frac{\exp(4 a_{tj})}{\sum_{i<t}\exp(4 a_{ti})},\qquad
u_{tj}=\frac{1}{t-1}.
\end{aligned}
\]

The overlap of the empty set is defined as zero. Let $v_t$ denote the normalized complete prefix suffix representation, $v_0=0$. Define $d_t=v_t-v_{t-1}$, then take $e_t=d_t/\lVert d_t\rVert$. The $e_t$ zero-difference vector is defined as zero. This adjacent difference uses the state of the earlier sentence, with the first sentence starting from zero. Let $\ell_{tj}=(t-j)/(t-1)$. The following items are all computed at the sentence end, with the range including all $j<t$.

| Feature | The formula |
| --- | --- |
| Maximum word form overlap | $\max_j a_{tj}$ |
| Average word form overlap | $\operatorname{mean}_j a_{tj}$ |
| Relative context distance of word-form weighting | $\sum_j(\pi_{tj}-u_{tj})\ell_{tj}$ |
| Maximum suffix update similarity | $\max_j e_t^\top e_j$ |
| Suffix-weighted update similarity difference | $\sum_j(\pi_{tj}-u_{tj}) e_t^\top e_j$ |
| Word-form weighted full-text state similarity difference | $\sum_j(\pi_{tj}-u_{tj}) v_t^\top v_j$ |
| When the magnitude of representation update before and after the sentence | $\lVert d_t\rVert$ |

All word count statistics and retrieval weights vary with the readout scope. These features describe text associations using observed lemmas and model representations. Semantic similarity is an internal representation metric of the model, while data observation is conducted sentence-by-sentence via human keystrokes. Each prediction reads only itself. the paired alternative, target sentence position, and human response are used solely for training and scoring.

## Shared training and validation

Two readout models fully utilize the conditional likelihood from the previous round, the sentence-wise risk set likelihood, the four-position cross-entropy, and 0.10 ridge. Each of the seven-feature and fourteen-feature configurations reconstructs all inner-layer training directions and feature scales, employing the same ${0,0.25,1,4}$ grid and selecting two loss weights via rule-based selection, followed by 20 outer-story held-out prediction scores. The seven-feature readout must reproduce the predictions of the previous sealed snapshot. Comparisons between the previous equal weighting and the original full prefix also utilize the sealed snapshot predictions. [Fixed loss, family held-out, and model selection rules](frick-conditional-protocol.md)

The primary validation adopts the target sentence effect direction, with equivalence to human magnitude of ±10 percentage points, and ±10 percentage points equivalence for early report bias under both conditions, as well as a whole-document NLL non-inferiority bound of +0.05 nats/document. Additionally, report the absolute effect error of the target sentence and whole-document NLL for the fourteen features relative to the seven features. To determine improvement from the added features, the reduction in the former must have a 95% confidence interval entirely positive, and the upper bound of the difference interval for the latter relative to the seven features must be below +0.05 nats/document. We employ 10,000 reader-story cross-samples with seed 735972. We also report the full distribution of first report positions under each prefix, along with feature and prefix prediction consistency for the two models.

[Original research record](../../../research/semantic_tracking/frick-context-protocol.md)
