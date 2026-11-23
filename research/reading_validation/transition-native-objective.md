# A shared objective for observed reference failures

A word can trigger distinct observable responses. A reader may report an unexplained character introduction, a missing event reference, or an unclear pronoun. A separate reading game records which antecedent a player chose. Each outcome supervises a shared causal representation while retaining its measured unit. [SNaC taxonomy](https://aclanthology.org/2022.emnlp-main.29/) · [Phrase Detectives task](https://aclanthology.org/N19-1176/)

## Measurement model

Let $h_t$ concatenate the two native hidden vectors at the final subtoken of word $t$. Normalize each layer at that position, center and scale its dimensions using the training articles, and scale each layer block by $\sqrt{48/d}$ for native width $d$. Let $u_t$ contain the causal surface controls. Observation $k$ has semantic weights $w_k$, control weights $v_k$ and an intercept $b_k$. Its probability before any source-domain intercept calibration is

$$
p_{tk}=\sigma(b_k+h_t^\top w_k+u_t^\top v_k).
$$

For SNaC, $y_{tk}=1$ when a recorded human-marked span of that type overlaps the word. In the reading game, let $n_e$ denote the number of distinct eligible players who chose entity $e$ after collapsing textual mentions of the same entity. For $n=\sum_e n_e$, the observed response rate is

$$
y_{t,\mathrm{game}}=1-\frac{\sum_e n_e(n_e-1)}{n(n-1)}.
$$

The estimand is disagreement between two different readers, conditional on both selecting a previous entity. A 0.2 prediction means an estimated 20% chance of different antecedent choices for such a pair. [Interpretation and quality checks](transition-game-supervision-protocol.md)

## Group-balanced learning

For task $k$, let $G_k$ be its observed source groups, and $I_{kg}$ the words with an identifiable outcome in group $g$. Source groups are book titles or reading-game articles and works. With $\ell(y,p)=-y\log p-(1-y)\log(1-p)$, minimize

$$
L=\frac{1}{K}\sum_{k=1}^K\frac{1}{|G_k|}
       \sum_{g\in G_k}\frac{1}{|I_{kg}|}
       \sum_{t\in I_{kg}}\ell(y_{tk},p_{tk}).
$$

A repeated pronoun string that cannot be assigned a unique SNaC span removes that segment from the corresponding task loss. Other observed outcomes at the same positions remain eligible. A game mention contributes one fractional response, avoiding pseudo-replication from pairs that share individual players.

For $W=[w_1,\ldots,w_K]$ and $V=[v_1,\ldots,v_K]$, the independent observer adds $\lambda(\|V\|_F^2+\|W\|_F^2)/(2K)$ to $L$. The shared observer adds

$$
R_{\mathrm{shared}} =
\frac{\lambda}{2K}\|V\|_F^2+
\frac{\lambda}{4K}\left(\|W\|_F^2+\|W\|_*^2\right),
\qquad \lambda=0.01.
$$

The nuclear norm is the sum of singular values. Penalizing its square favors semantic directions useful to more than one task, including directions with opposite signs on different response types. The control weights remain task specific. With one task both penalties agree. Each source fit uses full-batch L-BFGS with a line search and must reach a maximum absolute gradient of $10^{-6}$ or smaller. The implementation adds $10^{-12}I$ to the small task-by-task Gram matrix before computing its square-root eigenvalues. [Single- and three-target implementation](transition_direct_l4.py) · [Four-target training](transition_game_l4.py)

## Holdout interpretation

For a forecast $p$ and measured pair-disagreement fraction $y$, the expected Brier loss on an actual pair is $y(1-p)^2+(1-y)p^2$. The squared difference $(p-y)^2$ omits disagreement among readers at that item. Within-article ranking tests whether the more ambiguous of two pronouns receives the higher score. Each comparison uses matched articles, with document-cluster uncertainty and a separate check holding the literal pronoun form fixed. [Evaluation implementation](transition_game_score.py)
