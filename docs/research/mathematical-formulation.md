# CogLLM Mathematical Specification

CogLLM aims to reproduce the observable responses of a specified reader population using existing data. For each article, a single causal forward pass through Qwen3.5-2B gives the readout heads the representations needed to estimate token-level response probabilities, the spans that responses refer to, and intensity where a corresponding measure exists. The analysis software compares versions of the same material and validates comparable endpoints on entire sets of held-out observations. This document specifies the token-level design and records the implementation scope of round 31 of the joint readout. [Research protocol](training-protocol.md) · [Round 31 localization results](union_training/localization-results.md)

<a name="1-輸入讀者與座標"></a>

## 1. Inputs, readers, and coordinates

Let $X$ denote the full text, $x_{1:T}=\operatorname{Tok}(X)$ its content token sequence, and $I_t=[b_t,e_t)$ the original character interval for token $t$. The optional reader background $c$ uses only information available before reading. Let $C(c)$ be the prefix containing fixed start markers and delimiters, $P=|C(c)|$, and $j_t=P+t$. The full input is

$$
z_{1:N}=C(c)\mathbin{\Vert}x_{1:T},\qquad N=P+T,\qquad j_0=P.
$$

We tokenize the background and body separately before concatenation to preserve alignment with offsets in the body. $C(\varnothing)$ contains only the fixed markers, so the first body token has a preceding position. We record the observed stopping points in the text actually presented by each study. If a stop falls inside a token, retain the interval or encode the exact string visible for that trial. This prevents the representation for a stop from incorporating unseen characters.

For each measure $m$, we fix the reporting instructions, visible text, and rules for looking back. They define the observation conditions for the corresponding output head. The general interface takes the text and optional background. Study prompts, version labels, answers, and human reports stay in the research workflow. For $c=\varnothing$, we calibrate response probabilities to a specified reference reader population $P_{\mathrm{ref}}$. For an event $Y$, the target is

$$
p^*_{m,t,k}(X,\varnothing)
=\mathbb E_{C\sim P_{\mathrm{ref}}}
  \big[\Pr_m(Y_{t,k}=1\mid x_{\le t},C)\big].
$$

The implementation can train the empty-background head directly on existing observations from that population. If a trial has known background and comes from the reference population, training can mask $c$ to supervise the empty-background output. If the distribution of participants with recorded backgrounds differs from that of the reference population, calibrate with comparable existing observations or known weights. An empty background denotes missing individual information.

<a name="2-一次因果前向運算"></a>

## 2. One causal forward pass

Let $h^{(\ell)}_i\in\mathbb R^D$ denote the output representation of block $\ell$ before the final RMSNorm, with $D=2048$. Let $A$ denote the captured full-attention weights and $G$ the selected Gated DeltaNet signals. One pass gives

$$
\big(H^{(1)},\ldots,H^{(24)},A,G\big)
=F_\theta(z_{1:N}),\qquad
H^{(\ell)}\in\mathbb R^{N\times D}.
$$

The readout retains only the selected layers and specified statistics. $h^{(\ell)}_{j_t}$ depends on $z_{\le j_t}$. Qwen's attention mask prevents position $i$ from using keys at $j>i$. Gated DeltaNet convolution and state updates also use only the current and earlier inputs. Position-wise normalization and feed-forward operations use each position independently. Applying these properties layer by layer makes every block output prefix-dependent. The body passes through the backbone once. The model computes the readout without feeding back earlier response labels or running a text-generation loop. [Qwen3.5-2B architecture](https://huggingface.co/Qwen/Qwen3.5-2B#model-overview) · [Qwen implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

In round 31 of the joint readout, the backbone processed all 2,598,224 input tokens and saved final-layer vectors for 164,220 observation stops. Frick used the end of each presented sentence. PLABA's retrospective sentence ratings used the target sentence end and the article end. CompLex and the span-selection data also specified target word ends. The forward call set both `output_attentions` and `output_hidden_states` to `False`. Each source's observation stops determine which human responses can be aligned. A token-level readout should have a feature and output grid independent of each source's annotation grid. [Forward pass and extraction](../../research/backbone_transfer/extract.py#L295) · [Joint run verification](../../research/results/union-observations-settled-l4/run-verification.json) · [Current localization results](union_training/localization-results.md)

### Full attention

The Qwen3.5-2B full-attention blocks are $\mathcal L_A=\{4,8,12,16,20,24\}$. Each layer has 8 query heads, 2 key/value heads, and head dimension $d_h=256$. Let $\widetilde q^{\ell,a}_i,\widetilde k^{\ell,b}_j$ be the query and key after head RMSNorm and RoPE. Let $\rho(a)=\lfloor(a-1)/4\rfloor+1$ map a query head to its key/value head. For a single unpadded article,

$$
d_{ij}^{\ell,a}=\frac{(\widetilde q_i^{\ell,a})^\top\widetilde k_j^{\ell,\rho(a)}}{\sqrt{d_h}},
\qquad
A_{ij}^{\ell,a}
=
\begin{cases}
\exp\!\left(d_{ij}^{\ell,a}-
\operatorname{LSE}_{u\le i}d_{iu}^{\ell,a}\right), & j\le i,\\
0, & j>i.
\end{cases}
$$

$d_{ij}^{\ell,a}$ is the attention score before softmax. Qwen applies an output gate to the weighted values. The features below use the weights after softmax. Let $r=[s,e]$ denote a contiguous span of body tokens. Let $\alpha^{\ell,a}_{t,r}$ be the attention mass from position $t$ to span $r$ already read, and let $E^{\ell,a}_t$ be the query's attention entropy.

$$
\alpha^{\ell,a}_{t,r}=\sum_{k=s}^{e} A_{j_t,j_k}^{\ell,a},
\qquad E^{\ell,a}_t=-\sum_{u\le j_t} A_{j_t,u}^{\ell,a}\log A_{j_t,u}^{\ell,a},
\qquad e\le t .
$$

The denominator covers the entire visible prefix, including the background and fixed markers. The features can include attention mass on the latest $w$ body tokens or other prespecified summaries. The text read so far determines the positions and spans. If the backend's fused operation returns only weighted values, capture or reconstruct the selected weights from the query and key vectors of the same pass. [Full-attention computation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L646-L736)

### Gated DeltaNet

For the other 18 layers, use hidden states or gates from the same forward pass. Omitting layer and head indices, the core state update for each head uses key vector $k_i$, value vector $v_i$, retention factor $\gamma_i$, and write coefficient $\beta_i$.

$$
S_i=\gamma_i S_{i-1}(I-\beta_i k_i k_i^\top)+\beta_i v_i k_i^\top .
$$

Potential features include summaries across heads of $\gamma_i$, $\beta_i$, or $\|S_i-S_{i-1}\|_F$. These features describe the model's internal state. Claims about when human readers set aside or revive a question require independent measurements. [Gated Delta Rule, Eq. 10](https://arxiv.org/html/2412.06464v3#S3.SS1)

### Vocabulary logits and intermediate-layer projections

Let $U\in\mathbb R^{|\mathcal V|\times D}$ be the vocabulary projection matrix and $\mathcal N$ Qwen's final RMSNorm. For selected intermediate layers and the final layer, compute

$$
\ell_i^{(\ell)}=U\mathcal N(h_i^{(\ell)}),\qquad
\pi_i^{(\ell)}=\operatorname{softmax}_{\mathcal V}(\ell_i^{(\ell)}).
$$

$\ell^{(24)}$ gives the language model's next-token logits. The intermediate $\ell^{(\ell)}$ provides an additional *logit lens* readout. If the backend returns the final normalized representation, apply $U$ directly. An affine translator estimated on development data can provide a *tuned lens*. The vocabulary has 248,320 indices. The log-sum-exp for exact probabilities covers the entire vocabulary. [Qwen vocabulary head](https://huggingface.co/Qwen/Qwen3.5-2B#model-overview) · [Logit lens and tuned lens, §2–3](https://arxiv.org/html/2303.08112#S2)

Before $x_t$ is read, its surprisal comes from the language model at $j_t-1$. After $x_t$ is read, position $j_t$ can provide the entropy of the distribution over the next token and the difference between layers' distributions at that position.

$$
\begin{aligned}
\operatorname{Surp}^{(\ell)}_t
  &=\operatorname{LSE}_{v\in\mathcal V}\ell_{j_t-1,v}^{(\ell)}
    -\ell_{j_t-1,x_t}^{(\ell)},\\
\operatorname{Ent}^{(\ell)}_t&=-\sum_v\pi_{j_t,v}^{(\ell)}\log\pi_{j_t,v}^{(\ell)},\\
\operatorname{Div}^{(\ell)}_t
  &=D_{\mathrm{KL}}\!\left(\pi^{(24)}_{j_t}\,\|\,\pi^{(\ell)}_{j_t}\right).
\end{aligned}
$$

Although the pass receives the whole article, the features at $t$ use only $x_{\le t}$. The value $\ell_{j_t,x_{t+1}}$ requires the identity of the next unseen token, which makes it unavailable as an input at $t$. The implementation can compute attention and vocabulary statistics in blocks and reduce them immediately. Storing all six layers of attention requires $48N^2$ values. The vocabulary logits require $N|\mathcal V|$ values per layer.

<a name="3-平行讀出反應位置與強度"></a>

## 3. Parallel readout of responses, locations, and intensity

Let $\phi_t=\Phi(h_{j_t}^{(\ell)},E_t,G_t,\operatorname{Surp}_t,\operatorname{Ent}_t,\operatorname{Div}_t)$ be the feature vector at position $t$. $\Phi$ uses only features available after reading through $t$, including selected attention statistics. The first version uses prespecified subsets of layer-wise hidden states, attention features, and vocabulary features as candidates. We use development data to estimate readout weights, standardize features, and set thresholds.

<a name="每個-query-對已讀位置的關係"></a>

### Each query's relationship to positions already read

For each body token, obtain its source coordinates and its vectors from the final layer and selected intermediate layers. Retain the per-head relations $\{A^{\ell,a}_{j_t,i}\mid i\le j_t\}$ from the six full-attention blocks. The readout can access both the relevant attention weight and the content representation for any token or span already read. Compute the weights in blocks or reconstruct them from queries and keys retained in the same pass. In both cases, the normalization denominator covers the entire visible prefix. From the 18 Gated DeltaNet blocks, retain outputs, gates, or state-update signals for selected layers at every position. Keeping actual query-to-position relationships lets us test whether the model distinguishes responses to the same target word in different preceding contexts. [Pinned Qwen configuration](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json#L269) · [Backbone and prefix computation](#2-一次因果前向運算)

The existing Frick experiment captured signals from the last full-attention block at every token, aggregated them across heads, and then aggregated them by sentence. Adding these summaries improved the article-level first-report negative log-likelihood (NLL). The interval for improvement in the condition difference at the target sentence still crossed zero. PLABA's six-layer summaries also aggregated features at sentence-end queries. The proposed readout keeps the per-head and per-position relationships for future validation. [Frick attention-summary experiment](semantic_tracking/frick-signals-results.md) · [PLABA attention-summary experiment](stage-3-feature-ablation.md)

Validation compares a readout using only token vectors with one that also uses native attention relationships, on the same data and causal prefixes. Use the same training-set reporting base rates and check convergence separately for both readouts. This helps distinguish a gain from preceding-text information from an improvement in probability calibration. Compute the Frick likelihood at the original sentence-end reporting points, where it may aggregate states across the sentence. Validate word-level triggers, the preceding spans they refer to, and resolution times against separately aligned human observations or experimental manipulations. All positions first undergo checks with truncation, suffix replacement, and padding under shared-prefix conditions. [First-report readout diagnostics](union_training/localization-results.md#frick-readout-the-fit-at-overall-early-stopping)

### Responses and their referenced spans

Define $K$ response types using the human reporting rules. Examples include an unresolved reference, a perceived break in coherence between sentences, and successful identification of an earlier referent. Response types may co-occur. Let $Y_{t,k}=1$ mean that a reader reports type $k$ at stop $t$ under those rules. Let $Z_{t,r,k}=1$ mean that the report explicitly points to span $r$ already read. $Z_{t,r,k}=1$ implies $Y_{t,k}=1$. The index $t$ records where the response was observed and $r$ records which text it refers to. An experiment may separately annotate the manipulated trigger. It can differ from the span a participant identifies.

For maximum candidate length $B$, use $\mathcal R_t=\{[s,e]:1\le s\le e\le t,\ e-s+1\le B\}$. Candidate filtering, if used, follows the same prefix constraint. The span representation is $u_r=[h^{(24)}_{j_s};h^{(24)}_{j_e};(e-s+1)^{-1}\sum_{v=s}^{e}h^{(24)}_{j_v}]$. We can compute $u_r$ using the endpoints and prefix sums. In the readout below, $\alpha_{t,r}$ concatenates attention mass from selected layers and heads, $d(t,r)$ encodes distance within the text already read, and $D_p$ is the dimension of the bilinear projection.

$$
\begin{aligned}
p_{t,k}&=\sigma(w_k^\top\phi_t+b_k),\\
q_{t,r,k}
&=\sigma\!\left(
   \frac{(W_{Q,k}\phi_t)^\top(W_{R,k}u_r)}{\sqrt{D_p}}
   +a_k^\top\alpha_{t,r}+v_k^\top d(t,r)+b'_k
   \right),\qquad r\in\mathcal R_t .
\end{aligned}
$$

$p_{t,k}$ estimates $\Pr(Y_{t,k}=1)$ and $q_{t,r,k}$ estimates $\Pr(Z_{t,r,k}=1\mid Y_{t,k}=1)$. The joint probability of reporting $k$ and pointing to $r$ is $J_{t,r,k}=p_{t,k}q_{t,r,k}$. Several spans may be valid simultaneously, so the $q$ values need not sum to 1. If representative span annotations are available only for reports that can be localized, the estimate of $q$ applies to that subgroup. Estimating $J$ across all reports then requires additional information about the localizable fraction or the sampling design.

**Constructed example.** Suppose the probability of reporting a break in coherence on reading the first key word of the next sentence is $p=0.60$. Among readers who give that report, the proportion pointing to “that card” in the previous sentence is $q=0.75$. The joint probability is $J=0.45$. The observation occurs in the next sentence, while the referenced span is in the previous sentence.

Tensor operations can score all $p$ and valid $q$ pairs in parallel. The event output has shape $T\times K$. Span scores use sparse indices $(t,r,k)$ and can be computed in blocks. Enumerating all spans of length at most $B$ and their valid positions gives $O(BT^2)$ pairs. With a restricted candidate set, also report coverage of the spans annotated by humans.

### A separate intensity scale

The unit of $p_{t,k}$ is a reporting probability. If participants also rate mental effort or uncertainty on an $L$-level scale $R^{(m)}_t$, use a separate ordinal head. Let $\eta^{(m)}_t=v_m^\top\phi_t$ and define the scale using ordered cut points $\kappa_1<\cdots<\kappa_{L-1}$.

$$
F_{t,m}(\ell)=\Pr(R^{(m)}_t\le\ell)=\sigma(\kappa_\ell-\eta^{(m)}_t),\qquad
F_{t,m}(0)=0,\quad F_{t,m}(L)=1.
$$

The probability of level $\ell$ is $F_{t,m}(\ell)-F_{t,m}(\ell-1)$, and the mean score is $\sum_{\ell=1}^L\ell[F_{t,m}(\ell)-F_{t,m}(\ell-1)]$. If the study asks only readers who have a question to rate it, the readout also conditions on the presence of a question. Sentence-end self-ratings supervise sentence-end representations. Retrospective judgments use representations available after reading the full text and a dedicated head. Comprehension answers and reading times can serve as auxiliary endpoints with their own observed units and losses.

### Sentence-level outputs and masks

Let $S_j$ be the original character interval of sentence $j$. The within-sentence peak can be summarized by $\max_{t:I_t\cap S_j\ne\varnothing}p_{t,k}$, which retains the units of the score at an individual position. If the study aims to estimate the proportion of readers reporting difficulty with the whole sentence, calibrate that endpoint using separate sentence-level human annotations. For highlighting, choose thresholds $\xi,\zeta$ on development data. Let $I$ be an original text interval to render, such as a single token or an entire sentence, and let $I(r)$ be the character interval of the candidate span.

$$
m^{\mathrm{event}}_{t,k}=\mathbf 1\{p_{t,k}\ge\xi_k\},\qquad
m^{\mathrm{source}}_{t,k}(I)
=\mathbf 1\{\exists r\in\mathcal R_t:J_{t,r,k}\ge\zeta_k,\ I\cap I(r)\ne\varnothing\}.
$$

The first mask indicates when a response occurs. The second marks the original text that it refers to. If the span head is calibrated only to the localizable subgroup, preserve that condition in any displayed span scores. A separate supervision mask records which fields contain actual observations.

<a name="4-人類觀測與聯合訓練"></a>

## 4. Human observations and joint training

Let $i$ index a human trial with stimulus $X_i$, background $c_i$, observed stopping points, and the applicable reporting rules. An article can have multiple readers and stopping points. Identical model inputs can share features, while each human response contributes to the denominator and sampling variation. Code free reports into prespecified response types and referenced spans. INQUISITIVE offers a public starting point with questions at sentence ends and reader-selected spans. Its questions can supervise the reported doubts and their locations. To calibrate the reporting rate in a general reader population, identify the first word that triggers difficulty, or determine when it resolves, the data must support the corresponding sampling, negatives, and follow-up measurements. [INQUISITIVE, §3.2](https://aclanthology.org/2020.emnlp-main.530.pdf#page=3)

### Observation likelihoods for events and spans

Use the following model for a single stopping point and response type. Let $\mathbf z$ be the binary annotation vector over all valid spans, and let $\operatorname{Ber}(y;p)=p^y(1-p)^{1-y}$.

$$
\Pr(Y=y,\mathbf Z=\mathbf z\mid x_{\le t},c)=
\begin{cases}
1-p & y=0,\ \mathbf z=\mathbf 0,\\
0 & y=0,\ \mathbf z\ne\mathbf 0,\\
p\displaystyle\prod_{r\in\mathcal R_t}\operatorname{Ber}(z_r;q_r) & y=1 .
\end{cases}
$$

The product is the initial working model for multiple spans in a single report. If training uses only binary cross-entropy (BCE) for each span, it can be read as a marginal composite likelihood. The primary comparison uses $q$ or $J$ for individual spans and avoids relying on conditional independence across all spans. Marginals for multiple response types can share representations and training. Comparing whether two types co-occur requires data and a readout for that joint endpoint.

Let $\Omega_Y$ contain indices $(i,t,k)$ with a known positive or negative response and sampling suitable for estimating response rates. Let $\Omega_Z$ contain indices $(i,t,r,k)$ where $Y=1$ and whether the response targets the span is known. Let $\Omega_m$ index valid observations for measure $m$. Define the prespecified sampling or study weight $w_o>0$ and $n(\Omega)=\sum_{o\in\Omega}w_o$.

$$
\begin{aligned}
\mathcal L_Y
 &=\frac{1}{n(\Omega_Y)}\sum_{(i,t,k)\in\Omega_Y}
   w_{i,t,k}\operatorname{BCE}(y_{i,t,k},p_{i,t,k}),\\
\mathcal L_Z
 &=\frac{1}{n(\Omega_Z)}\sum_{(i,t,r,k)\in\Omega_Z}
   w_{i,t,r,k}\operatorname{BCE}(z_{i,t,r,k},q_{i,t,r,k}),\\
\mathcal L_m
 &=\frac{1}{n(\Omega_m)}\sum_{o\in\Omega_m}w_o[-\log P_{\psi,m}(y_o\mid \Phi_o)].
\end{aligned}
$$

$\operatorname{BCE}(y,p)=-y\log p-(1-y)\log(1-p)$. $\Phi_o$ depends on what was visible during measurement and on the object being evaluated. A stop-specific report uses $\phi_t$. An immediate rating for a sentence $r$ uses $[\phi_{e(r)};u_r]$. A retrospective rating for $r$ after the reader completes the article uses $[\phi_T;u_r]$. An article-level endpoint can use $\phi_T$. All vectors come from the same forward pass. Ordinal intensity uses the level probabilities from the previous section, and binary endpoints use a Bernoulli distribution. For $a$ responses and $n-a$ nonresponses that share a condition, the endpoint can use $-a\log p-(n-a)\log(1-p)$. Inference retains the original study's sampling unit. Sentence-level and article-level endpoints supervise their respective outputs.

- A completed report explicitly stating that there was no question can serve as a negative example under that reporting protocol. Exclude unsampled stops, incomplete reports, and indeterminate spans from their respective losses.
- Positive examples collected by requiring participants to ask a question can supervise the types and referenced spans of reported questions. If sampling retains only $Y=1$, the population denominator is unavailable and the record stays outside $\Omega_Y$. If the annotations cover only a subset of spans, other spans count as negative examples only when explicitly annotated.
- Define the target population for conditional span probabilities using the sampling rules. Suppose $L=1$ means a report is clearly localizable, and reports that cannot be localized, by definition, have no identifiable span. Estimate $\Pr(L=1\mid Y=1)$ separately and multiply it by $p$ and $\Pr(Z_r=1\mid Y=1,L=1)$. If the annotations are simply missing, use the source's observation mechanism to decide whether estimates can extend to all reports. The agent recodes any alignable raw reports. If the data remain insufficient, restrict estimates to the observable group.

When the measurements and sampling support the endpoint, the expected Bernoulli BCE equals the entropy of the true distribution plus its KL divergence to the model distribution. The expected loss therefore reaches its minimum at the true response probability. Use held-out human data to test whether the frozen representations retain sufficient information to estimate that probability. Sentence-end data constrain sentence-end responses. Outputs at unsampled tokens can propose finer-grained localization hypotheses.

<a name="從文獻效應回傳梯度"></a>

### Backpropagating from effects reported in the literature

Represent a published effect as the contrast from the original study, with its specified stimuli, reader groups, visible text, and measurement rules. Let $\mathcal I_{j,a}$ be the evaluable inputs and readout positions for condition $a$ in study $j$, and let $w_{j,a,i}$ be the normalized weights used by its estimator. Let $f_j(Y)$ be the measured quantity, such as whether a continuation refers to the first character or which level a reader selects on a subjective-effort scale. The model's condition means and effects are

$$
\widehat\mu^M_{j,a}=
\sum_{i\in\mathcal I_{j,a}}w_{j,a,i}
\sum_y f_j(y)\,P_{\theta,\psi,m(j)}(Y_i=y\mid X_i,c_i),\qquad
\boldsymbol\delta^M_j=C_j\widehat{\boldsymbol\mu}^M_j .
$$

$m(j)$ selects the readout corresponding to the study endpoint. If $Y$ is continuous, replace the inner sum with an expectation integral. A row of $C_j$ can specify a difference between two conditions or a difference-in-differences for prior knowledge × text coherence. If a study averages by reader, weights each article equally, transforms the scale, or estimates a mixed-model coefficient, record the original analysis and first check whether the data support recomputation. The expression above covers contrasts defined by condition means. Other statistics require a prespecified $g_j$ matching the original analysis. Complex endpoints can use a trial-level likelihood for training while retaining the original statistic for evaluation.

For development studies with the original stimuli and sufficient estimation precision, let $\widehat{\boldsymbol\delta}^{\,H}_j$ be the human effect estimate and $\widehat\Sigma_j$ its estimated covariance matrix. The loss can be

$$
\mathcal L_{\mathrm{effects}}=\sum_{j\in\mathcal J_{\mathrm{dev}}}
  \big[\boldsymbol\delta^M_j-\widehat{\boldsymbol\delta}^{\,H}_j\big]^\top
  \widehat\Sigma_j^{-1}
  \big[\boldsymbol\delta^M_j-\widehat{\boldsymbol\delta}^{\,H}_j\big].
$$

A single contrast is weighted by the inverse of its estimated variance. Effects from repeated measures or shared trials require covariance estimates. If these are unavailable, the first version selects one primary contrast for training and uses the others for diagnostics. If the same trials already contribute to the trial-level loss, the first version reserves effects aggregated from them for development diagnostics. When a study reports only an effect direction and the original comparison is reconstructable, use a prespecified directional diagnostic. Retain the precision of nonsignificant results. If the stimuli or matching endpoint are missing, mark the effect as untestable. If the estimate lacks precision, retain its uncertainty.

$\boldsymbol\delta^M_j$ is computed directly from model probabilities or ordinal distributions, so gradients can reach the readout head and any active LoRA parameters. Training on an effect requires forward passes for the texts in its conditions. Each text and reader condition still uses one Qwen pass. With Qwen frozen, cache the representations for each condition and update the small readout head on the complete contrast. A published effect retains its statistical grain. Two models can reproduce the same difference in response rates while assigning different absolute probabilities within each condition or placing difficulty at different tokens. Trial-level annotations and held-out localization test these two possibilities. Earlier weak-supervision methods have constrained expected model outputs. [Mann & McCallum, §2](https://jmlr.org/papers/v11/mann10a.html) · [Liang et al., §2](https://www-cs.stanford.edu/people/pliang/papers/measurements-icml2009.pdf)

### Joint objective

The first version freezes $\theta=\theta_0$ and minimizes the following objective over readout parameters $\psi$.

$$
\widehat\psi=\arg\min_\psi
\left[
\lambda_Y\mathcal L_Y+\lambda_Z\mathcal L_Z+
\sum_m\lambda_m\mathcal L_m+
\lambda_E\mathcal L_{\mathrm{effects}}+
\lambda_{\mathrm{reg}}\|\psi\|_2^2
\right].
$$

First set $\lambda_E=0$ and train endpoints that have trial-level observations. Continue training shared parameters with the effect loss in the second stage. Use trial-level annotations in every round of that stage to protect within-condition calibration while fitting average effects. Keep the split, available features, and readout head fixed when comparing trial-level and joint losses. Choose the weights on development data. If LoRA is used, low-rank parameters update selected matrices in $\theta=\theta_0+\Delta\theta$. The input, causal mask, and readout contract stay the same. All held-out stimuli and experimental families are excluded from setting weights, features, and thresholds. [Staged training protocol](training-protocol.md#訓練順序)

### First comparison with a fine-tuned backbone

Compare frozen Qwen with a targeted LoRA configuration using the same token-level readout, supervision, and stimulus split. The first intervention targets the query path and the key projection, `k_proj`, of the layer 24 full-attention block. The query and key directly determine attention logits for text already read. Qwen's `q_proj` interleaves query and output-gate components within each head's output. Query-only LoRA updates only the query output rows. The original matrices stay frozen while low-rank increments change the effective weights. Training can cache the full frozen sequence states before layer 24 and run the differentiable layer 24 to recompute causal attention and its dependent readout representations. Inference still uses one forward pass. [Pinned Qwen implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L688-L743) · [LoRA definition](https://arxiv.org/abs/2106.09685)

Check reporting base rates, readout convergence, and prefix causality independently for both configurations. On unseen stimuli, evaluate localization error, calibration, and the human effect of placing the same target word in different preceding contexts. The earlier PLABA fine-tuning experiment targeted the final-layer MLP `down_proj` for sentence-level ratings. That configuration had higher out-of-fold NLL. The result applies to that configuration. Each layer's role in maintaining or resolving questions still requires independent evidence. [Expository-text LoRA results](expository_lora/results.md) · [Frick base-rate diagnostics](union_training/localization-results.md#frick-readout-the-fit-at-overall-early-stopping)

<a name="5-同一疑問何時維持解開或重新出現"></a>

## 5. When a question persists, resolves, or reopens

In the basic model, $p_{t,k}$ is the proportion of readers who report the response at a stop under specified conditions. The difference $p_{t+1,k}-p_{t,k}$ measures a change in that population proportion. To estimate a transition between positions for an individual reader, track the same question for that reader. For example, half of the readers may still have questions at each of two stops. The same half could have questions at both stops, or one half could replace the other. Marginal rates cannot distinguish these cases.

With tracking data, assign a discrete state $S_{t,r}$ to each trackable question $r$. In this section, $r$ includes the span identifiable at that point and the question's identity. Several questions about one span require labels that can be matched across stops. The state set $\mathcal S=\{N,U,P,R\}$ denotes a question that has not yet arisen, remains unresolved, has been set aside, or has been confirmed as resolved, respectively. Using the set-aside state requires an operational measure of question tracking or retrieval. The available data determine whether to include it. Tracking starts when the candidate span becomes visible. The initial state at the preceding position is $N$. We prespecify the legal transition set $\mathcal A$. Examples are $\mathcal A(N)=\{N,U\}$, $\mathcal A(U)=\{U,P,R\}$, $\mathcal A(P)=\{P,U,R\}$, and $\mathcal A(R)=\{R,U\}$.

Given $x_{\le t},c,r$, each new state depends only on the preceding state. For every legal state pair $(a,b)$, score the transition in parallel with the readout $g_{a,b}(\phi_t,u_r)$. Here $u_r$ represents both the referenced span and the question type. Normalize separately for each starting state to obtain the transition matrix.

$$
K_{t,r}(a,b)=
\begin{cases}
\exp g_{a,b}(\phi_t,u_r)\big/\displaystyle\sum_{b'\in\mathcal A(a)}
                       \exp g_{a,b'}(\phi_t,u_r), & b\in\mathcal A(a),\\
0, & b\notin\mathcal A(a).
\end{cases}
$$

Let $e$ be the first position where the whole candidate span is visible. Let $\mu_{e-1,r}\in\mathbb R^{1\times|\mathcal S|}$ be the initial distribution concentrated at $N$. Multiply the matrices from left to right in token order. The state probabilities and the probability of each transition after reading through $t$ are

$$
\mu_{t,r}=\mu_{e-1,r}\prod_{s=e}^{t}K_{s,r},\qquad
\Pr(S_{t-1,r}=a,S_{t,r}=b)=\mu_{t-1,r}(a)K_{t,r}(a,b).
$$

For example, $N\to U$ is the first onset, $U\to R$ is resolution, $P\to U$ is a return after setting the question aside, and $R\to U$ is reopening after confirmed resolution. The accumulated population quantity for questions that remain unresolved can be defined as $\sum_r\mu_{t,r}(U)$. If the candidate questions are distinct, this is the expected number of unresolved questions, measured in questions. A perceived break in coherence at the start of the next sentence retains its own $p_{t,k}$. Readers may report it while an earlier question remains unresolved or while they are setting it aside. The report and the count of unresolved questions are distinct observations.

Given the exact states $a,b$ of the same question at $u,v$, the interval negative log-likelihood is

$$
-\log\left[\prod_{s=u+1}^{v}K_{s,r}\right]_{a,b}.
$$

For multiple stops with noisy or partial observations, let $D_{t,r}$ be a diagonal matrix with diagonal entries $P_\psi(o_{t,r}\mid S_{t,r}=a,\phi_t,u_r)$. Conditional on the states and features at their respective stops, the observations follow their corresponding distributions. For an exact state annotation, use the indicator of the matching state. In the absence of an observation, use the identity matrix. The likelihood of the whole trial is

$$
P_\psi(o^{\mathrm{obs}}_r\mid X,c)
=\mu_{e-1,r}\left(\prod_{s=e}^{T}K_{s,r}D_{s,r}\right)\mathbf 1 .
$$

Adding tracking data allows training to minimize this negative log-likelihood. When the state model also accounts for the same stop-specific endpoint, derive its report probability from $\sum_a\mu_{t,r}(a)P_\psi(o_{t,r}\mid S_{t,r}=a,\phi_t,u_r)$. One model then determines both the transitions and the reporting rates at the stops for that endpoint.

For ordinal reports such as TED-Q, incorporate the scale into the observation model $D_{t,r}$. Earlier recorded ratings and the end of the excerpt determine when tracking stops. Under the original study rules, the first rating of 3 can be used to evaluate the interval when an answer is partially available. To evaluate complete answers, use identity matrices at unmeasured stops after ratings of 3–4. The original question's state can continue to change. That stopping rule provides no evidence for setting the latent state to $R$. Directly evaluating the recorded rating of 5 requires retaining the denominator and bounds for first unmeasured stops where a later segment was available. [Observed TED-Q missingness bounds](semantic_tracking/tedq-full-answer-bounds.md)

All $K_{t,r}$ and report probabilities come from the same Qwen forward pass and parallel readout. The loss or analysis program computes the matrix products above. An order-preserving associative scan can also compute all prefix products at once. Sampled states never feed back into Qwen. Even for the same reader and question, two stops typically constrain only the matrix product between them. Concentrating all updates at the first or the last unobserved token can yield the same interval likelihood. If existing data lack finer observations within the interval, validation reports the interval. Testing last-in-first-out ordering requires observations of when several questions arise and resolve.

## 6. Validation, localization, and new predictions

### Engineering validation

Let $O_{\le t}$ contain all token-indexed outputs through $t$ and their references to candidates already read. For two inputs with the same background, position indices, and visible token prefix,

$$
x^A_{\le t}=x^B_{\le t}\ \Longrightarrow\
O_{\le t}(X^A,c)=O_{\le t}(X^B,c).
$$

Check this theoretical equality using deterministic inference and the backend's numerical tolerance. Compare a shared token prefix with different suffixes, and compare the full text with a truncated prefix. Before constructing the tests, verify the actual token IDs the tokenizer produces at the boundary. When comparing character-span outputs, also check the original character offsets in the common prefix. Check features, candidates, and normalization against the same causal policy. When comparing prefix outputs, any ranking based on the full text must leave the prefix's candidates unchanged.

<a name="真人效度"></a>

### Validation against human responses

Keep originals, rewrites, shared stimuli, and reanalyses from the same stimulus family in one split. Once the model, features, weights, thresholds, and scoring rules are fixed, report performance on held-out stimuli, readers, or experimental families. Use the measures appropriate to each endpoint.

| Endpoint | Validation measure |
| --- | --- |
| Whether a response occurs | Log loss, Brier score, and binned calibration on held-out trials. The Brier score is the mean squared difference between estimated probabilities and binary reports. Its range is $[0,1]$, and lower is better. |
| Where a response points | Candidate coverage of the human-annotated spans and a full localization score that counts candidate misses. Match original character spans one to one using exact match or a prespecified overlap criterion. |
| Subjective intensity | Ordinal negative log-likelihood and absolute error of the mean score on the original scale. |
| When a response occurs or changes | Compare human reports and estimated probabilities at prespecified aligned stops. Use the expressions below to report the direction, magnitude, and identifiable location range. |

Index stimulus family by $d$, version by $v$, reader group by $g$, and prespecified aligned stop by $j$. Let $\tau(d,v,j)$ map the same experimental position to a valid token in each version. Aggregate both sides over the common distribution $\nu_g$ of stimuli and reader backgrounds. The model estimates and condition effects are

$$
\begin{aligned}
\mu^{M}_{v,g,j,k}
 &=\mathbb E_{(d,c)\sim\nu_g}
       [p_{d,v,\tau(d,v,j),k}(c)],\\
\Delta^{M}_{g,j,k}&=\mu^{M}_{B,g,j,k}-\mu^{M}_{A,g,j,k},\\
E_{g,j,k}&=\Delta^{M}_{g,j,k}-\Delta^{H}_{g,j,k}.
\end{aligned}
$$

Estimate $\Delta^H$ from human trials on the same scale and with the same weights. When versions are randomly assigned, it estimates the effect of the rewrite for that group. Comparing $\Delta$ across background groups can test effect heterogeneity. Interval estimation accounts for readers, stimuli, and version pairings according to the study's assignment and sampling units. With a prespecified tolerance $\varepsilon$, one replication criterion is that the chosen confidence interval lies entirely within $[-\varepsilon,\varepsilon]$.

To estimate the earliest difference, first define a practically meaningful magnitude $\delta>0$, the direction, the grid of observation stops, and intervals that account for multiple comparisons. The model can predict a position where its estimate first crosses $\delta$ in the prespecified direction. Test the human results only at the observed stops. Distinguish the first stop with evidence of a difference from the range in which evidence constrains its onset. If the interval at the previous stop indicates a near-zero difference and the interval at the next stop indicates a difference above $\delta$, use the region between the two stops for localization. Sparse observations leave open whether a difference appeared and disappeared between stops.

## 7. Experimental contract for the first version

In the first round, define each reference reader population and endpoint using the existing observations for that source. Use $\mathcal L_Y$ for response types with a denominator, $\mathcal L_Z$ for readouts with span annotations, and ordinal heads for available rating scales. Train and validate state transitions only on data that can track the same question for the same reader. Report results at the sentence or segment level when observations stop at that resolution. The token-level response curves remain model estimates.

[Original research record](../../research/mathematical-formulation.md)
