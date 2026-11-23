# What the bilingual sentence readout measures

Let an original passage contain ordered words and let *t* index the end of one visible sentence. A causal Qwen3.5-2B encoder reads the full passage in one forward. At each eligible word it yields a contextual state, word surprisal (the sum of its subword surprisals), and attention paid to completed earlier sentences. The causal mask and source character offsets restrict the features at *t* to words seen through that sentence. Coordinates projecting states from two native layers, their recency differences, and native backward attention are shared across the language-specific readouts. Length, repetition and lexical recurrence remain explicit controls. A target contains the original text only.

For each boundary, the readout forms an 86-dimensional vector. Eighteen features describe form, eight describe word surprisal, and sixty describe contextual states, similarity and attention. Let \(x_{it}\) be that vector after centering and scaling by the corresponding language's training material. The common difficulty score is \(d_{it}=x_{it}^{\mathsf T}\beta\). Its numerical zero follows from source normalization. It has no absolute psychological unit.

For an English study that supplies *k* positive marks in *n* usable responses at that boundary, the predicted mark probability is

\[
p_{it} = \operatorname{logit}^{-1}(b_E + d_{it}).
\]

Minimize binomial cross-entropy, giving each article the same total weight and retaining its real numbers of reports. A separate Italian study provides only a final mean rating \(y_i\) on a 1–5 scale. Pool its noninitial boundaries and predict the observed passage mean using

\[
h_i=\log \left(|T_i|^{-1}\sum_{t\in T_i} e^{d_{it}}\right),\qquad
\widehat y_i = 1+4\operatorname{logit}^{-1}(b_I-e^g h_i).
\]

Use squared error on that published scale. High pooled disruption lowers expected coherence. Its derivative distributes training signal over boundaries in proportion to \(e^{d_{it}}\), leaving their individual latent locations uncertain. An Italian paragraph's final rating supplies one observation regardless of how many sentences it contains.

The basic shared head has 89 fitted quantities, comprising 86 slopes, two intercepts and one log scale. Both tasks retain their own loss; normalize each by the source-only constant-prediction loss and give each task equal weight. The slopes have a quadratic penalty and the log scale is bounded and regularized. An independent head for each task uses its own slopes. Source-development groups estimate how much to mix each task's independent and shared *predictions*. Thus a difficult local boundary in English can influence which features the common slopes count in Italy, while shared fitting must still predict the actual Italian response. The collection of independent control weights and feature normalization constants exceeds the 89 quantities of the common head. This is a research readout separate from the writer app's trained heads.

For physically certified Italian sentence permutations let superscripts \(C\) and \(S\) denote source order and swapped order, and let \(t_C,t_S\) indicate the same sentence after its original or disturbed predecessor. The augmented head adds a single penalty

\[
0.25\left[\frac{\mathbb E_{\rm paired}\operatorname{softplus}(-(h^S-h^C))}{2\log 2}+
\frac{\mathbb E_{\rm localized}\operatorname{softplus}(-(d^S_{t_S}-d^C_{t_C}))}{2\log 2}\right].
\]

Each average weights original source documents equally. Every comparison preserves all words and punctuation. The target states the source's physical order. Only the original published passage carries a human rating. Restoration training can improve source-order detection while moving human-response agreement in a different direction; both effects have separate registered tests.

A displayed English probability estimates the recorded task-specific decision at the end of a sentence. An Italian or Danish paragraph estimate describes a predicted retrospective evaluation. Neither calibrates the intensity of a particular reader's continuously experienced load at a particular word. No synthetic human judgments enter this model. [Registered estimates](coherence_order_protocol.md) · [Exact source intervention](italian-contrast-protocol.md) · [Replay instructions](discourse-order-reproduction.md)
