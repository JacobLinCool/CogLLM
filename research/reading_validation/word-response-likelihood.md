# A single causal pass with response-specific observation models

Let a complete printed sentence be tokenized as \(x_{1:T}\). A causal encoder produces \(h_t=f_\theta(x_{1:t})\) at every position in one forward evaluation. For printed word \(i\) ending at native token \(t_i\), the readout uses \(z_i\), a vector containing word form, native whole-word surprisal, and a training-only projection of contextual states at that stop. Word surprisal sums \(-\log p_\theta(x_t\mid x_{<t})\) over the complete printed word. A participant’s response to the word enters the likelihood once, at \(t_i\). The public releases supply population measurements from different cohorts. The three endpoints therefore have separate observation models. [English voltage measurements](https://doi.org/10.1016/j.bandl.2014.10.006) · [English behavioral measurements](https://doi.org/10.3758/s13428-023-02261-8)

Let \(j\in\{E,C,R\}\) denote N400 negativity, cloze miss count, and rated unpredictability. The conditional predictors share input features and have distinct coefficients \(\eta_{ij}=\alpha_j+z_i^\top b_j\). Within each training fold the continuous outcomes are centered and standardized. The three observation laws are

\[
\begin{aligned}
y_{iE} &\sim \mathcal{N}(\eta_{iE},\sigma_E^2),\\
k_{iC} &\sim \operatorname{BetaBinomial}\{n_i,\phi\,q_i,\phi(1-q_i)\},\quad q_i=\operatorname{logit}^{-1}(\eta_{iC}),\\
\bar y_{iR} &\sim \mathcal{N}(\eta_{iR},\sigma_R^2+s_{iR}^2/n_{iR}).
\end{aligned}
\]

Here \(k_{iC}\) is the number of people who supplied a different word in the cloze task. The beta-binomial accounts for dispersion across words. The term \(s_{iR}^2/n_{iR}\) incorporates the uncertainty of the released mean rating. A missing endpoint has indicator \(m_{ij}=0\). It supplies no term to its likelihood or observed information matrix. The conditional objective is

\[
-\sum_{i,j} m_{ij}\log p_j(y_{ij}\mid\eta_{ij})
+\tfrac12\sum_{d=1}^{D} B_{d,:}\Sigma^{-1} B_{d,:}^\top ,
\]

where row \(B_{d,:}\) contains the three readout coefficients for feature \(d\). The intercepts are outside the penalty. A diagonal \(\Sigma\) learns separate endpoint shrinkage. A full \(\Sigma\) allows data for one observed response to inform a coefficient for another response. In a training split, update the covariance from the estimated coefficient means and posterior coefficient variances,

\[
\Sigma^{\rm next}
=\frac{\widehat B^\top\widehat B+\sum_d V_d+8(0.35)^2 I}{D+8}.
\]

\(V_d\) is the corresponding three-by-three block of the inverse training Fisher information including the prior precision. The beta-binomial information sums over every possible count from zero to the released number of respondents. Model parameters and variance components are fitted using training sentences, and each fold retains its own transformation. The covariance describes statistical borrowing among feature coefficients. There is no within-person joint response in these independently recruited cohorts from which to estimate a reader-specific latent trajectory.

The sparse experiment establishes a concrete operating regime. Ten measured EEG sentences supplied 80–88 physiological words per training fold. With complete training cloze and rating observations, full covariance improved held EEG log density by 0.0078 nats per word over equally estimated independent margins (95% sentence-cluster interval 0.0019–0.0135). With all three endpoints observed throughout the training split, the incremental sharing gain was indistinguishable from zero. A lower-dimensional trained comparison remained stronger in the sparse setting. [Estimator implementation](convergent_adaptive.py) · [Sparse acquisition experiment](convergent-sparse-protocol.md) · [Measured outcomes and comparisons](convergent-adaptive-results.md)
