# AmbiStory upper bound correction

The Colab fit on 2026-11-21 used the original sealed Qwen causal features. The surface-form control completed fitting. The readout with the ending state reached the original L-BFGS-B iteration limit at step 300, and SciPy reported nonconvergence. That run produced no sealed AmbiStory predictions, and the human holdout ratings remained unused. [Original experimental protocol](ambistory-context-update.md) · [Original sealed fitting protocol](../../../research/results/ambistory-fit-protocol.json)

Set the readout iteration ceiling to 3,000 for outputs containing terminal states. Maintain the vocabulary size at 300. Both utilize identical zero initialization, L-BFGS-B, five-class negative log-likelihood, L2 regularization, `ftol`, and `gtol`, requiring the solver to report convergence and generate finite weights. Adhere to the original sealed snapshot specifications for raw data, features, train/test splits, evaluation metrics, and acceptance thresholds. Verify raw versus corrected specifications prior to scoring. Additionally, save the iteration count, objective function, and gradient diagnostics in the corrected version.

[Original research record](../../../research/stage6/ambistory-numerical-amendment.md)
