# Adaptive joint training with retention constraints

In the initial fixed and adaptive comparison, both full-data fits lost the Qwen seed's ability to locate early and late resolution cues. This follow-up addresses that regression using the [original data, observation, and adaptive-weighting protocol](protocol.md). The same human evaluation materials and previously scored engineering diagnostics make this a retrospective evaluation.

We freeze the token readout coefficients from initialization. The new observation heads can fit freely, and all observations can update the shared state through the same final-layer residual. After each update, we project the residual matrix into a Frobenius norm ball of radius 0.10. For any original projected state, the residual can change it by at most 10% of that state's norm. This is a bound on vector change. Its implication for the event probabilities depends on the readout.

The seeds, data splits, observation likelihoods, adaptive method, update sequence, and budget match the initial experiment. We select checkpoints using only the original development set. Relative to the seed curve, we allow development NLL to rise by up to 0.05 nats on Frick and 0.12 nats on ExpMRC. Among eligible checkpoints, we select the one with the lowest source-balanced development loss. If all checkpoints exceed the limits, we retain the one with the smallest total excess and report that outcome. We replay the scores on the same materials for the fixed and unconstrained adaptive arms and report retention, joint fit, efficiency, and human effects separately.

[Original research record](../../../research/adaptive_joint/retention.md)
