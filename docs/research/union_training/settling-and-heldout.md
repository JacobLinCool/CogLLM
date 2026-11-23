# Subsequent training and formal held-out validation specifications

Round 12 is the designated upper limit for full joint training and also the round with the lowest development loss among these 12 rounds. Fine-tuning reads weights from the verified Round 12 readout, replays the source family equal weighting development NLL, with an allowed absolute numerical error of 0.00002. Training uses the same 24-source family data packets and the saved 18-batch causal representations, while fine-tuning reinitializes the AdamW momentum state. [Original Full Result](observation-full-results.md) · [Batch Continuation](observation-resume.md)

## development set early stopping rule

Update the early stopping reference value only if NLL development loss decreases by at least 0.001. Starting from 0.0003, if no improvement threshold is met for 8 rounds, sequentially reduce to 0.0001 and then 0.00003. The final learning rate is also recorded as development loss stable if the threshold is not met for 8 rounds. A maximum of 72 additional rounds may be added. The final weights are taken from the initial checkpoint and all additional rounds with the lowest measured development NLL. This criterion measures the practical stability under the current development set and learning rate schedule. [Continued training implementation](../../../research/union_training/fit_observations.py)

## formal held-out

After freezing the candidate with the corresponding observational endpoints, data exclusions, scaling, and criteria for each study, the formal stimulus representation is extracted using an independent L4 stage. Its scope is limited to text and the original author's reading stop. Following the readout calculation and prediction based on the registered readout per study, the formal human trial is loaded only after the sealed snapshot is created. Results are reported according to the source family of the original sealed snapshot. The scope of applicability is determined by each validated readout and the target measurement. [Original sealed snapshot list](holdout-plan.md) · [Independent forward entry](../../../research/union_training/formal_colab.py)

Anglin's complete evidence pages were transcribed column-by-column into the model using manually verified text and charted numerical values. The length limit for this corpus was set at 4,096 tokens, with the formal execution capacity of the fixed backbone verified prior to execution. Traditional Chinese MECO and other studies adhere to the existing range of 2,048 tokens. All states remain a single causal forward pass until the corresponding reading stop. [Per the study's forward specification](../../../research/union_training/formal_extract.py)

[Original research record](../../../research/union_training/settling-and-heldout.md)
