# Post-forward same-question state computation

We introduce a differentiable question-state computation. The input consists of per-token transition scores derived from features obtained during a single causal forward pass through the prediction head, along with a mask indicating when each question becomes visible. For each question, we simultaneously maintain the probabilities of four states: unseen, unsolved, pending, and solved. The readout layer outputs the state probabilities for each token, the probabilities of each transition event, and the distribution of the first time a question is solved. After solving, the system can be reopened, allowing transitions from the solved state back to the unseen state. The computation employs balanced matrix multiplication in log space. The output is generated at the readout layer, without backpropagating estimated states to Qwen. [Code](../../../research/semantic_tracking/question_states.py) · [Mathematical Definition](../mathematical-formulation.md#5-同一疑問何時維持解開或重新出現)

The state of a single problem can be constrained by confirmed observations, observations that exclude only certain states, or ordered report constraints. When missing, the masked element is the observation itself, while the query itself may still shift to a read position. The first interval opened uses the probability of first entering the solved state, and if no observation is made until the end, the right-hand bound is retained. Even if a problem is reopened later, the distribution of first entering the solved state remains consistent with its original meaning. [TED-Q observation missing example](tedq-full-answer-bounds.md)

## Compare with existing sealed snapshot scores

Treating the initial hazard responses of the original three groups as a special case of computing a new state, allowing transitions only from unsolved to solved, with solved states remaining solved. The replay of 196 questions and 588 sealed snapshot prediction paths across three stories. Of these, 183 questions have observation intervals, including 6 cross-line intervals, while 13 are right-censored. The results from the first-solution distribution scoring, observation-state scoring, and the independent product of probabilities are consistent, with a maximum error below **4.5 × 10⁻¹⁵ nats**. The average error per story matches the prior score, and the two CPU replays are identical at the bit level. [Original Experiment](../semantic-joint-first-run.md) · [Replay Code](../../../research/semantic_tracking/question_states_replay.py) · [sealed snapshot values](../../../research/results/question-states-interval-replay.json)

In small-scale enumeration, we exhaustively enumerate all state paths, verifying mask, first-unlock, parallel tracking, re-open, gradient, and long-sequence numerical stability. When replacing or truncating the suffix, the prefix states and first-unlock probabilities match exactly across all values. Additionally, we validate the prefix using 8 synthetic trajectories, each 8,192 tokens long. The FP32 state probability normalization error does not exceed 7.1 × 10⁻⁶, and FP64 does not exceed 1.3 × 10⁻¹⁴. [Independent enumeration verification](../../../research/semantic_tracking/verify_question_states.py)

## Observation scope for temporary suspension and reopening

Onea/PITQ re-computation involved 16 original lines simultaneously opened with one question and first closed with another, supporting variations where a single position can have multiple different questions. The `?offen` is the annotator's uncertainty about whether a solution already exists there. after the question ends, subsequent marks are cleared per instructions, so this data only validates the existing first-closed interval. Re-opening is a supported transition in the state model, but there is still a lack of empirical sequences of the same reader on the same question. The new computation has completed the program and replayed the old endpoints. model weights and Gradio's experimental display maintain the original acceptance status. [Original Workbook Verification](onea-reopening-source.md)

Execute the following command from the project root directory using Python with PyTorch. Replay the per-line probabilities and confidence intervals from the original public results, without loading Qwen.

```sh
python3 -m research.semantic_tracking.verify_question_states
python3 -m research.semantic_tracking.question_states_replay \
  --source research/results/semantic-joint-first-run \
  --output /tmp/question-states-interval-replay.json
```

[Original research record](../../../research/semantic_tracking/question-states-run.md)
