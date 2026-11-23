# Full joint training representation continuation

The full development corpus follows 24 source families, 62 observation blocks, and existing material splits. The splitter chunks 27,167 inputs, 164,220 matched reading stops, and 2,598,224 tokens into 18 batches according to the original document order. Each input belongs to exactly one batch. Batch limits are 150,000 tokens and 10,000 reading stops, preserving the forward pass of individual documents intact. [Observation Specifications](observation-protocol.md) · [Batch Planning and Merging](../../../research/union_training/observation_shards.py)

## Preservation and Continuation

Each batch carries a SHA-256 hash of the original observation packet, model version, tokenizer, extraction program, and study specification. The L4 extraction program loads a frozen Qwen once per work session. Each input undergoes a formal causal forward pass and an independent prefix check. Immediately after batch completion, the representation, position, extraction manifest, and prefix check are published and returned to the host via a 12 MiB file. The controller first compares the fragments and complete packets, then validates the representation coordinates and causal checks. Only after the complete file passes is the batch formally saved. [Remote Executor](../../../research/union_training/shard_worker.py) · [Host Controller](../../../research/union_training/checkpointed_colab.py)

When the original training phase is lost but the host still retains complete batches, the controller requests only the missing batches in the next L4. After validation on the host, the training phase can be resumed by combining the representations received from the original execution with the representations recovered from the host. The final manifest records the hash of all batches. The controller can be re-invoked from the original execution directory within the configuration window using the same immutable batch plan. The host simulation acceptance covers scenarios including disconnection after submitting the first batch, retrieving the remaining batches, and restoring the original batch.

## Joint training validation

Merge the full-scale representation into memory-mapped files following the original batch order, ensuring each reading stop is accounted for exactly once. Subsequently, run 12 iterations of the new likelihood measurement with shared projection, and develop a multiple-choice checkpoint within the material grouping. For training and results, bind the sealed snapshot manifest to each batch. Predict the re-computed development loss and 36 paired effects using the sealed snapshot in its original scale. The formal study group adopts the role of the pre-existing sealed snapshot. [Training](../../../research/union_training/fit_observations.py) · [Result Validation](../../../research/union_training/verify_observations.py) · [Effect Diagnosis](../../../research/union_training/observation_effect_diagnostics.py) · [Formal Sealed Snapshot](holdout-plan.md)

## Entry point execution

Create a batch package in a Python environment with NumPy and tokenizers installed in the project root directory. The first command outputs the execution directory.

```sh
python -m research.union_training.checkpointed_colab prepare \
  --packet .cache/cogllm/union-training/observations-2 \
  --tokenizer-path "$QWEN_TOKENIZER" \
  --holdout research/union_training/holdout.json \
  --output .cache/cogllm/union-training/checkpointed
```

When initiating or resuming the same packet, pass the original execution directory to the same command. The controller verifies available stored batches and reclaimed slices, then configures L4 based on remaining work. Results are submitted to the `fit/` within that directory, and the status of each L4 is written to `sessions/`.

```sh
python -m research.union_training.checkpointed_colab run "$COGLLM_RUN_DIR"
```

Full execution completed, resuming automatically from 15 verified batches following a disconnection during a single remote training epoch. [Full training and actual backfill logs](observation-full-results.md)

[Original research record](../../../research/union_training/observation-resume.md)
