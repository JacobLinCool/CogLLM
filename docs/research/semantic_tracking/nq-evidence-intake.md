# Acquisition and verification of Natural Questions held-out data

From the official validation index with IIRC source overlaps excluded, page versions are sorted by `SHA256('cogllm-nq-v1:' + revision)`. the top 72 groups and all 153 original items are selected. The index SHA-256 is `6a4a3921b7fa142eaf34db148e0f93a44fd0da6313a2e9c2b63b8853cad247b5`. Before download, fixed selection rules and complete IDs are applied. Data originates from `google-research-datasets/natural_questions` on Hugging Face, using `default/validation` and commit `e8103d566bef4154c2c12b17c6095ec5275840cc`. [Release data](https://huggingface.co/datasets/google-research-datasets/natural_questions) · [Google original data and license](https://ai.google.com/research/NaturalQuestions/download)

The original document directory is `.cache/cogllm/natural-questions/holdout-raw-v1`. `grade-free.jsonl` preserves the original index, ID, question, complete raw HTML, token field, and optional sections provided by the author. `outcomes.jsonl` saves the raw `annotations` with the same ID using `0600`. Before predicting the sealed snapshot, only record identification and file integrity are validated. manual labeling is deferred to scoring.

| File | SHA-256 |
| --- | --- |
| `manifest.json` | `70396f1836192abfceec936de5612c583639704bdba9f2a16b4adb2dc3f840e3` |
| `grade-free.jsonl` | `1c35576a14366846ff6ebbb11fb86def078cec4121612aec8a66341e0ddaf698` |
| `outcomes.jsonl` | `f1b87047a2fef50293dbd037a68ae59cb5e4e59608d563affa9afed27ab2333e` |

The delivered machine file consists of two retrieval path combinations. The Hugging Face Rows API first completes 100 records, validates the `X-Revision`, original index, ID, URL, and truncation fields, then saves hashes in batches of four. After rate limiting, the remaining 53 records use the `default/validation-00000-of-00007.parquet` to `default/validation-00006-of-00007.parquet` from the aforementioned commit. Data is streamed in chunks to a temporary disk file, with chunk lengths verified against the release list. Reading via `pyarrow.parquet.ParquetFile(path, memory_map=False, pre_buffer=False, buffer_size=262144).iter_batches(batch_size=4, use_threads=False)`, rows corresponding to the index's `row_in_shard` are selected before conversion to Python. After verifying IDs, URLs, and fields, a single JSONL writer separates columns, which are then merged according to a fixed selection order. Temporary chunks are subsequently removed. The successfully saved API content comprises 13,058,839 compressed bytes, while Parquet chunks total 1,338,488,354 bytes. The Parquet program's measured RSS peak was 742.8 MiB, and the program has terminated.

After rate limiting resolution, the Rows API independently completed all 153 rows. The successfully saved compressed content was 18,789,577 bytes. The complete Rows result matches the two data files in the table bit-for-bit. The following command uses independent output to reconstruct the identical data file. The manifest records this retrieval path and transfer volume. [Retrieval Program](../../../research/semantic_tracking/nq_evidence_fetch.py) · [Rows API Documentation](https://huggingface.co/docs/dataset-viewer/en/rows)

```sh
python3 -m research.semantic_tracking.nq_evidence_fetch \
  --index .cache/cogllm/natural-questions/iirc-disjoint-index.jsonl \
  --output .cache/cogllm/natural-questions/holdout-raw-v1-replay
```

[Original research record](../../../research/semantic_tracking/nq-evidence-intake.md)
