# ARAU 2.1 bridging prefix held-out data

The Open Edition of GNOME includes 5 medical and art history documents, while PEAR includes 20 verbatim oral history transcripts. Both corpora retain the full documents as published by the authors, with no training, validation, or test splits provided. A program verified the SHA-256 hashes of both compressed archives against the Apache-2.0 license file, with the version fixed to `764d5e53c10656ff1858503c1d32752b03241174`. [Official Data and License](https://github.com/arrauproject/data/tree/764d5e53c10656ff1858503c1d32752b03241174) · [Cited Paper](https://doi.org/10.1017/S1351324919000056)

| The set | File | MMAX word position | Bridging link | Single-word prediction | Both endpoints are located in different sentences. |
| --- | ---: | ---: | ---: | ---: | ---: |
| GNOME | 5 | 21,458 | 693 | 549 | 400 |
| PEAR | 20 | 14,059 | 333 | 306 | 261 |

Read the primary and secondary fields of `related_object=yes` and parse the marker pointed to by `related_phrase` as an exact word span. The 693 GNOME source links contain 1 secondary field link. Retain only one complete antecedent span for the primary score, which must be fully located before the trigger span. Annotations lacking an antecedent, requiring multiple markers, involving non-contiguous spans, or overlapping positions are retained in the local validation log. Cross-sentence scoring requires each span to be fully contained within different sentence segments of the original MMAX annotations. GNOME has only 5 document sampling units, while PEAR's segments are verbatim transcripts. These datasets represent manually annotated discourse, not measuring reader immediate confusion or understanding resolution. [Data construction program](../../../research/semantic_tracking/arrau_bridge_prepare.py)

`model_input` Output only `id`, `text`, `stops`, and `paragraphs` per document. `text` preserves the MMAX token string verbatim, tokenized by single spaces, with line breaks retained to maintain original sentence boundaries. `stops` includes all annotated token end positions. Since the original RAW text cannot be perfectly reconstructed in MMAX word order even after removing whitespace, all `paragraphs` are empty. Both datasets can be used for token position and cross-sentence antecedent identification. The answers differ from the original annotations only at `scoring` and `private_metadata`.

After applying Unicode NFKC normalization, case folding, and punctuation removal, we compared each of GUM's three splits against GENTLE. Neither GNOME nor PEAR found an exact full-text match or shared a consecutive 12-word span with any of the reference documents. This validation covers the held-out and matching sources currently in the project, without consulting the base model's pretraining corpus.

```bash
python3 -m research.semantic_tracking.arrau_bridge_prepare \
  --source .cache/cogllm/arrau-bridge/source \
  --output .cache/cogllm/arrau-bridge/prepared
```

`audit.json` Record the hash of every output. All outputs from two runs are byte-identical, and re-running will refuse to overwrite existing outputs. Verify each instance to confirm that all prefixes and anchor spans correspond to a valid reading stop, with both ends of the cross-sentence entries falling within the declared segments. This step involves no model fitting or scoring.

## Qwen prefix coordinate validation

Using the current candidate's fixed tokenizer, we compared the independent encoding truncated at the source word end against the token prefix of the full text. All 35,517 word ends across 25 documents matched perfectly. the longest file contained 6,816 model tokens. For 855 bridging annotations, both endpoints met the word boundary condition, including 661 cross-sentence cases. The prefix stage utilized only the tokenizer, consuming 39.6 seconds on a single-threaded CPU with a peak memory usage of approximately 259 MiB. The candidate ranking conditions and results are documented elsewhere. [prefix manifest ](../../../research/results/arrau-bridge-preflight.json) · [no-issue packet construction program](../../../research/semantic_tracking/arrau_projection_prepare.py) · [held-out ranking results](arrau-bridge-results.md)

[Original research record](../../../research/semantic_tracking/arrau-bridge-intake.md)
