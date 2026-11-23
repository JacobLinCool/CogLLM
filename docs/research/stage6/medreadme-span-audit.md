# MedReadMe alignment of original text for difficult words and phrases

The authors' annotation team, after training, independently selected terms and difficult phrases, which were then verified by adjudicators. Annotations utilized a predefined set of seven lexical categories, encompassing general difficult words, abbreviations, medical nouns, and two levels of query difficulty for medical terms. This serves as an external word-level highlighting interface. The authors did not provide data on reader word-level self-assessments, query latency, or individual vote counts. [Original §§2.1–2.3](https://aclanthology.org/2024.emnlp-main.958.pdf#page=3)

Fixed-author published `jargon.json` and `readability.csv`. Within each source, original, simplified, and split, perform unique pairing based on identical character sequences after removing only whitespace. Segment-by-segment verify annotated token coordinates, original fragments, and mapped character ranges back to the CSV source sentence. Published test splits contain 1,140 sentences, with 1,131 uniquely matched, retaining 2,130 annotated segments. Additionally, 9 sentences could not be disambiguated and were excluded from alignment. [Fixed-version published file](https://github.com/chaojiang06/medreadme/tree/7c52205d18f16b80e3ffccb4e9da88b1942df310/dataset) · [Alignment audit](../../../research/results/medreadme-span-audit.json)

Among the 194 medical terms that are difficult to comprehend via short-term queries, 145 contain at least two Gradio colorable complete words, and 49 contain one. Applying existing word-level readout to these annotations enables measurement of word localization and multi-word segment coverage. Data were not released for article ID, simplified pairing ID, or per-person scoring. Formal testing requires first verifying and matching the text overlap of the data, and specifying the evaluation grouping granularity. The CEFR sentence-level scoring follows a different administration procedure, with original alignment establishing only the usable scope for segment annotations. [Term Rules and Annotation Constraints](https://aclanthology.org/2024.emnlp-main.958.pdf#page=7) · [Existing sentence-level Audit](textcomplexityde-transfer.md#the-availability-scope-of-other-data)

```sh
python -m research.stage6.medreadme_span_audit \
  --raw .cache/cogllm/medreadme-audit/raw \
  --output research/results/medreadme-span-audit.json
```

The two cross-checks yielded identical results. At this stage, only the summary counts and the reproducible script are released. [Script](../../../research/stage6/medreadme_span_audit.py)

[Original research record](../../../research/stage6/medreadme-span-audit.md)
