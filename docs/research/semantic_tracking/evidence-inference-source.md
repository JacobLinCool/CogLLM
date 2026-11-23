# Cross-domain evidence localization for Evidence Inference 2.0

Evidence Inference 2.0 provides full clinical trial papers with physician-annotated evidence spans, allowing verification of whether scores shift to another manually selected position when the same body text is queried with a different known query. The original study first had physicians construct queries conditioned on intervention, comparator, and outcome, then had a second physician read the body text, answer, and select evidence, followed by a third physician auditing the answers and evidence. Experimental endpoints were restricted to results reported in the body text and offline-selected positions. The released version lacks the time of first evidence discovery, the status of questions during reading, and reports from the same reader across different reading stop points. [Original Study §2.1–2.3](https://aclanthology.org/2020.bionlp-1.13.pdf)

## Source and Authorization

Verify the author's GitHub `a661e8c14f973398380c8865cf2f27a535aaaf6d`. The following paths are relative to that version's `annotations/`. Use `(PMCID, PromptID)` to link queries with evidence, and `UserID` to distinguish annotators. Character coordinates are based on the author's same-version `txt_files/PMC<id>.txt`. Article licensing is verified against the original rights statement by `xml_files/PMC<id>.nxml`. The author also marks the dataset as MIT on Hugging Face. PMC article terms are independent. before evaluation, each must be retained with explicit evidence of CC BY, CC0, or public domain licensing, and exceptions for third-party materials must be checked. [Fixed Version File](https://github.com/jayded/evidence-inference/tree/a661e8c14f973398380c8865cf2f27a535aaaf6d/annotations) · [Author's Hugging Face Release](https://huggingface.co/datasets/jaydeyoung/evidence_infer_treatment) · [PMC Open Access Terms](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/)

| File | Number of published files | SHA-256 |
| --- | ---: | --- |
| `annotations_merged.csv` | 24,686 determinations | `a5dd1a3583e5105a0f77d3e1c695008879859425c9a93dfbc07054ef462e0255` |
| `prompts_merged.csv` | 12,865 queries | `0d70d9e1e78d113fdbbd4919310eab7768d6a328368f7d8f9d5a3b48f99c25d3` |
| `splits/test_article_ids.txt` | 449 papers | `4fe6c633ba8f077a68a8feea48d4408c1749175e1c46f2b77d96727d291ef9d8` |

## Verified positions

All approved evidence intervals for the two queries in each column are mutually non-overlapping. Characters are extracted from the UTF-8 decoded body using the half-open position `[start, end)`, and each matches the original citation in the published file exactly. Digitally identical intervals may be selected by multiple physicians. however, each interval type within the table is listed only once. [Author field and body description](https://github.com/jayded/evidence-inference/tree/a661e8c14f973398380c8865cf2f27a535aaaf6d/annotations#article-types-xml-vs-txt)

| Official test PMCID | Blank token count | Article authorization | Query A and interval | Query B and interval |
| --- | ---: | --- | --- | --- |
| `PMC1871574` | 5,648 | CC BY 2.0 | `286`，`[17327,17629)` | `298`，`[19921,20317)` |
| `PMC3536709` | 6,114 | CC BY 2.0 | `3166`，`[17080,17192)` | `3169`，`[17429,17524)` |
| `PMC4497728` | 5,459 | CC BY 4.0 | `8982`，`[1711,1987)` | `8988`，`[2536,2658)` |

The initial screen identified 1,711 duplicate pairs from 275 official test papers. The initial screen included only entries where the verifier approved both the answer and the rationale, the interval was resolvable, and citations were present, requiring that the two queries be fully disjoint within the retained intervals. This represents an availability estimate prior to formal screening, not yet verified against authorization and complete citations. The authors additionally published problematic PromptIDs and recorded partial intervals as `-1` or offset by several characters. Formal ingestion requires first applying fixed exclusion rules, then requiring each adopted interval to align character-for-character. The field description writes the endpoint as inclusive. the above actual data aligns perfectly with slices that are exclusive of the endpoint. [Author's coordinate revision and problem list](https://github.com/jayded/evidence-inference/tree/a661e8c14f973398380c8865cf2f27a535aaaf6d/annotations#caveat)

Article length should be measured using the actual tokenizer, including query and separator tokens. The empty split token counts in the table above cannot determine whether the full text fits within the existing 8,192-token budget. Before formal testing, fixed candidate boundaries that can be pre-extracted from the text must be established to verify the alignment between manual character intervals and causal prefix token boundaries, recording coverage rates caused by length or alignment. The aforementioned three cases have demonstrated the existence of matching pairs with identical text but different evidence. This source audit was completed prior to model testing.

[Original research record](../../../research/semantic_tracking/evidence-inference-source.md)
