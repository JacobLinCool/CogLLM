# Segment-wise judgment updates by Qiao and Lagnado

Two experiments tracked how the same reader re-evaluated the same specified proposition after encountering new written evidence. For each observation point, the public data included a guilty probability rating of 0–100, a binary decision choice, a confidence rating of 0–100, and two conditional probabilities. The first rating appeared only after the evidence was read. The decision following complete evidence remained a reader response, with neither the stimulus nor the observation table specifying a correct answer. There were no records of spontaneous questions, first-noticing moments, or question resolution. [Original Study 1 Methods, Study 2 Methods](https://escholarship.org/content/qt1zj2p06s/qt1zj2p06s_noSplash_1f0b500b4d2d2385f750b4aa8628058b.pdf) · [Author Data](https://osf.io/u72fr/)

## Data and event locations

| Sample and presentation method | Author-reader record | Observations per person | D→P/P→D readers | Final guilty rating D→P / P→D |
| --- | ---: | ---: | ---: | ---: |
| Study 1, item-by-item | 81 | 8 | 43／38 | 67.33／48.29 |
| Study 1, per-group completion rating | 78 | 2 | 38／40 | 54.95／49.98 |
| Study 2, a within-subjects design followed by a between-subjects design. | 93 | 5 | 45／48 | 60.67／55.35 |

D denotes the defense and P denotes the prosecution. Final scores are the equal-weighted average of each reader's last 0–100 score. Study 1's 804 rows use the author's analysis ID, consistent with their R code. The underlying platform column has two values each matched to two analysis IDs, resulting in 159 author-reader records having 157 distinct public platform values. Study 2's spreadsheet contains 465 rows and 93 distinct analyses and platform values. The platform values across the two studies are disjoint. The paper records that Study 2 recruited 95 participants, with 2 excluded due to attention issues. [Study 1 CSV](https://osf.io/download/whmzj/) · [Study 2 XLSX](https://osf.io/download/2y4pr/) · [Two-studies code](https://osf.io/u72fr/)

Page 1 of the stimulus PDF gives the case, page 2 lists 4 numbered pieces of evidence for each side, and pages 3–4 list the conditional-probability questions. Both orders in both studies preserve the order 1–4 within each side. We aligned Study 1's item-by-item group to the end of each segment and its grouped condition solely to the ends of segments 4 and 8. We aligned Study 2 to the ends of segments 1, 2, 3, 4, and 8. One reader's source-evidence tags at positions 1 and 5 in Study 1 were swapped relative to the authors' design. Because the other stops also depend on earlier sequence order, we retained that reader's scores and left the text stops blank for all eight rows. Source labels and sequences were consistent with the design across the other 796 rows. Study 2's table contains only group and stage, so alignment of its 465 rows follows the presentation design in the paper. The public inventory includes no Qualtrics file. We could verify the authors' materials and intended order, while the exact screens, whether headings appeared again, and whether prior ratings were redisplayed remain unverified. [Stimulus PDF](https://osf.io/download/znkef/) · [Original Procedure](https://escholarship.org/content/qt1zj2p06s/qt1zj2p06s_noSplash_1f0b500b4d2d2385f750b4aa8628058b.pdf)

In Study 1, the last difference between the D→P and P→D readout was +19.04 points, with a within-group comparison (p=.009). The whole-group report yielded (p=.983). In Study 2, the last difference was +5.31 points, with (p=.280). Both studies shared one case and eight pieces of evidence, allowing comparison only of the presentation order and question timing for this case. Study 2 provided different readers and new mixed prefix conditions, with no independent article available for held-out. If evaluating single-article causal forward, the body text remains identical when the same order and paragraph are read. To test response distribution differences across scoring timelines, the data design must specify the scoring timeline known to readers in advance, or constrain to a single timeline. [Original Study 1 Main analysis, Study 2 Main analysis](https://escholarship.org/content/qt1zj2p06s/qt1zj2p06s_noSplash_1f0b500b4d2d2385f750b4aa8628058b.pdf)

## File and reuse

| File | OSF | SHA-256 |
| --- | --- | --- |
| Study 1 CSV | [whmzj](https://osf.io/download/whmzj/) | `f8c6ca10bda24d510e4f8f8f940faad054335dc0499b02100bc389ea5c3ac2e6` |
| Study 1 Rmd | [p9nfq](https://osf.io/download/p9nfq/) | `7c851861a8a3d7a67aeb1bcc2b1f0ef318e942e37ffadb30309d3be6a0f174c7` |
| Study 2 XLSX | [2y4pr](https://osf.io/download/2y4pr/) | `ff5aca229054500dbbe37a1e12a4d3294b6afc447cf99dce46f7c3c21cef6c6d` |
| Study 2 Rmd | [mn8xb](https://osf.io/download/mn8xb/) | `8f4b83e27b8b71ee01bc301b7c89f63e4e3445d98b4a6f96a38c6d2c6b54dade` |
| Material PDF | [znkef](https://osf.io/download/znkef/) | `31c660cb7a79d573ecf558f7d8f4c21a8cc3ec4fe28f500502e2531d0ca41b21` |

The paper is licensed CC BY 4.0. The OSF project is public. the project license field is empty, and no license association is present. No alternative license applicable to standalone spreadsheets, code, and materials was found. The reproducible code uses a fixed-version PDF to extract case facts and evidence, merges paragraph-level spacing, removes headings and numbering, and employs exclusive character coordinates. The original platform field, author ID, free-response, and independent body text were not submitted to the version control system. Outputs from two local executions are bitwise identical. [Paper License](https://escholarship.org/uc/item/1zj2p06s) · [OSF Metadata](https://api.osf.io/v2/nodes/u72fr/) · [Local Build Code](../../../research/semantic_tracking/qiao_evidence_prepare.py)

[Original research record](../../../research/semantic_tracking/qiao-evidence-source.md)
