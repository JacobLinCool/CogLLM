# Onea/PITQ concurrent re-opening and parallel state

Frick's workbooks can annotate a story while keeping certain questions open. They cannot determine whether the same reader first marked a question as resolved and later reopened it. After re-reading the full text, annotators clear the remaining `offen` entries starting from the row where an answer is confirmed. If new phrasing only evokes the same still-relevant question, the guidelines require merging and deleting duplicate entries. Annotators may separate questions with similar phrasing but distinct clear time intervals. Blank entries in closed sections and subsequently reopened entries with identical phrasing lack the same link that a question reverts to unresolved status. [English annotation guidelines §§3–4](https://osf.io/download/h9mxg/)

Study annotators constructed the question wording and states. Study 1 includes expert annotations by the researchers. Study 2 used two groups of student annotators. A separate reader group rated suspense at text positions during reading, without reports for individual questions. The paper also states that the experiment provides no direct evidence about whether a particular question arose during reading. [Onea et al., §§3.1, 4.1, and 5](https://doi.org/10.1017/langcog.2025.10057) · [OSF release and README](https://osf.io/m524v/)

## Workbook recalculation

Five original `Textannotation` workbooks encompassed three stories, 1,597 workbook × original line positions, and 196 question columns. A column-by-column scan of the first confirmed-close original marker yielded zero re-openings. This result describes how published data record question status. When two annotators independently annotated the same story, positions were counted separately. therefore, these lines cannot be treated as independent reader samples. [Workbook SHA-256, per-line events, and summaries](../../../research/results/onea-reopening-audit.json)

| Identifiable events on the original row or question column | Quantity |
| --- | ---: |
| Peers initiate one or more questions and first confirm closing other questions. | 16 lines |
| Peers have confirmed open and uncertain unresolved status markers. | 369 lines |
| There are at least two questions marked as uncertain whether they have been solved. | 262 lines |
| Within the same workbook, pair different question columns that have identical normalized wording. | 5 pairs, 4 with nonoverlapping intervals |

In *The Brazilian Cat* `Katze / AA_AA`, row 38 (worksheet column 40) simultaneously marks one newly opened and another confirmed closed, with the order unknown within the row. Row 76 contains 13 `offen` and 1 `?offen`. The wording of the same worksheet's rows 38 and 94 normalizes to identical, with marked intervals spanning rows 114 to 121 (previously) and rows 398 to 406 (previously). This pair demonstrates that identical wording alone can erroneously merge two originally separate intervals into a single reopened event. [Original coordinates and statistics](../../../research/results/onea-reopening-audit.json)

After re-timing the retention of the 2nd column, the A column contains the original rows with the body text, excluding the importance column. Each column opens at the first `offen`, with the subsequent first whitespace confirming the close. Re-check the traversal after closing all rows to find `offen` or `?offen` within the same column. `?offen` indicates that the annotator, after reading and looking back at the text, remains uncertain whether an answer has already been provided. The wording pairing uses Unicode NFKC, casefold, and whitespace folding, and checks for overlap between two half-open intervals within the same workbook. Re-use the existing [spreadsheet readout function](../../../research/semantic_tracking/resolution_prepare.py) to directly scan the original state, without employing the existing lifecycle validator. [Field definition and source hash](../../../research/results/onea-reopening-audit.json)

We can use this data to check whether the model can retain multiple questions within the same original line while updating different questions. Re-initialization supervision requires preserving the identity of the same question and observing it transition from unsolved to solved and back to unsolved in sequence. This re-computation only adopts original lines that have the original workbook. `Verkehrsunfall` extends the index and suspense score without including statistics. [Existing position and labeling source explanation](../semantic-tracking-first-run.md)

[Original research record](../../../research/semantic_tracking/onea-reopening-source.md)
