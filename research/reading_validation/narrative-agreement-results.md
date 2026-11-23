# Where do two human annotation teams agree that a question resolves?

Three original German stories have two team workbooks for the same printed lines. Each team recorded its own potential plot questions and when they remained open. Among 164 team-labelled definite changes, the other team made a definite change on the identical line on 42 occasions. It agreed in direction on 34 of those 42. For the 58 definite declines, only eight had any determinate change from the other team at that line, and four of those eight agreed. Counts include the two directions of the comparison; 42 paired team decisions correspond to 21 shared line positions. The source researchers describe variation across annotators and distinguish these retrospective annotations from live reader responses. [Original study](https://doi.org/10.1017/langcog.2025.10057) · [Source and analysis protocol](narrative-agreement-protocol.md)

| Original and target team | Line with the largest guaranteed decline | Target team's change | Other team's change at that line | Frozen model's change |
| --- | ---: | ---: | ---: | ---: |
| Auftrag, AA | 199 | −11 | 0 | −0.123 |
| Auftrag, JE | 100 | −3 | 0 | −0.004 |
| Thomas, AA | 357 | −4 | 0 | +0.117 |
| Thomas, JE | 358 | −4 | 0 | −0.015 |
| Verkehrsunfall, AA | 68 | −6 | 0 | +0.052 |
| Verkehrsunfall, JE | 61 | −3 | −1 | −0.004 |

Each entry is the net change in open questions per line. We report the earliest of tied maxima. In *Thomas* both teams' largest declines were four questions and occurred one printed line apart. In *Auftrag* the largest declines were 99 lines apart. This is a direct reason to report both location and tolerance, with the original text granularity visible. The frozen count model recovered 26 of the 58 strictly negative directions and 80 of the 106 strictly positive ones. A rising duration curve recovered zero and 106 respectively. Their overall counts of correct directions were identical. [Exact intervals, ties, curves and team-specific counts](../results/reading-validation-narrative-agreement/scores.json)

The other team's raw count yielded mean squared error 25.87 outside the allowed count interval. Calibrating its offset and nonnegative slope on the other two works reduced that error to 8.00. The text-only count model scored 16.19 and elapsed duration 12.41. The calibrated peer had access to a second human annotation of the held work. Its performance is a measurement reference, unavailable when an editor types an unseen passage. Exact online question onset, closure and identity need evidence from a reader performing the relevant task at the time. [Frozen causal count forecasts](stock-process-results.md) · [Reproduction code](narrative_agreement.py)

## Timing tolerance in the original text

Allowing a one-line timing difference raised the number of distinct drop pairs from two to eight. Each human change could match only one from the other team. Across 58 human drop labels the symmetric F1 therefore rose from .069 to .276. Rotating change locations within the same 30-line story sections yielded 3.94 matches on average at that tolerance (one-sided randomization p = .0135). Extending the window to two lines still gave eight real matches and 6.26 under this local chance control. Rises were more reproducible, with 15 exact matches, 20 within one line and 22 within two. The one-line tolerance was motivated by the already inspected *Thomas* example, and only three works contributed. The data identify a timing tolerance and substantial residual disagreement. [Timing rule and null model](narrative-window-protocol.md) · [All original matched coordinates](../results/reading-validation-narrative-agreement/window-scores.json)
