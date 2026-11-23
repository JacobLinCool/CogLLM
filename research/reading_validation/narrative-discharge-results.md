# Does a common clue resolve several open questions at once?

At the lines with the largest annotated drop in four German stories, an average of 5.75 questions closed. CogLLM's shared-clue model predicted **0.10 closures** at those same lines. It estimated a net count change of −0.026 questions. The model learns that questions can close together and still misses the words that made several questions resolve for the annotators. These four complete works contain 222 annotated plot questions. The annotators could consider a complete story when marking open questions and their lifetimes. [Original study](https://doi.org/10.1017/langcog.2025.10057) · [Shared-clue observation model](narrative-discharge-protocol.md) · [All story-level scores](../results/reading-validation-narrative-discharge/scores.json)

| Held story | Independent closures | Shared clue | Shared clue with known annotation team | Most at once, observed | Estimated net change at that line, shared clue |
| --- | ---: | ---: | ---: | ---: | ---: |
| Auftrag | 13.33 | 14.24 | 6.65 | 7 | −0.13 |
| Katze | 147.71 | 142.28 | 85.91 | 7 | −0.04 |
| Thomas | 13.99 | 15.00 | 7.32 | 4 | +0.01 |
| Verkehrsunfall | 17.01 | 17.42 | 21.75 | 5 | +0.06 |

The three model columns give squared error in the number of questions outside the interval allowed by a team's annotation, in questions squared. Each story served once as the intact holdout. The shared model's average error was 47.24, versus 48.01 for independent hazards. With a known team's prior annotation style the scores were 30.41 and 32.02 respectively. Team information is unavailable for an unknown reader. The shared model selected a closing source over an active competitor 69.5% of the time, versus 70.8% for elapsed duration from the previous study. Given the true closing line, it placed a correct source first among all earlier lines 8.8% of the time. Four previously examined stories do not support a general claim about discharge magnitude or same-question identity. [Earlier baselines and annotated trajectories](stock-process-results.md) · [Fitted probabilities and convergence](../results/reading-validation-narrative-discharge/forecast/seal.json)
