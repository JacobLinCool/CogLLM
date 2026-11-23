# CogLLM

CogLLM uses a language model to simulate **measurable human responses to reading**. Our goal is to predict where readers may struggle to follow an explanation and how their response may change when an author revises it. For example, an editor can score an original sentence and a plain-language rewrite, then compare their predicted ease on the same five-point scale that readers used. The public app takes an English passage and marks the original words and sentences in one causal forward pass through Qwen3.5-2B.

[Try the public Gradio Space](https://huggingface.co/spaces/JacobLinCool/CogLLM) · [Download the trained CogLLM readouts and read the model card](https://huggingface.co/JacobLinCool/CogLLM-Qwen3.5-2B)

## How CogLLM learns from readers

We keep Qwen3.5-2B frozen and train small readouts on its representations of the text. **Each observation contributes a loss for the action the reader took, at the point where the study measured it.** A one-to-five ease rating trains the probabilities of the five ordered responses. In a study where readers could report a contradiction after each sentence, the model learns from how many readers first reported it at each sentence and how many reached the end without reporting it. When annotators identify a passage containing evidence, training uses the annotated interval. The exact word at which a reader recognized the evidence remains unknown. [Observation models](docs/research/union_training/observation-protocol.md) · [Joint training protocol](docs/research/adaptive_joint/protocol.md)

The released token checkpoint combines **65 training objectives from 26 sources**, including human reading studies and annotated evidence tasks. They update a shared representation, with a fixed weight budget per source in the selected checkpoint. We fitted the editor's readouts separately. PLABA-EVAL supplies biomedical sentence ratings, phrases readers named as difficult, and original–rewrite pairs. CoheSentia supplies readers’ sentence-by-sentence judgments about whether a story still makes sense. At inference, the editor and the token curve reuse the same pass over the passage. [Checkpoint selection](docs/research/adaptive_joint/results.md) · [Wording and ease training](research/reading_validation/local-protocol.md) · [Connection training](research/reading_validation/transition-external-results.md)

## Which human responses did it reproduce?

| Test | Human response | CogLLM result |
| --- | --- | --- |
| Did plain-language revisions make biomedical sentences easier? | Across 265 original–rewrite pairs, readers rated the rewrites **0.537 points easier** on average on a five-point scale. | When evaluated on held-out reader and article groups, CogLLM predicted a gain of **0.497 points**. It recovered the direction and approximate size of the average effect. [Paired revision results](research/reading_validation/local-results.md) |
| Could it locate the wording a reader identified? | Readers explicitly named 94 difficult expressions in biomedical texts. | Given the relevant sentence and one underline, CogLLM overlapped **45 of 94** named expressions. A model using word frequency, length and position found **25 of 94**. [Phrase localization](research/reading_validation/local-results.md) |

These results support using CogLLM to simulate **specific responses of a reader population**, such as the average ease gain from a rewrite. Tracking one person's unresolved question remains an open goal. In eight constructed cases with two open questions, the released token curve linked both questions to their respective answers in **0 of 8** cases. [Question-tracking evaluation](docs/research/adaptive_joint/results.md)

## Read the marks

Start with **Load example**, or paste an English draft of up to 512 model tokens and select **Review draft**. Read the whole marked passage before deciding what to edit.

| Look for | What it means for an editor |
| --- | --- |
| Orange to pale green sentence backgrounds | Orange indicates a lower predicted ease rating. Look for a sentence whose idea or construction deserves a second read. The scale comes from readers' ratings of biomedical sentences. |
| A solid underline | The model's first wording candidate in that sentence. Even an easy sentence can have a candidate. |
| A dashed outline | Readers may find the connection difficult to follow here. Read the outlined sentence together with the preceding one. |

Open the experimental token trace below the passage to see how the signal changes as each token becomes visible.

[Download the illustrated research report](output/pdf/CogLLM-reading-study.pdf), [read the study and its release decision](research/reading_validation/manuscript.md) or [inspect the experiments by human response](research/reading_validation/validation-report.md).

[See where readers’ next-word guesses narrow or widen in a complete passage](research/results/reading-validation-writer/word-expectations.html). The shaded range estimates how often two people will write different next words. A wide band shows how much the answer depends on spellings outside the study’s 8,192 named alternatives.

[See a short physical story with its earlier action and possible conflicting sentence highlighted](research/results/reading-validation-writer/physical-story.html). This optional view uses one original passage and explains why it chose those two sentences.

## Use the hosted model from Python

Install `gradio_client` and call the same endpoint as the editor. The result is HTML with the highlighted original text. A request sends the passage to the public Hugging Face Space.

```python
from gradio_client import Client

editor = Client("JacobLinCool/CogLLM")
marked_html = editor.predict(
    "The sensor measures reflected light. Its reading changes when the leaf dries.",
    api_name="/review_english",
)
```

For local inference, the [v0.1.0 Space source](https://huggingface.co/spaces/JacobLinCool/CogLLM/tree/v0.1.0) contains the runner and its pinned dependencies. The [v0.1.0 model files](https://huggingface.co/JacobLinCool/CogLLM-Qwen3.5-2B/tree/v0.1.0) include the trained token checkpoint (209,097 parameters), the sentence and wording heads, the continuity coefficients, and their SHA-256 manifest. The runner downloads the frozen Qwen backbone at the revision used for training.

On four reference passages, the hosted CPU editor reproduced all 11 underlines from the L4 run. Every displayed ease rating was within 0.005 points of its L4 reference on the five-point scale. [Release verification](release/v0.1.0-verification.json)

## Additional research views

The full research interface supports up to 2,048 model tokens and provides the optional views below. Each estimates a different observable reading response. The [local guide](docs/try-cogllm.md) includes instructions and examples.

| Mark on your draft | What the model estimated |
| --- | --- |
| Whole-summary percentage | For an English summary up to 1,024 model tokens, estimates the chance a rater would report being unable to read and understand it. [View a summary flagged for review](research/results/reading-validation-writer/summary-review-example.html) |
| Purple overline in narrative mode | A word may introduce a person the reader cannot yet identify. Select **Check character introductions in narrative summaries** to see these marks. |
| Blue earlier phrase and outlined pronoun | Select a pronoun you intend to refer back. Its proposed earlier phrase is marked in the original draft. [Inspect an example](research/results/reading-validation-writer/reference.html) |
| Green term marks in technical prose | Follow an earlier term to possible explanatory wording. The curve estimates how many earlier mentions still await a definition. [View a complete example](research/results/reading-validation-writer/terminology.html) |
| Solid brown or dotted blue under a word | Readers' next guesses spread out or narrow after the marked word. The optional view underlines a word when the estimated change reaches 20 percentage points even after allowing for unlisted spellings. [Open the example](research/results/reading-validation-writer/word-expectations.html) |

An editor might change the term, introduce the missing idea earlier, or keep an intentional shift. The tool leaves that judgment to the author. [Run the model and read an example](docs/try-cogllm.md)

View a saved result from the live model for an [English science passage](research/results/reading-validation-writer/review.html) or a [narrative summary](research/results/reading-validation-writer/narrative.html).

## What the marks can do

- In the official English summary test, the whole-summary observer flagged 92 of 206 difficult summaries that contained scorable text. One in three alerts corresponded to a professional difficulty judgment. [English summary study](research/reading_validation/transition-seahorse-results.md)
- On 39 held-out film summaries the default narrative screen marked 650 of 26,417 words as possible unexplained characters. Depending on where readers intended repeated names, 64–82% of those alerts matched a recorded mark. On unambiguous locations it recovered 337 of 413 marked words, with 94 additional flags. The higher-recall mode produced 1,467 flags. [Both thresholds, uncertainty and a previously unseen film](research/reading_validation/transition-writer-balanced-results.md)
- On English paragraphs from a separate laboratory, the next-word observer identified the direction in which readers’ guesses changed at 74.2% of 1,559 positions with a measurable change. The uncalibrated backbone achieved 73.6% on the same grouped answers. Of 482 conservative marks in the writer interface, 422 matched the direction of actual full-spelling changes and 315 also reached the marked size. [Human word distributions and directions](research/reading_validation/cloze-population-results.md) · [Tests of the actual highlights](research/reading_validation/word-mark-certification-results.md) · [Independent eye movements](research/reading_validation/expectation-rereading-results.md)
- In 49 held-out short stories, the continuity model signalled 77 of 123 reader-marked breaks and flagged 48 of 647 unmarked reader decisions. On a separate set of 100 learner essays it flagged 114 of 446 editorially marked breaks and 131 of 906 unmarked boundaries. A simpler model combining word cues and Qwen surprisal ranked the essay breaks better. [Boundary evaluation](research/reading_validation/transition-external-results.md)

The English wording and connection readouts evaluate completed sentences. The experimental curve estimates a state at each token from the prefix available there. The writing app takes the passage alone. Projects on measured reader knowledge and cross-study training are described in the [research documentation](docs/README.md).

## Citation

```bibtex
@misc{cogllm2026,
  author = {Jhen-Ke Lin},
  title = {{CogLLM: Modeling How Reading Difficulty Changes}},
  year = {2026}
}
```
