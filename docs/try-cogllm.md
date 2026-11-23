# Review an English draft with CogLLM

Open the [public writing editor](https://huggingface.co/spaces/JacobLinCool/CogLLM), paste a passage or choose **Load example**, and select **Review draft**. The model keeps your words and line breaks and marks the entire draft at once. The public editor accepts up to 512 English model tokens and includes an expandable experimental token curve. The local research interface supports up to 2,048 tokens and the additional views described below. A token can be a word, part of a word, or punctuation.

> A sensor measures how much light a leaf reflects. The resulting hyperspectral reflectance signature varies with the concentration of pigments in the leaf. This means that two leaves can reflect different amounts of light even when they look equally green.

**Start with the sentence color.** An orange background indicates lower estimated reading ease on a fixed 1–5 scale learned from readers of biomedical abstracts. A pale green background indicates greater ease. Sentences without a verified endpoint are left unshaded. The list below the passage gives up to three sentences with the lowest estimated ease.

**Read the underlined words in context.** CogLLM offers one wording candidate per eligible sentence. For the example above, an editor can ask whether readers already know what a reflectance signature measures and whether a definition would be more helpful before the technical term. The underline is a relative rank within a sentence. In the biomedical evaluation the top candidate overlapped 45 of 94 phrases that readers explicitly named. [Wording evaluation](../research/reading_validation/local-results.md)

**Check dashed sentences against the preceding paragraph.** A dashed outline is an alert about continuity at sentence completion. The model was trained on readers of short stories. In independent learner essays the same alert caught 114 of 446 expert-marked breaks, including 17 of 112 discourse-relation problems. An unmarked sentence can still need a clearer explanation. [Connection evaluation](../research/reading_validation/transition-external-results.md)

**Review a character's introduction.** Select **Check character introductions in narrative summaries**. A purple line marks a word that may introduce someone the reader cannot identify. Across 39 held film summaries, the default marked 650 of 26,417 words. Between 64% and 82% of flags matched a recorded reader mark, accounting for repeated names with uncertain locations. For a higher-recall screen, launch with `--narrative-operating-point high_recall`. [Saved example](../research/results/reading-validation-writer/narrative.html) · [Both policies on the same movies](../research/reading_validation/transition-writer-balanced-results.md)

**Inspect a backward reference.** Select **Inspect a pronoun that refers back** when you know the word refers to something earlier. Open its line under the draft. CogLLM outlines the selected pronoun and marks the proposed earlier phrase in the original passage. Opening another choice uses the same analysis. In 37 held texts the top word fell in the phrase a person actually selected on 78.4% of backward-reference choices. On decisions that also allowed a new or nonreferring expression, selective precision was 54.6%. Your selection supplies a meaningful distinction. [Try a captured example](../research/results/reading-validation-writer/reference.html) · [Human choices](../research/reading_validation/transition-pointer-results.md)

**Follow a technical term to its explanation.** Select **Track technical terms and their explanations** for a passage up to 1,024 model tokens. A solid green mark shows the earlier term; a dotted green mark shows where the model thinks its explanation begins. The graph below the draft estimates how many earlier terminology mentions still await an explanation. One term can appear more than once. In unseen biology and physics textbooks 485 of 1,627 proposed links matched both the term and its later definition interval. The count's units are terminology mentions; individual readers can know a term already. [Open a complete example](../research/results/reading-validation-writer/terminology.html) · [Terminology evaluation](../research/reading_validation/stock-results.md)

**Check which words an English learner might need help with.** Open **Which words might an English learner need help with?**, enter one English sentence, then choose **Check learner vocabulary**. A percentage over a word estimates the share of similar learners who would say they did not understand it. Select the reader's CEFR level (A1–C2) if known, or leave it unspecified for the study-wide estimate. Changing the level reuses the same sentence. Among forecasts from 10% to 20%, readers reported difficulty on 14% of the recorded responses. For the 20–40% range, the model predicted 28% on average and readers flagged 33%. Evidence is limited for the highest scores and the A1 group. [See an actual model example](../research/results/reading-validation-nns/writer/example.html) · [Direct learner reports](../research/reading_validation/nns-results.md) · [Observed rates by score and proficiency](../research/reading_validation/nns-calibration-results.md)

**Check an entire English summary.** Select **Check whether this summary is understandable** for a summary up to 1,024 model tokens. The percentage estimates how often a study rater would say they could not read and understand the complete summary. On an actual short news summary, the model estimated 49%, flagged it for review, and the original professional judgment recorded difficulty. Other sentences in the same view use independent sentence ratings. [Open the flagged original](../research/results/reading-validation-writer/summary-review-example.html) · [Open a clear how-to example](../research/results/reading-validation-writer/summary-study-example.html) · [Which judgments the screen recovered](../research/reading_validation/transition-seahorse-results.md)

**Check a physical story.** Open **Do two events in a short story conflict?** and enter five to seven sentences about everyday actions. In the example, the water freezes in sentence 3 and someone pours it in sentence 5. The app preserves the entire story and marks those two sentences together. On 153 held story families, the model identified both sentences in 55% of the edited stories and flagged 33% of the unedited ones. [See the verified example](../research/results/reading-validation-writer/physical-story.html) · [Scope and held evidence](../research/reading_validation/state-transport-results.md)

**Follow the reader’s next guess.** Select **Show where readers’ next guesses change** for an English passage of 2–64 words. A brown underline means guesses spread out after this word. A dotted blue underline means they narrow. An underline requires a model-estimated change of at least 20 percentage points even if readers supply different unlisted words. The graph shows the resulting range at each word. A wide band means many guesses fell outside the 8,192 named spellings. Open **Values at each word** for the before and after ranges. Across 482 marks in English study paragraphs, 422 matched the direction of the change in actual answers and 315 also reached 20 points. [Open the complete example](../research/results/reading-validation-writer/word-expectations.html) · [Grading the underlines](../research/reading_validation/word-mark-certification-results.md)

**Open Experimental token signals** to see how the research estimates evolve through the passage. The current displayed checkpoint matched both competing question–answer links in 0 of 8 controlled pairs. Its curves have no validated personal effort scale. Use the English wording and continuity marks above for document review. [Token experiment](research/adaptive_joint/results.md)

## Run the live demo

The portable demo uses an NVIDIA L4, a verified Qwen3.5-2B snapshot, and the selected joint-training checkpoint. From the repository root on the L4 host, activate the CUDA PyTorch environment and install the pinned Gradio and model-loading packages.

```sh
python -m pip install 'transformers==5.17.0' 'accelerate==1.13.0' 'gradio==6.17.3' 'wordfreq==3.1.1' 'nltk==3.10.4' 'scipy==1.17.1'

cogllm_snapshot=$(python - <<'PY'
import json
from pathlib import Path

from huggingface_hub import snapshot_download

spec = json.loads(Path("research/backbone_transfer/models.json").read_text())["models"]["qwen"]
print(snapshot_download(repo_id=spec["repo"], revision=spec["revision"], allow_patterns=spec["files"]))
PY
)

python -m research.token_readout.app \
  --runner research.token_readout.portable_runner:load \
  --model "$cogllm_snapshot" \
  --checkpoint research/results/adaptive-joint-fixed-retention-l4/guarded-fixed-1729/curve.pt
```

Open [the local Gradio address](http://127.0.0.1:7860/) on the host or through an authenticated local port forward. Each press of **Review draft** performs one backbone pass. Open the experimental panel to inspect the same stored result. The separate multilingual research interface is available with `--view signals`. [Pinned model revision and file hashes](../research/backbone_transfer/models.json) · [Selected checkpoint and verification](research/adaptive_joint/results.md#reproduction-and-artifacts)
