# CogLLM App Design

**The intended first users are authors and editors.** Before finalizing a draft, they paste in the text, find sentences worth checking first, inspect earlier passages associated with a question, and compare their own revisions. The main decision is whether to revise a passage and where the revision might help. Researchers can expand the same results to inspect calibration conditions, observation granularity, and model version.

The current Gradio token-curve preview accepts article text, highlights the original passage, and lets users inspect a position and its associated earlier text. The workflow below describes the broader design, including reader backgrounds and revision comparison. The implementation starts with text submission, sentence selection, and mapping to original offsets against verified readouts, so the interface and inference can be tested independently. Final acceptance calls for one sealed checkpoint and an end-to-end test from article input through token and sentence marks, a single forward pass, and evidence-supported trajectories for individual questions. [Current token-curve route](token_readout/transfer-results.md) · [Model and app acceptance criteria](execution-plan.md#完成條件)

<a id="輸入與主要操作"></a>

## Inputs and Main Actions

| Action | User input | App behavior |
| --- | --- | --- |
| Analyze an article | Article text is required. The first version focuses on plain-text paste, preserving paragraphs and punctuation. | Select **Analyze article**. The original text stays in the main pane and selectable locations for review appear beside it. |
| Specify reader background | Optionally provide a reader group or known concepts, such as “learned percentages and has yet to encounter Bayesian inference.” | An empty field uses the reference reader group calibrated for the selected endpoint, and the results name that group. Estimates use only model-validated fields and groups. |
| Compare versions | Use **Try revising this sentence** in the results sidebar to edit a copy, or paste a complete revision. | Analyze the revised article using the same background and response endpoint. Align and display the revision alongside later sentences whose estimates change so authors can see where the original difficulty eases and where additional difficulties appear. |

**Design proposal.** The first workflow is paste → locate → inspect earlier text → revise → compare. The sidebar lists a few sentences to examine first under the selected endpoint. Clicking a sentence scrolls to and selects it. Version comparison reuses the same marks and selection controls and follows verification of the first endpoint. Authors can begin with the reader background field empty. The app indicates when a selected endpoint lacks a calibrated reference reader group.

<a id="結果畫面"></a>

## Results View

| Area | What users see and do |
| --- | --- |
| Toolbar | Reference reader group, the displayed response endpoint, such as ease of comprehension or question reporting, and available sentence or phrase views. |
| Original text | Paragraphs retain their original layout. Underlines mark response locations. Selecting one outlines the previously read span associated with the question. The interface offers a selection control when several earlier spans qualify. It defaults to the finest verified granularity for that endpoint. |
| Selection details | The point the reader has reached, the span they may be asking about, the corresponding reader response, and its scale. Endpoints with a validated intensity scale show that scale and its direction. Event probabilities use full labels, such as “Estimated proportion of readers who report a reference question at the end of this sentence.” |
| Step-by-step reading | Previous and next controls advance one sentence at a time. Each stop shows the available text and response. Where paired data support it, the app lists the later observed intervals in which the same question became clearer. |
| Evidence and export | Expand to see the language, reader group, text scope, validation conditions, and evaluation results. Research exports preserve original character offsets, visibility conditions, endpoint, granularity, model version, and estimates for alignment and reruns. |

**Constructed example.** The first sentence reads “Group A uses a threshold of five points, and Group B uses eight.” The second reads “The next batch uses the same threshold.” Selecting “the same threshold” can show two candidate earlier spans, outlining the Group A and Group B thresholds separately. If the third sentence says “Here, the threshold is Group A’s five points,” that evidence appears only when the reader advances to sentence three. The example tests whether the interface supports separate actions for the response location, its proposed earlier referent, and the later evidence.

<a id="介面與模型的分工"></a>

## Interface and Model Responsibilities

CogLLM runs one causal forward pass per article and reader background and returns structured responses. Step-by-step reading retrieves the appropriate prefixes from that pass. The interface maps token positions to the original text, draws marks, and displays comparisons across versions or reader backgrounds. The data retain continuous estimates. Endpoint-specific display thresholds govern the underlines and highlights. Sentence-level results use sentence readouts where available, and any aggregation of token scores states its rule. Comparison graphics describe differences in the model estimates under two conditions. Separate study data establish whether those estimates correspond to human effects. [Model output and causal tests](mathematical-formulation.md#3-平行讀出反應位置與強度)

The existing baseline supports a demonstration of **sentence-level ease and self-rated comprehension when the entire English biomedical abstract is visible**. Retrospective sentence ratings use their own visibility condition. Step-by-step reading uses causal readouts. Feature descriptions follow each endpoint’s actual validation granularity and conditions. A release labels features still under evaluation as experimental. A curve finer than the granularity of the human evidence is an experimental output and should give the actual observation stops and evaluation scope. [Existing baseline](training-protocol.md#已完成的首輪基線) · [Human observations relevant to the interface](stage-1-observations.md)
