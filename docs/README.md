# CogLLM documentation

[Download the illustrated research report](../output/pdf/CogLLM-reading-study.pdf), [read the empirical study and release decision](../research/reading_validation/manuscript.md) or [inspect the experimental evidence by reading behavior](../research/reading_validation/validation-report.md).

## Start with the demo

- [Review an English draft](try-cogllm.md) to see sentence ease, difficult wording, continuity alerts and optional character introductions across the whole passage.
- [View a science passage](../research/results/reading-validation-writer/review.html) or a [narrative summary](../research/results/reading-validation-writer/narrative.html) with marks captured from the L4 app.
- [See readers’ next-word expectations change across a passage](../research/results/reading-validation-writer/word-expectations.html), alongside every word, probability and measured update.
- [Gradio interfaces](research/gradio_app/README.md) describes the current preview and the separate sentence- and word-level research interface.
- [App concept](research/app-concept.md) describes the intended workflow for writers and editors.

## Read the evidence

- [Results and open questions](research/findings-and-gaps.md) summarizes which human observations the model reproduced and which remain open.
- [Token readouts across Qwen, LFM3, and Gemma](research/token_readout/transfer-results.md) covers first reports, evidence spans, and early versus late resolution.
- [Joint training results](research/adaptive_joint/results.md) compares adaptive and fixed source weights with and without retention constraints.
- [Expository writing experiments](research/expository_lora/results.md) compares sentence readouts trained on original and rewritten passages, including a test with and without headings.
- [Reading studies and available data](research/reading-evidence.md) catalogs the human evidence used to plan the evaluations.

## Reproduce the research

- [Training protocol](research/training-protocol.md) defines the tasks, evaluation criteria, and experimental setup.
- [Execution plan](research/execution-plan.md) records the stages, completed work, and remaining acceptance criteria.
- [Mathematical formulation](research/mathematical-formulation.md) specifies the causal pass and the position-specific readouts.
- [Observation model](research/observation-model.md) describes which human observations can support each kind of readout.
- [Joint training plan](research/joint-training-plan.md) and [observation-specific losses](research/union_training/observation-protocol.md) describe how different datasets supervise a shared model.
- [Research document catalog](catalog.md) lists the full English collection.

The English collection links each document to its original research record under `research/`. Some of those original files are part of sealed experiment inputs, which depend on their exact bytes. Historical notes use machine-assisted English translations. Consult the linked original and run artifacts for exact protocol wording and numerical decisions. Quoted stimuli, code, and literal source identifiers may retain their original language.
