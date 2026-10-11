# Decisions

- 2026-10-11: Use frozen pretrained Qwen2.5-0.5B. Native pretraining and collaborative
  generation are outside phase one per user clarification.
- 2026-10-11: Compare the same four-token block decoder across J-lens readout,
  individual residual/attention layers and a configurable selected-layer set.
- 2026-10-11: Independent private repository, direct main commits, regular pushes,
  uv dependency management, and an AGENTS.md documentation workflow adapted from
  Age of Agents, as requested by the user.
- 2026-10-11: User requires distinct train/validation/test data and separate code,
  translation and prose reporting. Extend the exploration to pretrained Qwen
  checkpoints through approximately 1B parameters; do not treat cross-family
  comparisons as pure parameter-count scaling.
- 2026-10-11: Train one shared drafter on all three domains; domain categories
  define evaluation slices, not isolated domain training. Keep test sealed during
  budget selection. Prioritize data-size/optimization learning curves before
  the expanded conditioning and model-size sweep.
- 2026-10-11: After three-seed sizing and a 16k-update continuation, retain
  196,608 mixed-domain windows and a 4k-update/batch-128 screening budget for
  decoder diagnostics. Larger data consistently helps, but poor later-token
  generation and diminishing optimization returns do not justify further data
  expansion yet. This is a diagnostic choice, not sufficient-quality evidence.
- 2026-10-11: Proceed with the requested trained layer/J-lens conditioning study
  now, using the fixed diagnostic budget; do not defer the central experiment
  until generation quality is solved. Keep probe and decoder conclusions separate.
