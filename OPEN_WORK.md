# Open work

Private repo: koogle/latent-workspace, main; direct commits and regular pushes authorized.

The original seven-condition WikiText pilot completed but generated repeated
punctuation. See results/pilot-l40s-01.md. It is not evidence of useful generation.

Current gate: [data sufficiency study](docs/data-budget.md). Implemented mixed
prose/code/translation training, disjoint grouped train/validation/test partitions,
nested data-size learning curves, actual masked-supervision counts and sealed-test
selection. Local checks pass; updated Modal mixed-data smoke and learning curves
must be measured before selecting a training budget.

Remaining:
- Run and analyze the learning curves; extend brackets/optimization if warranted.
- Tiny-set memorization and multi-seed confirmation before a definitive budget.
- Layer-information scan and selected/all-layer ablations after useful training.
- Qwen3-0.6B-Base / Qwen3.5-0.8B-Base adapters need real-model verification.
- Larger validation sets, near-duplicate audit and final untouched-test evaluation.
- No collaborative proposal module or native backbone pretraining in this phase.
