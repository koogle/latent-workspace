# Open work

Private repo: koogle/latent-workspace, main; direct commits and regular pushes authorized.

The original seven-condition WikiText pilot completed but generated repeated
punctuation. See results/pilot-l40s-01.md. It is not evidence of useful generation.

Current gate: [data sufficiency study](docs/data-budget.md). Implemented mixed
prose/code/translation training, disjoint grouped train/validation/test partitions,
nested data-size learning curves, actual masked-supervision counts and sealed-test
selection. 35 local tests and the updated Modal mixed-data smoke passed. The first
[learning curves](results/data-study-01.md) completed: more data improves NLL,
but generation remains degenerate. A sufficient budget has not been established.

Current follow-up: diagnose high absolute loss using 96-window memorization
and a zero-training repeated-Qwen-distribution baseline before scaling further.
The shared-output head failed memorization; position-specific outputs reached
99.5% training-token accuracy. A residual readout improved first-token quality.
Expansion attempt data-repaired-01 stopped before training on a uniqueness
audit. The parent pool had 75 duplicate windows and one train/validation overlap.
The corrected data-repaired-02 comparison completed with zero exact split
overlaps and unchanged held-out tensors: 12k/49k losses 6.00/5.74. Quality remains
poor. The 196k size and a final-layer-only control are running; saved-head
anchor inference is being evaluated separately. Forty-one local tests pass.

Remaining:
- Next proposed bracket: 12,288 versus 49,152 windows, 2k/4k/8k updates.
  Reuse frozen source revisions/held-out groups; do not regenerate partitions.
- Tiny-set memorization and multi-seed confirmation before a definitive budget.
- Layer-information scan and selected/all-layer ablations after useful training.
- Qwen3-0.6B-Base / Qwen3.5-0.8B-Base adapters need real-model verification.
- Larger validation sets, near-duplicate audit and final untouched-test evaluation.
- No collaborative proposal module or native backbone pretraining in this phase.
