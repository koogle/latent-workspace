# Open work

Private repo: koogle/latent-workspace, main; direct commits and regular pushes authorized.

Current gate: [data sufficiency](docs/data-budget.md). Train one diffusion drafter
on mixed prose/code/translation and report domain validation separately. Test
metrics remain sealed. The exact-window audit removed a historical overlap;
clean studies preserve held-out tensors and have no exact split overlaps.

The original shared output collapsed all positions. Separate position readouts
passed tiny-set memorization (99.5% training accuracy). Combined with a residual
readout, the clean 12k/49k/196k study improves validation, but generation is poor.
At batch 128, seed 42, best NLL is 5.511 for 49k (2k updates) and 5.264 for 196k
(4k updates). The latter agrees with Qwen for only 1.06 consecutive tokens / 4.
See [curves](results/data-scaling.png). This is not a sufficient quality result.

Running: exact checkpoint continuation of 196k to 8k/16k updates, plus seeds
43/44 for the 49k/196k comparison through 4k updates. Forty-four local tests pass,
including exact next-update study continuation. Real GPU continuation is pending.

Remaining:
- Complete repeated-seed size study and continuation; retain validation-best heads.
- Diagnose weak later-token generalization; final-layer-only conditioning, known
  first-token anchoring, left-to-right unmasking and input normalization did not
  resolve it. Human corpus labels are still used, not Qwen distillation targets.
- Larger validation set and near-duplicate audit before a definitive data budget.
- Layer-information scan and selected/all-layer ablations after useful training.
- Qwen3-0.6B-Base / Qwen3.5-0.8B-Base adapters need real-model verification.
- Freeze choices before evaluating the untouched test set.
- No collaborative module or native backbone pretraining in this phase.
