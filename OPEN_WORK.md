# Open work

Private repo: koogle/latent-workspace, main; direct commits and regular pushes authorized.

Current work: [trained conditioning study](docs/conditioning-study.md). Train one diffusion drafter
on mixed prose/code/translation and report domain validation separately. Test
metrics remain sealed. The exact-window audit removed a historical overlap;
clean studies preserve held-out tensors and have no exact split overlaps.

The original shared output collapsed all positions. Separate position readouts
passed tiny-set memorization (99.5% training accuracy). Combined with a residual
readout, the clean 12k/49k/196k study improves validation, but generation is poor.
At batch 128, seed 42, best NLL is 5.511 for 49k (2k updates) and 5.264 for 196k
(4k updates). The latter agrees with Qwen for only 1.06 consecutive tokens / 4.
See [curves](results/data-scaling.png). This is not a sufficient quality result.

Completed: seeds 42/43/44 all favor 196k over 49k; the mean paired best-checkpoint
NLL difference through 4k updates is -0.272. Exact GPU continuation reached 16k
updates: NLL 5.201, consecutive Qwen agreement 1.099/4, full-block agreement 4.7%.
All runs completed within bounds. Forty-four local tests pass. Full reports and
[updated curves](results/data-scaling.md) are retained; test metrics remain sealed.

Recommendation: retain 196,608 mixed-domain windows as the fixed diagnostic
pool, use 4k updates at batch 128 for initial comparisons, and extend promising
changes selectively. This is not enough evidence to call the data sufficient
for useful generation. Prioritize the later-token bottleneck before more scaling.

The conditioning and Qwen3-0.6B smokes completed on Modal; 50 local tests pass.
The 48-probe scan completed: residual subset 18/20/17, attention subset 20/16/15.
Final/all-residual Qwen2.5 decoder runs completed: NLL 5.390/5.259. Selected,
single-layer, attention and J-lens decoders are training, plus matched final/all
Qwen3-0.6B decoders. These are actual 4k-update mixed-domain training comparisons.
Continue to completion, freeze validation-selected checkpoints, then evaluate
chosen models/controls on test once under a recorded protocol.

Remaining:
- Diagnose weak later-token generalization; final-layer-only conditioning, known
  first-token anchoring, left-to-right unmasking and input normalization did not
  resolve it. Human corpus labels are still used, not Qwen distillation targets.
- Larger validation set and near-duplicate audit before a definitive data budget.
- Complete layer-information scan and trained selected/all-layer/J-lens ablations.
- Qwen3-0.6B-Base / Qwen3.5-0.8B-Base adapters need real-model verification.
- Freeze choices before evaluating the untouched test set.
- No collaborative module or native backbone pretraining in this phase.
