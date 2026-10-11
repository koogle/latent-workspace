# Trained conditioning study

User direction: proceed with the actual layer/J-space diffusion study after data
sizing; poor absolute quality is a limitation to report, not a gate that prevents
this experiment. No collaborative generation or backbone pretraining.

## Fixed protocol

- Frozen pinned Qwen2.5-0.5B; same 196,608 mixed-domain windows and frozen
  validation/test tensors as data-large-01. Exact-window audit runs before training.
- Four-token targets, 128-token context, batch 128, seed 42, AdamW 3e-4,
  checkpoints 500/1k/2k/4k; same initialization and sampling within decoder ablations.
- Extract all 24 transformer blocks, one-based. Distinguish post-block residuals
  from attention outputs before residual addition. All-layer means all 24 layers.
- Decoder: 512 width, four layers, eight workspace slots, four tail positions,
  separate position outputs. All decoder conditions share final-state residual
  readout, so these measure incremental information beyond that common shortcut.
- Mixed prose/code/translation training; per-domain validation NLL, position
  accuracy, iterative generation and consecutive/full-block Qwen agreement.
- Validation selection is exploratory; test metrics remain sealed. No speed claim.

## Layer scan

Train equal-capacity nonlinear parallel probes at every residual and attention
output layer: 48 probes, 500 updates each, same initialization/sampler, last prefix
token only. Disable the final-state residual shortcut. Rank by mean validation
NLL at target offsets 2–4; record all offsets and domain slices. Select three
intermediate residual layers (exclude final) and three attention-output layers.
This measures accessible prediction under a finite-budget probe, not intrinsic
information or proof of a complete future answer. Probe ranking and decoder
performance can disagree. Repeated selection on 192 validation windows is noisy.

## Decoder comparisons

Train the same diffusion architecture conditioned on final residual only, the
best intermediate residual, three probe-selected residuals, all residuals, the
best attention output, three selected attention outputs, and all attention outputs.
Compare the selected residuals with their compressed J-lens readout and include
a parallel all-residual head as a non-denoising baseline. The parallel head has
the same declared modules, but token/time inputs are inactive; effective trained
capacity is therefore not perfectly matched.

J-lens calibration uses 100 eligible training prefixes only. Preserve indices,
transport matrices and source hashes in checkpoints. This is the documented
compressed top-25 J-lens adaptation, not the paper's full sparse J-space frame.
The primary paper/code references remain in README and references/papers.bib.
Calibration convergence beyond 100 prefixes is not established by this run.

## Execution

Configs use `conditioning-*.json`; reports use matching `conditioning-*-01` IDs.
The scan saves per-layer progress; decoder jobs save validation-best and final
resumable checkpoints to the dedicated Modal volume. L40S workers request 128 GiB
host RAM to hold all 24 layers without silently reducing the conditioning set;
GPU memory remains separate. Each job retains the one-hour timeout and $5 local
reservation, with a $200 aggregate allocation cap. Run a bounded J-lens smoke
before full comparisons. Larger-model comparisons are separate experiments and
must verify tokenizer/corpus compatibility before reusing token IDs.
