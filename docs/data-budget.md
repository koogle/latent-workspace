# Drafter data budget: research and first learning curves

## What is known

The first WikiText pilot supplied each decoder only 24,000 target-token slots
(750 updates × 8 examples × 4 tokens). Context exposures are not output labels;
random masking further reduces the number of supervised positions. Its repeated
punctuation and high held-out loss do not establish a representation limitation.
The readout now includes Qwen's frozen final normalization, with a regression
check against actual Qwen logits. Historical results used the earlier readout.

[DFlash](https://arxiv.org/html/2602.06036v2) is a close architectural reference:
it conditions a lightweight diffusion drafter on multiple target hidden features,
shares frozen embeddings/output weights, and uses target verification. Its
training distribution and verification objective differ from our standalone
four-token generator. Published sequence counts cannot be converted directly into
our number of supervised four-token blocks. We therefore measure learning curves
before choosing a full training budget or attempting a broad conditioning sweep.

## Controlled initial experiment

Use frozen Qwen2.5-0.5B, one fixed diffusion decoder and selected residual layers.
Train every model on an equal mix of prose, code and English-to-German translation.
Nested training subsets contain 768, 3,072 and 12,288 unique four-token windows.
Each starts from identical initialization and gets 100/500/1,000/2,000 update
checkpoints, batch size 16. At the final checkpoint each sees 128,000 target slots;
record the actual masked positions receiving loss separately. Compare dataset
sizes at equal updates, and optimization progress within each dataset size.

The subsets are exploratory brackets, not a claim that 12,288 windows suffice.
Validation has 64 windows per domain; test has another 64 per domain and remains
sealed. These are small screening samples, unsuitable for definitive significance
claims. Split prose by official article partition, code by repository hash, and
translation by official partition with source deduplication. Quarantine normalized
exact duplicates; near-duplicate and pretrained-model contamination are unresolved.

Report per-domain all-masked token NLL, per-position accuracy, iterative block
accuracy, and a balanced training diagnostic. Train on human corpus targets in
this first study. Target-generated continuations/distillation should be a separate
controlled follow-up if agreement with Qwen is the intended drafting objective.
Do not interpret ground-truth agreement as speculative acceptance rate.

## Decisions after measurement

- If training and validation loss still improve with updates, extend optimization
  before concluding that unique data is insufficient.
- If more unique data helps at equal updates, expand the largest bracket by 4×.
- If training improves while validation stalls, prioritize diversity/data volume.
- If even training loss stalls at poor quality, test tiny-set memorization,
  readout calibration, learning rate and conditioning before buying more data.
- Once curves approach a practical plateau, repeat the candidate budget with
  three seeds and larger validation partitions. Select a budget using validation,
  freeze the protocol, then evaluate untouched test once.

Layer scans, all-layer attention, J-lens ablations and larger Qwen models remain
next-stage experiments. This first study does not establish minimum data needs,
quality parity, lossless decoding or a speed advantage.

## First measured outcome

[The completed L40S study](../results/data-study-01.md) shows smaller-set
overfitting and better validation NLL with more unique data. It does not yet
produce useful generation or establish a sufficient budget. Next: a controlled
12,288/49,152-window comparison with longer optimization, preserving this study's
held-out groups and revisions. This recommendation is not an executed result.

## Follow-up diagnostic gate

The high absolute loss and repetitive outputs require a decoder/training diagnosis
before attributing the problem solely to dataset size. A 96-window memorization
run uses the original training pool and frozen validation/test tensors. It records
all-masked training accuracy as well as loss. A zero-training prefix-only baseline
repeats Qwen's next-token distribution at every block position; its first position
is especially useful for detecting loss of pretrained predictive information.

`study_parent` reloads the parent corpus and pins its model/source revisions.
Expansion retains every original training row, appends distinct windows, excludes
held-out groups/content hashes and carries validation/test tensors unchanged.
The original full-pool training order is retained as the nested subset's prefix.
The proposed 12k/49k, 8k-update expansion is configured but gated on diagnosis.

The original 96-window memorization run reached about 1.71 training NLL but only
26% all-masked training-token accuracy after 2,000 updates. This fails a practical
memorization sanity check. Qwen's first-token validation NLL is 1.15–2.69 across
domains, versus 6.77–7.37 for the original 12k-window drafter. Later-token losses
from repeating Qwen's distribution are poor, as expected; it is not a solution.

Two explicit diagnostic options are under evaluation before more data spending:
`residual_readout` starts at the frozen final residual state and learns a zero-
initialized correction; this adds final-state conditioning and changes readout
initialization/normalization, so it is not a pure intermediate-layer ablation.
`position_readout` uses a separate output projection at each block position to
check whether shared decoding loses position information. It increases trainable
parameters by about 1.38M for four tokens. Neither is a validated fix yet.

The position-specific 96-window run achieved 99.2% training-token accuracy at
500 updates and 99.5% at 2,000 (NLL 0.035). Validation is deliberately terrible
for this overfit tiny set; this establishes trainability, not generalization.
The residual-only 12k-window run reached validation NLL 5.96 at 4k updates and
preserved much of Qwen's first-token quality, but later positions remained weak.
The combined readout is now being compared at 12k/49k windows and up to 8k updates.

Future evaluations also record greedy prefix agreement with Qwen along the
proposed block: only consecutive matching tokens before the first mismatch count.
This uses no reference continuation labels and is separate from ground-truth
accuracy. It is an evaluation diagnostic, not a deployed verifier, speculative
sampling correctness proof or measured end-to-end speed improvement.

## Window-level audit correction

The first expansion stopped before training because full-document deduplication
left repeated context/target windows. The original 12,288-row pool contains 12,213
distinct windows (75 duplicates), with one exact training/validation overlap and
no training/test or validation/test overlaps. Historical validation results are
exploratory and are marked accordingly. The corrected comparison preserves all
held-out tensors but necessarily changes the training control: remove duplicates
and the overlapping window, then fill the smaller balanced subset from clean data.
Both new sizes share this cleaned nested pool. The corrected collector reserves
held-out windows first and checks window-level duplicates in addition to groups
and document/source hashes. Near-duplicate and pretraining contamination remain
outside this exact-match audit.
