# Experiment protocol

Read before modifying representations, datasets, training or evaluation.

## Scope (2026-10-11)

Frozen `Qwen/Qwen2.5-0.5B`, four target tokens, configurable 8/16-token blocks.
No collaborative proposer or from-scratch backbone. The identical denoising head
conditions on: one post-transformer residual state, a selected set of residual
states, one self-attention output (after output projection, before residual add),
a selected set of attention outputs, or a sparse J-lens vocabulary readout.
A parallel all-masked head is the context-only reference.

Layer indices are **one-based transformer blocks**, captured with forward hooks.
Post-block residuals are before the model's final RMSNorm, including layer 24.
The selected layer list must end at the final model layer. Selecting an earlier
layer for conditioning still executes the whole backbone; it is not a truncated
backbone speed claim. Every representation uses only already committed tokens.

## J-space provenance and adaptation

[Anthropic paper](https://transformer-circuits.pub/2026/workspace/index.html),
[arXiv](https://arxiv.org/abs/2607.15495), and
[reference code](https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e)
are retained for reproducibility. `jspace.py` independently implements the
reference's batched Jacobian estimator: sum cotangents over valid future output
positions, average gradients over source positions, then average training-prefix
Jacobians. Exclude the first 16 and final positions. Matrices map raw post-block
residuals to raw final-block residuals; final layer's map is identity.

For conditioning, apply the transport, Qwen's RMSNorm and frozen unembedding;
keep the top 25 vocabulary scores, softmax within that set, and average their
frozen token output vectors. This is **compressed J-lens readout conditioning**,
not a full sparse nonnegative reconstruction of the paper's J-space frame. The
smoke calibration uses very few prompts and is unsuitable for scientific claims;
use at least 100 and test convergence before interpreting J-space quality.

## Data and fairness

Use WikiText official train/validation splits, group lines into articles, remove
exact held-out article duplicates from training, and window within articles.
No target tokens enter context extraction or Jacobian calibration. Model and
corpus commit hashes, article hashes and actual window counts are saved.
WikiText continuation is a plumbing/prediction task, not an answer-reasoning test.

All variants use the same sampled training windows, batch size and step budget;
conditioning ablations share architecture and trainable parameter count. The
loss samples a nonempty mask count uniformly and averages cross-entropy on only
masked positions per example. This is a simple masked denoising objective, **not**
the [BD3-LM](https://arxiv.org/abs/2503.09573) ELBO or its official reproduction.

Report held-out all-masked NLL/accuracy by target position, exact block accuracy,
paired per-example losses, parameter counts, exposures, elapsed time and memory.
All-masked NLL is a comparable conditional prediction metric, **not** normalized
sequence likelihood for the iterative sampler. Teacher-forced Qwen sees earlier
true target tokens and is labeled a distinct reference. Validation is used for
exploration; a separate untouched test split is needed for confirmatory claims.

Warm latency includes prefix encoding and committed-block cache updates. Do not
infer efficiency without matched output quality. Cached features accelerate
training; those savings do not imply faster generation. Repeat promising runs
across seeds and report paired differences before expanding the budget.

`selected_layers` optionally restricts the `multi`, `attention_multi` and `jspace`
conditions to a subset of extracted `layers`. Empty means all extracted layers.
Generation metrics also report greedy iterative block accuracy separately from
all-masked one-shot accuracy. Generation latency computes J-lens readouts only
for the J-space condition, not for raw-state comparisons.
