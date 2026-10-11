# Drafting baseline before representation experiments

User steering, 2026-10-11: defer hidden-state generation/J-space/workspace/layer
ablations until a useful diffusion drafter exists. Previous results remain evidence,
but do not establish a suitable drafter. The next experiment is one DFlash-style
Qwen2.5-0.5B baseline, not another architecture sweep.

## Reference and interpretation

Read Chen, Liang and Liu, [DFlash](https://arxiv.org/html/2602.06036v1), especially
§4.2–4.3, §5 and Appendix A.1; inspect the
[official implementation](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a).
The repository now also includes DFlash 2; this baseline follows classic DFlash,
not its later candidate selector/convolutions. MIT attribution is retained in
[references/dflash-LICENSE](../references/dflash-LICENSE).

The paper trains on target-generated responses (~800k samples in the main mixture),
random response anchors with all subsequent block positions masked, frozen shared
embeddings/output head, fixed multi-layer target features injected as K/V at every
draft layer, and one parallel denoising pass. It uses exponentially decayed CE,
AdamW at 6e-4, cosine decay, 4% warmup and six epochs; gamma=4 for block size 8.
Its training implementation packs independent blocks with restricted attention.

DFlash uses hidden **conditioning**, so deferring hidden-state **prediction** does
not mean removing this standard part of the baseline. There is no auxiliary
hidden-state generation/loss, J-lens, workspace compression, position-specific
output head, final-state output shortcut or learned layer selection here.

## Implemented baseline

- Pinned Qwen2.5-0.5B target; original tokenizer, embedding and vocabulary head frozen.
- Two Qwen2-style draft layers at target width, with fixed post-block features from
  layers 2 and 22 (one-based, uniformly spaced following the reference rule).
- Eight block positions: **one known target anchor plus seven new draft tokens**.
  The frozen `<|fim_pad|>` embedding represents masked inputs. No anchor loss.
- Full-token target feature memory, fused pointwise and injected into each draft
  layer's K/V; RoPE uses actual positions. Packed queries attend only to context
  strictly before their own anchor and to their own bidirectional block.
- Shared vocabulary readout, gamma=4 weighted CE, 6e-4 AdamW with cosine/warmup.
  Random anchors resampled on every update; mixed-domain balanced sampling.
- Cached target verification and incremental draft context K/V. Reject speculative
  suffixes, crop target cache and commit only verified tokens. Greedy decoding only.

This is a scaled Qwen2 adaptation, not a full reproduction of the paper: two draft
layers versus its usual five, 128-token prompts/64-token pilot responses rather
than 3k-token sequences, eight anchors per sampled sequence rather than 512, and
PyTorch SDPA masks rather than Flex Attention. No pretrained 0.5B draft checkpoint
was found in the official supported-model list. Small-target latency benefits are
unproven and must be measured, not inferred from larger-model results.

## Data contract

One prompt per source group. Prose and code are natural continuations; translation
prompts contain English source only. Human continuation/German reference labels
are not used to train the drafter. The frozen target generates greedy continuations
without repetition penalties. EOS is retained; responses too short for a complete
block and exact duplicate generated responses are excluded and counted.

Prose/WMT use official splits; code splits by repository hash. All new validation
and test groups exclude all historical groups; training excludes historical held-out
groups. Exact group/source/prompt duplicates are rejected. Across new partitions,
prompts with token-5-gram Jaccard >=0.8 are rejected. This is a prompt-level near-
duplicate screen, not a comprehensive whole-document contamination audit.

The manifest pins dataset and model revisions, group/source/prompt hashes and token
IDs. Teacher caches contain whole prompt+response sequences and causal features;
strict attention masks prevent clean future features from leaking into predictions.
Unique generated response tokens, eligible anchors and repeated supervised exposures
are separate quantities. The paper's 800k **responses** must not be confused with
our old counts of 132-token windows.

Pilot: 384 requested training prompts, 48 validation and 48 reserved test prompts,
equally allocated across prose/code/translation. Usable counts after response
filtering are reported. Test responses/features/metrics are not generated. This
pilot validates construction and optimization; it cannot determine the final data
requirement. Dataset-size scaling follows only after pipeline correctness.

## Predeclared evaluation and selection

Record step zero and every 100 updates through 1000. Report unweighted CE and
accuracy for each of seven NEW positions, consecutive matches **beyond the anchor**,
and full-seven-token acceptance on fixed held-out teacher trajectories. Training
curves use four diagnostic prompts/domain and are explicitly labeled a sample.
Select maximum validation accepted-new-token count, breaking ties by lower NLL.
No test-based selection. This does not assume seven independent correct tokens form
a correct sequence; only the consecutive prefix counts.

Separately run actual verified generation against ordinary cached target AR. Require
identical greedy output (including EOS/length limits). Initial timing uses one held-
out prompt/domain, 32-token cap, three alternating-order warmed repeats, same target
and precision, CUDA synchronization and end-to-end time including prefill. That is
only a pipeline latency check, not enough evidence for a general speed claim.

Before scaling: demonstrate a stable multi-token accepted prefix on held-out target
responses, inspect domain/position failure patterns and repetition, expand independent
validation coverage, and profile end-to-end latency. Do not resume representation
ablations merely because training loss falls or a tiny set can be memorized.

## Bounded operation

`uv run modal run modal_dflash.py --config configs/dflash-pilot.json --run-id dflash-pilot-01`

The worker has an actual 900-second timeout, no retries, one L40S, and a 720-second
internal deadline. Its $1.25 conservative reservation uses the same $5/hour rate
and $200 cap as historical jobs. Existing reservations are not refunded or reset.
The historical runner still reserves its full 3600-second timeout. Latest and best
checkpoints, optimizer/RNG state, prompt manifest and offline teacher cache persist
in the dedicated volume. Resume loading for this new trainer is not implemented yet;
saved state preserves the ingredients but is not claimed as a tested resume path.
