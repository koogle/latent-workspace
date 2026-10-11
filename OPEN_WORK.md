# Open work

Private repo: koogle/latent-workspace, main; direct commits and regular pushes authorized.

Current priority: [DFlash-style Qwen drafting baseline](docs/dflash-baseline.md).
Representation experiments are deferred. The aligned baseline has one known anchor
plus seven masked proposals, target-generated labels, fixed feature K/V injection,
random-anchor packed training and cached greedy verification.

Executed: 380 usable train responses, 47 fresh validation responses and 48 reserved
test prompts (no test responses/metrics). The initial 1000-update fit reached only
0.145 accepted NEW tokens / 7; training-sample NLL 7.747, validation 7.820. It is
not useful drafting. The original BF16 rollout check failed. A fixed-input GPU
replay isolated a BF16 AR/block argmax flip (0.0625 top-two margin, 0.25 logit
change); FP32 arithmetic matched all three diagnostic prompts. The CLI and future
rollout checks now use FP32 on upcast stored BF16 weights. BF16 parity and any
speedup remain unestablished.

`dflash-fit-01` is running: continue the SAME model/AdamW/RNG and exact teacher
cache for up to 20000 additional updates (goal 21000), with a declared new cosine
cycle. No architecture, feature-layer, mask-policy or data changes. This addresses
underfitting before further study. At update 11000 the 12-prompt training diagnostic
reached 6.015 consecutive new matches / 7 and NLL 0.360, versus validation 0.185
and NLL 9.078. The emerging finding is memorization without useful generalization;
final results and an all-training-response audit are pending. A readout-scale audit found 1.13 versus target
7.12 RMS, but its analytical CE bound is essentially zero: it does NOT establish
a loss floor or justify calling normalization the cause.

69 local tests pass, including exact next-update restoration of the new trainer's
optimizer/sampler. The real checkpoint CLI ran on CPU. Original pilot and diagnostic
artifacts are retained; new continuation results are pending. `dflash-pilot-01`
failed during remote import and was explicitly stopped; its reservation remains.
The user-authorized original allocation range was $200–$500. The initial conservative
$200 ceiling has now been raised to that original $500 upper bound for the drafting-
first continuation; cumulative reservations are $205, not a confirmed bill.

Pending: finish/interpret the optimization continuation; then determine the target-
response data scale from fitting versus held-out acceptance. Do not claim a working
fast drafter or resume layer/J-space experiments based only on decreasing loss.

The [trained conditioning study is complete](results/conditioning-findings.md).
Executed: 96 residual/attention probes across two seeds; nine Qwen2.5-0.5B decoder
conditions and two Qwen3-0.6B conditions at 4k updates each; one frozen-checkpoint
test suite across all 11 heads. Every decoder used the same 196,608 mixed-domain
training windows. Full corpus identity checks and source hashes are retained.

Findings: all-layer residual conditioning improves test NLL over final-only for
both Qwen families. The compressed J-lens loses to matching raw states. Diffusion
does not beat the parallel baseline here. The all-layer Qwen2.5 diffusion head
matches only 1.141 consecutive Qwen tokens / 4, with 1.6% full-block agreement.
Absolute generation quality remains poor. The test split has now been evaluated;
it must not be described as sealed or untouched for subsequent adaptive work.

At completion of the original study, 50 local tests and both real-model GPU
smokes passed, with $192/$200 reserved (not actual billed usage). The
all-layer inference checkpoint is downloaded and checksum-verified against the
tested head; other best/final checkpoints remain in the dedicated Modal volume.

Remaining:
- Improve later-token generalization using controlled training/decoder changes;
  human corpus labels were used, not Qwen distillation targets.
- Larger fresh validation/test partitions and near-duplicate audit before further
  adaptive experiments. Current slices are only 64 windows/domain, Python code,
  WikiText prose and English-to-German translation.
- Repeat trained decoder comparisons across seeds; only the probes and earlier
  dataset-size comparisons have repeated seeds so far.
- Qwen3.5-0.8B requires a separate tokenizer/data preparation path and real-model
  adapter verification. It was not trained in this study.
- More J-lens calibration/convergence and readout controls before broader claims
  about the paper's J-space representation.
- No collaborative module or native backbone pretraining in this phase.

Follow-up complete: [small-data layer controls](results/small-layer-controls.md).
All 12 fits (96/768/3072 windows × baseline/dropout/null/both) reached 3200 updates,
batch 32, on one bounded L40S job. Baseline final training accuracy:
100%/98.9%/94.3%; every size overfits validation. At 3072 windows, best validation
NLL is 7.264 baseline, 7.333 dropout, 7.262 null, 7.314 both. No clear gain from
these controls; one seed and a reused small validation slice are exploratory.
Tiny-set fitting works, but this does not establish enough data for a useful drafter
or convergence at 3072 windows. Retain larger-data work as the generalization
baseline; do not replace it with the tiny memorization sets.

At the small-control checkpoint, 55 local tests passed, including exact dropout
resume and compact feature caching.
All four controls completed actual frozen-Qwen GPU training. Source, ordered
subsets, split hashes, supervision counts and full-training diagnostics verified.
That completed checkpoint had $197 reserved against the initial $200 allocation.
See the current drafting-first status above for subsequent reservations and cap.
Checkpoints: `/runs/small-layer-controls-01/{baseline,dropout,null,dropout-null}/`.
