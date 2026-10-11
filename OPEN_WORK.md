# Open work

Private repo: koogle/latent-workspace, main; direct commits and regular pushes authorized.

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

55 local tests pass, including exact dropout resume and compact feature caching.
All four controls completed actual frozen-Qwen GPU training. Source, ordered
subsets, split hashes, supervision counts and full-training diagnostics verified.
All Modal work completed; cumulative reservations $197/$200, not billed usage.
Only $3 remains under the current conservative cap, below another $5 L40S reservation.
Checkpoints: `/runs/small-layer-controls-01/{baseline,dropout,null,dropout-null}/`.
