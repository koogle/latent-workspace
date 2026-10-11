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

Fifty local tests and both real-model GPU smokes pass. All Modal jobs completed.
Cumulative conservative reservations: $192/$200, not actual billed usage. The
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

Follow-up in progress: `small-layer-controls-01`, one bounded L40S suite, 96/768/3072
windows and four all-layer diffusion controls. Full train curves, dense checkpoints;
no new test evaluation. Local checks: 53 tests pass. Reservations are now $197/$200
(conservative allocations, not billed cost). Results are pending; see
[protocol](docs/small-data-controls.md).
