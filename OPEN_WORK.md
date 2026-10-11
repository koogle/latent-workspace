# Open work

## Current state

Pushed to private `koogle/latent-workspace` on main (direct commits authorized).
Frozen Qwen extraction, four-token denoising, J-lens conditioning, selected raw
residual/attention conditions, data splits, checkpoints and uv/Modal runner work.
The first seven-condition L40S smoke and 21.63M-parameter L40S/H100 profiles
completed. See results/README.md for measurements and limits.

`pilot-l40s-01` is running on L40S: 100-prefix lens fit and seven conditions,
750 updates each, 10.75M total context-token exposures if completed. No trained
quality result is claimed before it finishes.

## Next

- Finish pilot, inspect errors/metrics/samples, and push measured evidence.
- Run final validation and check checkpoint retrieval.
- Multi-seed, untouched-test and reasoning-task evaluation remain future research.
- No collaborative generation or native pretraining in this phase.
