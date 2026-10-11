# Open work

## Current state

Implementation in progress on `main`. Private remote:
https://github.com/koogle/latent-workspace. Frozen Qwen extraction, block decoder,
Jacobian calibration, dataset splitting, evaluation and checkpoint code exist.
These are not yet GPU-verified.

## Next

- Finish bounded Modal entrypoint and configurations.
- Validate leakage/freeze invariants, J-lens estimator and sampling.
- Run an L40S pilot, inspect metrics and preserve measured evidence.
- Profile before a larger token-exposure run. No quality or speedup conclusion yet.
