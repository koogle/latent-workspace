# Open work

## Current state

Implementation in progress on `main`. Private remote:
https://github.com/koogle/latent-workspace. Frozen Qwen extraction, block decoder,
Jacobian calibration, dataset splitting, evaluation and checkpoint code exist.
The 16 local CPU tests pass, including tiny-Qwen hook/cache equivalence and
Jacobian versus direct-autograd agreement. Ruff passes. A bounded L40S smoke
job is building/running; no GPU result is claimed yet.

## Next

- Run an L40S pilot, inspect metrics and preserve measured evidence.
- Profile before a larger token-exposure run. No quality or speedup conclusion yet.
