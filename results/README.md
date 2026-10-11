# Experiment evidence

## Verified initial checks (2026-10-11)

- `smoke-l40s-01.json`: seven variants, 10 updates each, 64 training windows,
  16 held-out windows, one J-lens calibration prefix, 3.66M trainable parameters.
  About 51.8 seconds elapsed and 1.45 GB peak allocated GPU memory. Validates the
  original pipeline; **not a useful model-quality comparison**. This older run's
  generation timing also computed J-lens features for raw-state variants, so do
  not use those latency columns as conditioning-specific performance evidence.
- `profile-l40s-01.json`: 21.63M parameters, 25 steps, 6.64 steps/sec, 1.46 GB peak.
  Warm block decoding 7.38 ms/token, AR 12.14 ms/token on the same prompt.
- `profile-h100-01.json`: identical configuration, 5.47 steps/sec, 1.51 GB peak.
  Warm block decoding 12.01 ms/token, AR 16.13 ms/token.

Both profiles are tiny single-container observations with timing noise and Python
orchestration overhead. They support selecting L40S for this pilot, not a general
hardware ranking or matched-quality speedup. The decoder is poorly trained here.

## Pilot in progress

`pilot-l40s-01`: 21.63M-parameter decoder, 100 training-prefix J-lens calibration,
selected layers 6/12/18, seven conditions, 750 updates × batch 8 each, 256-token
contexts, four-token targets. Planned exposures: 1,536,000 context tokens and
24,000 target tokens per condition; 10,752,000 and 168,000 across seven conditions.
These are repeated exposures; token windows are sampled from 4096 training
examples. Pilot results and text samples will be added after completion.

Use `uv run python scripts/summarize.py results/RUN.json --output results/RUN.md`
for a readable table. Raw JSON retains per-position and per-example metrics.
