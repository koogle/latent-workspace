# Operations

Use `uv sync --extra dev --extra modal` and `uv run ...` from repository root.
Run tests, Ruff and diff checks before a verified push. The research branch is
`main`; push it regularly; direct main commits are authorized by the user.

Modal app/volumes are dedicated to this project. Use configured SDK credentials;
do not copy credentials into files or Git. The first remote run must be a bounded
smoke/profile job before larger training.

## Managed-cloud connection finding (2026-10-11)

Modal 1.6.1 needs its `api-proxy-support` extra here. grpclib explicitly loads
certifi's bundle, so SSL_CERT_FILE alone did not establish trust for the session
proxy. Appending the environment's trusted public CA certificate to the isolated
Python environment's certifi bundle fixed `modal app list`. Keep TLS verification
and inherited proxy settings enabled. Do not publish the environment-specific CA.

GPU runs, budgets and retrieval commands will be documented once verified.

## Bounded runs

```bash
uv run modal run modal_app.py --config configs/smoke.json --run-id smoke-l40s-01
uv run modal run modal_app.py --config configs/profile.json --run-id profile-l40s --gpu L40S
uv run modal run modal_app.py --config configs/profile.json --run-id profile-h100 --gpu H100
uv run modal volume get latent-workspace-qwen-runs /smoke-l40s-01 artifacts/smoke-l40s-01
```

Each run has a 3600-second remote timeout, no automatic retry, and one GPU.
`max_seconds` stops the training loop earlier and leaves cleanup time. The local
`artifacts/budget.json` ledger reserves the full timeout at conservative allocation
rates ($5/h L40S, $12/h H100) against a $200 cap. Failed launches retain reservations.
These are allocation assumptions, **not live prices or an account-wide billing
cap**; storage/build costs and runs from other machines are not enforced by this
ledger. Keep the ledger when moving machines and check actual Modal usage. No
long pilot is launched automatically. Resumes need a new reservation/run ID and
`--resume-from ORIGINAL_ID`, with the original config; they extend compute time.

The locked uv environment is built inside Modal. Source is mounted independently,
so code-only changes reuse the dependency image. Results JSON is returned to the
local results/ directory; model heads, optimizer/RNG state, configuration, dataset
hashes and model revisions live on the dedicated runs volume. Model and corpus
caches live on a separate cache volume. Checkpoints exclude frozen Qwen weights.
