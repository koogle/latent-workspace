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
