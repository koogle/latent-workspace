"""Short, bounded DFlash baseline validation; shares dedicated project volumes."""

import json
import re
from pathlib import Path

import modal

from latent_workspace.budget import reserve

app = modal.App("latent-workspace-qwen-dflash")
WORKER_TIMEOUT = 900
cache = modal.Volume.from_name("latent-workspace-qwen-cache", create_if_missing=True)
runs = modal.Volume.from_name("latent-workspace-qwen-runs", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv==0.10.12")
    .add_local_file("pyproject.toml", "/project/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/project/uv.lock", copy=True)
    .run_commands("cd /project && uv sync --frozen --no-install-project")
    .env({"PATH": "/project/.venv/bin:/usr/local/bin:/usr/bin:/bin", "HF_HOME": "/cache/hf"})
    .add_local_python_source("latent_workspace")
)


@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=32768,
    timeout=WORKER_TIMEOUT,
    retries=0,
    max_containers=1,
    volumes={"/cache": cache, "/runs": runs},
)
def pilot(config: dict, run_id: str):
    import sys

    sys.path.insert(0, "/project/.venv/lib/python3.12/site-packages")
    from latent_workspace.dflash import DraftConfig
    from latent_workspace.dflash_train import run_dflash

    try:
        return run_dflash(DraftConfig(**config), f"/runs/{run_id}", on_checkpoint=runs.commit)
    finally:
        runs.commit()
        cache.commit()


@app.local_entrypoint()
def main(config: str = "configs/dflash-pilot.json", run_id: str = "dflash-pilot-01"):
    from latent_workspace.dflash import DraftConfig

    cfg = DraftConfig(**json.loads(Path(config).read_text())).validate()
    if cfg.max_seconds > WORKER_TIMEOUT - 120 or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
        raise ValueError("invalid run ID or insufficient cleanup margin")
    amount = reserve("artifacts/budget.json", run_id, "L40S", timeout_seconds=WORKER_TIMEOUT)
    print(f"Reserved ${amount:.2f} for a hard-bounded {WORKER_TIMEOUT}s L40S worker.")
    report = pilot.remote(cfg.to_dict(), run_id)
    Path(f"results/{run_id}.json").write_text(json.dumps(report, indent=2) + "\n")
