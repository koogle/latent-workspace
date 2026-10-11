"""Local conservative reservation ledger; not a Modal account billing limit."""

import json
import math
from pathlib import Path

CAP_USD = 200.0
# Deliberately conservative allocation rates, not a quotation of Modal prices.
HOURLY_RESERVATION = {"L40S": 5.0, "H100": 12.0}
TIMEOUT_SECONDS = 3600


def reserve(path, run_id, gpu):
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads(path.read_text()) if path.exists() else {"reservations": []}
        amount = HOURLY_RESERVATION[gpu] * TIMEOUT_SECONDS / 3600
        spent = sum(item["usd"] for item in ledger["reservations"])
        if not math.isfinite(spent) or spent + amount > CAP_USD:
            raise ValueError("$200 allocation cap reached; no job submitted")
        if any(item["run_id"] == run_id for item in ledger["reservations"]):
            raise ValueError("run ID already reserved; use a new ID, including for resumes")
        ledger["reservations"].append({"run_id": run_id, "gpu": gpu, "usd": amount})
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(ledger, indent=2))
        temporary.replace(path)
    return amount
