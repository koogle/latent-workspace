"""uv run --with matplotlib python scripts/plot_study.py results/RUN.json"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = Path(sys.argv[1])
report = json.loads(path.read_text())
report = report.get("report", report)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for size, curve in report["curves"].items():
    rows = curve["checkpoints"]
    steps = [r["step"] for r in rows]
    val = [r["macro_validation_nll"] for r in rows]
    train = [
        sum(r["training_sample_nll_by_position"]) / len(r["training_sample_nll_by_position"])
        for r in rows
    ]
    line = axes[0].plot(steps, val, marker="o", label=f"{int(size):,} blocks")[0]
    axes[1].plot(steps, train, marker="o", color=line.get_color(), label=f"{int(size):,} blocks")
for ax, title in zip(axes, ("Mixed-domain validation", "Balanced training diagnostic")):
    ax.set(xlabel="Optimizer updates", ylabel="All-masked token NLL", title=title)
    ax.grid(alpha=0.2)
    ax.legend()
fig.suptitle("Initial data-budget screening · Qwen2.5-0.5B · one seed · test sealed")
fig.savefig(path.with_suffix(".png"), dpi=160)
fig.savefig(path.with_suffix(".svg"))

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True, sharey=True)
for ax, domain in zip(axes, ("prose", "code", "translation")):
    for size, curve in report["curves"].items():
        rows = curve["checkpoints"]
        ax.plot(
            [r["step"] for r in rows],
            [r["validation_by_domain"][domain]["nll"] for r in rows],
            marker="o",
            label=f"{int(size):,} blocks",
        )
    ax.set(xlabel="Optimizer updates", title=domain.capitalize())
    ax.grid(alpha=0.2)
axes[0].set_ylabel("Validation all-masked token NLL")
axes[-1].legend()
fig.suptitle("One mixed-domain drafter · validation slices · one seed · test sealed")
fig.savefig(path.with_name(path.stem + "-domains.png"), dpi=160)
fig.savefig(path.with_name(path.stem + "-domains.svg"))
