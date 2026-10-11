# Latent workspace block generation

Research prototype: freeze **Qwen2.5-0.5B**, train a small decoder to generate
four tokens jointly, and compare conditioning on an Anthropic-style J-lens
readout, individual transformer/attention blocks, and selected combinations.
No backbone pretraining or collaborative proposal module in this first phase.

## Development

```bash
uv sync --extra dev --extra modal
uv run pytest
uv run ruff check .
uv run modal run modal_app.py --config configs/smoke.json --run-id smoke-l40s
```

Modal uses its configured account, a dedicated app and volumes. Read
[operations](docs/operations.md) before launching; checkpoints and measured
results are persisted. A smoke test establishes plumbing, not model quality.

## Research references

- Gurnee et al. (2026), [Verbalizable Representations Form a Global Workspace in
  Language Models](https://transformer-circuits.pub/2026/workspace/index.html),
  [arXiv:2607.15495](https://arxiv.org/abs/2607.15495).
- Anthropic, [Jacobian lens reference implementation](https://github.com/anthropics/jacobian-lens),
  reference commit `581d398613e5602a5af361e1c34d3a92ea82ba8e` (Apache-2.0).
- [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language
  Models](https://arxiv.org/abs/2503.09573) (BD3-LM): a research baseline to
  reproduce later, **not** an implementation or quality claim for this code.
- [Qwen2.5-0.5B model card](https://huggingface.co/Qwen/Qwen2.5-0.5B).

A [BibTeX bibliography](references/papers.bib) and [measured run evidence](results/README.md)
are retained in the repository.

See [experiment protocol](docs/experiment.md), [knowledge index](docs/knowledge/INDEX.md),
and [open work](OPEN_WORK.md). All dependencies and files belong to this project;
it has no connection to the Age of Agents game.

## Current gate: training-data sufficiency

The initial WikiText pilot did not produce useful generation. Before scaling,
run the [mixed-domain learning-curve study](docs/data-budget.md): nested dataset
sizes at matched update budgets, per-domain validation, and a sealed test set.

```bash
uv run modal run modal_app.py --config configs/mixed-smoke.json --run-id mixed-smoke-01
uv run modal run modal_app.py --config configs/data-study.json --run-id data-study-01
```

Additional reference: Chen et al., [DFlash: Block Diffusion for Flash Speculative
Decoding](https://arxiv.org/abs/2602.06036), [official code](https://github.com/z-lab/dflash).
This prototype does not reproduce its verifier or lossless speedup claims.

The [first measured data-budget curves](results/data-study-01.md) are available:
more data improved validation loss, but sample generations remain degenerate.
No sufficient training budget or useful drafter quality is established yet.


The [clean follow-up curves](results/data-scaling.png) now compare up to 196,608
mixed-domain windows. Best seed-42 validation NLL is 5.264, but consecutive Qwen
agreement is only 1.06/4 tokens: useful block generation is still unestablished.
See [current work](OPEN_WORK.md) for repeated seeds and optimization continuation.
