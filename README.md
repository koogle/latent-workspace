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

## Completed studies

The initial WikiText pilot did not produce useful generation. The subsequent
[mixed-domain learning-curve study](docs/data-budget.md) separated corpus size
from optimization exposure. Its original pool had an exact-window overlap;
clean follow-up runs removed it and preserved held-out data identities.

Additional reference: Chen et al., [DFlash: Block Diffusion for Flash Speculative
Decoding](https://arxiv.org/abs/2602.06036), [official code](https://github.com/z-lab/dflash).
This prototype does not reproduce its verifier or lossless speedup claims.

The [first measured data-budget curves](results/data-study-01.md) are available:
more data improved validation loss, but sample generations remain degenerate.
No sufficient training budget or useful drafter quality is established yet.


The [data-size curves](results/data-scaling.md) established a 196,608-window
mixed-domain diagnostic pool. The [trained conditioning study](results/conditioning-findings.md)
is now complete: 96 layer probes, nine Qwen2.5-0.5B decoder conditions, two
Qwen3-0.6B conditions, and a frozen-checkpoint test evaluation of all 11 heads.
All-layer residual conditioning helps, but compressed J-lens conditioning and
denoising do not improve on their respective controls here. Generation remains
poor; there is no demonstrated quality parity or speedup.

![Conditioning learning curves](results/conditioning-comparison.png)

The [test report](results/conditioning-test.md) includes all domain measurements,
paired uncertainty and actual continuations. Test was withheld during training
and selection, then evaluated once; it is no longer an untouched holdout for
future adaptive work.

## Trained checkpoint

The all-layer Qwen2.5 diffusion head is saved on Modal and downloaded locally at
`artifacts/conditioning-all-best.pt`. Its checksum matches the frozen test head.
It is a research checkpoint with poor generation quality.

```bash
uv run modal volume get latent-workspace-qwen-runs \
  conditioning-all-01/size-196608-best.pt artifacts/conditioning-all-best.pt
uv run latent-workspace generate --checkpoint artifacts/conditioning-all-best.pt \
  --prompt 'Once upon a time' --tokens 24
```

Checkpoint files reference pinned pretrained Qwen weights; they do not bundle
those weights. See [current limitations and remaining work](OPEN_WORK.md).

## Reading the training curves

One **window** is a 128-token prefix and the following four target tokens. The
four-token generation block stays fixed. Curve labels such as 500/1000/2000/4000
count optimizer updates, not block length or learning-rate changes. At batch 128,
4000 updates sample 512000 windows with replacement—about 2.6 sampled passes over
196608 distinct windows. This is a fixed screening budget, not proof of convergence.

The [small-data follow-up](docs/small-data-controls.md) measures the entire training
subset alongside validation, with denser checkpoints and 96/768/3072-window sets.
It tests all-layer conditioning, whole-layer dropout, zero-content attention, and
both together. Learning rate remains 0.0003. Existing validation is reused for
exploration; the earlier test has already been evaluated and is not reused here.
