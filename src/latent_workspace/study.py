"""Data-size/optimization learning curves; test metrics stay sealed."""

import hashlib
import json
import time
from pathlib import Path

import torch

from .backbone import FrozenBackbone
from .corpus import DOMAINS, extend_frozen_corpus, prepare
from .evaluate import evaluate, generate, repeated_next_token_baseline, teacher_baseline
from .models import BlockHead, loss_and_logits
from .scan import cached_metrics
from .train import atomic_save, checkpoint, seed_all, write_json


def nested_indices(ranges, sizes, seed, priority=None):
    generator = torch.Generator().manual_seed(seed)
    orders = {
        domain: torch.randperm(end - start, generator=generator) + start
        for domain, (start, end) in ranges.items()
    }
    if priority:
        for domain, first in priority.items():
            first = torch.tensor(first, dtype=torch.long)
            orders[domain] = torch.cat([first, orders[domain][~torch.isin(orders[domain], first)]])
    return {
        size: torch.cat([orders[domain][: size // len(DOMAINS)] for domain in DOMAINS])
        for size in sizes
    }


def run_study(cfg, output, on_checkpoint=None):
    cfg.validate()
    if cfg.corpus != "mixed" or len(cfg.variants) != 1 or cfg.evaluate_test:
        raise ValueError(
            "data study requires mixed training, one fixed variant and a sealed test set"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "study.json").exists():
        raise ValueError("study already exists; use a new run ID")
    start = time.monotonic()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(cfg.seed)
    parent = None
    if cfg.study_parent:
        parent_path = Path(cfg.study_parent)
        parent = json.loads((parent_path / "study.json").read_text())
        for key in ("model", "context_length", "block_size"):
            if cfg.to_dict()[key] != parent["config"][key]:
                raise ValueError(f"parent corpus incompatible: {key}")
        cfg.model_revision = parent["config"]["model_revision"]
    backbone = FrozenBackbone(cfg, device)
    cfg.model_revision = backbone.revision
    priority = None
    if parent:
        base = torch.load(parent_path / "corpus.pt", weights_only=True)
        if cfg.train_examples == len(base["tensors"]["train"]):
            data, metadata = base["tensors"], base["metadata"]
            ranges = metadata["domain_ranges"]["train"]
        else:
            candidate = prepare(cfg, backbone.tokenizer, base["metadata"])
            data, metadata, ranges = extend_frozen_corpus(base, candidate, cfg.train_examples)
        original_order = parent["curves"][str(len(base["tensors"]["train"]))]["train_indices"]
        priority = {}
        for domain, (new_start, _) in ranges.items():
            a, b = base["metadata"]["domain_ranges"]["train"][domain]
            priority[domain] = [new_start + i - a for i in original_order if a <= i < b]
        for split in ("validation", "test"):
            if len(data[split]) != getattr(
                cfg, "eval_examples" if split == "validation" else "test_examples"
            ):
                raise ValueError("parent held-out sizes must be preserved")
    else:
        data, metadata = prepare(cfg, backbone.tokenizer)
    atomic_save(output / "corpus.pt", {"tensors": data, "metadata": metadata})
    source_hash = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(source.name.encode() + source.read_bytes())
    result = {
        "config": cfg.to_dict(),
        "data": metadata,
        "source_sha256": source_hash.hexdigest(),
        "test_status": "sealed; no test losses or accuracy computed",
        "curves": {},
        "readout_norm_weight_rms": float(
            backbone.model.model.norm.weight.float().square().mean().sqrt()
        ),
    }

    result["teacher_validation_by_domain"] = {
        domain: teacher_baseline(backbone, data["validation"][a:b], cfg, device)
        for domain, (a, b) in metadata["domain_ranges"]["validation"].items()
    }

    result["repeated_next_token_baseline"] = {
        domain: repeated_next_token_baseline(backbone, data["validation"][a:b], cfg, device)
        for domain, (a, b) in metadata["domain_ranges"]["validation"].items()
    }

    def persist():
        write_json(output / "study.json", result)
        if on_checkpoint:
            on_checkpoint()

    persist()
    memories = {}
    for split in ("train", "validation"):
        chunks = []
        for rows in data[split].split(cfg.batch_size):
            chunks.append(backbone.encode(rows[:, : cfg.context_length].to(device))[0].cpu())
            if time.monotonic() - start > cfg.max_seconds:
                raise TimeoutError("data study budget exhausted extracting features")
        memories[split] = torch.cat(chunks)
    subsets = nested_indices(
        metadata["domain_ranges"]["train"], cfg.study_sizes, cfg.seed, priority
    )
    variant = cfg.variants[0]
    for size, subset in subsets.items():
        seed_all(cfg.seed)
        head = BlockHead(backbone.hidden_size, cfg, variant).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
        sampler = torch.Generator().manual_seed(cfg.seed)
        selected = data["train"][subset]
        memory = memories["train"][subset]
        per_domain = size // len(DOMAINS)
        diagnostic_indices = torch.cat(
            [
                torch.arange(i * per_domain, i * per_domain + min(32, per_domain))
                for i in range(len(DOMAINS))
            ]
        )
        masked_exposures = 0
        curve = {
            "unique_blocks": size,
            "unique_target_token_slots": size * cfg.block_size,
            "train_indices": subset.tolist(),
            "checkpoints": [],
        }
        result["curves"][str(size)] = curve
        for step in range(1, max(cfg.study_checkpoints) + 1):
            if time.monotonic() - start > cfg.max_seconds:
                result["stopped"] = "time budget; incomplete curves"
                persist()
                return result
            head.train()
            index = torch.randint(size, (cfg.batch_size,), generator=sampler)
            optimizer.zero_grad(set_to_none=True)
            loss, _, mask = loss_and_logits(
                head,
                backbone,
                memory[index].to(device),
                selected[index, cfg.context_length :].to(device),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite study loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            masked_exposures += int(mask.sum())
            if step in cfg.study_checkpoints:
                train_metric = cached_metrics(
                    head,
                    backbone,
                    memory[diagnostic_indices],
                    selected[diagnostic_indices, cfg.context_length :],
                    cfg,
                    device,
                )
                by_domain = {}
                for domain, (a, b) in metadata["domain_ranges"]["validation"].items():
                    by_domain[domain] = evaluate(
                        head, backbone, data["validation"][a:b], cfg, device
                    )
                row = {
                    "step": step,
                    "target_token_slots_seen": step * cfg.batch_size * cfg.block_size,
                    "masked_supervised_tokens_seen": masked_exposures,
                    "training_sample_nll_by_position": train_metric["nll_by_position"],
                    "training_sample_accuracy_by_position": train_metric["accuracy_by_position"],
                    "training_position_collapse_fraction": train_metric[
                        "same_prediction_at_all_positions"
                    ],
                    "validation_by_domain": by_domain,
                    "macro_validation_nll": sum(v["nll"] for v in by_domain.values())
                    / len(DOMAINS),
                }
                curve["checkpoints"].append(row)
                checkpoint(
                    output / f"size-{size}.pt", head, optimizer, step, cfg, backbone, sampler
                )
                result["elapsed_seconds"] = time.monotonic() - start
                persist()
                print(
                    f"Data study size={size} step={step} validation_nll={row['macro_validation_nll']:.4f} supervised_tokens={masked_exposures}",
                    flush=True,
                )
        curve["validation_samples"] = {}
        for domain, (a, _) in metadata["domain_ranges"]["validation"].items():
            prefix = data["validation"][a : a + 1, : cfg.context_length].to(device)
            tokens = generate(head, backbone, prefix, max_new_tokens=24)
            curve["validation_samples"][domain] = {
                "prefix": backbone.tokenizer.decode(prefix[0][prefix[0] >= 0].tolist()),
                "generated": backbone.tokenizer.decode(tokens),
            }
        persist()
        del head, optimizer, memory
    result["completed"] = True
    result["elapsed_seconds"] = time.monotonic() - start
    persist()
    return result
