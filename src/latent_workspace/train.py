import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .backbone import FrozenBackbone
from .data import load_data
from .evaluate import benchmark, evaluate, generate, sync, teacher_baseline
from .models import BlockHead, loss_and_logits


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def checkpoint(path, head, optimizer, step, cfg, backbone, generator):
    payload = {
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": cfg.to_dict(),
        "variant": head.variant,
        "model_revision": backbone.revision,
        "jacobians": backbone.jacobians,
        "rng": torch.get_rng_state(),
        "sampler_rng": generator.get_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def run(cfg, output, *, resume=False, on_checkpoint=None):
    cfg.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() and not resume:
        raise ValueError("run exists; use a new run ID or explicit resume")
    started = time.monotonic()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(cfg.seed)
    source_hash = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(source.name.encode() + source.read_bytes())
    source_digest = source_hash.hexdigest()
    backbone = FrozenBackbone(cfg, device)
    # Persist actual immutable revisions before training.
    cfg.model_revision = backbone.revision
    if resume:
        old = json.loads((output / "manifest.json").read_text())
        cfg.dataset_revision = old["data"]["dataset_revision"]
        if old.get("source_sha256") != source_digest:
            raise ValueError("source changed since checkpoint; resume requires identical code")
        comparable = json.loads(json.dumps(cfg.to_dict()))
        if comparable != old["config"]:
            raise ValueError("resume config differs from saved config")
    train, valid, data_info = load_data(cfg, backbone.tokenizer)
    cfg.dataset_revision = data_info["dataset_revision"]
    calibration_indices = torch.randperm(
        len(train), generator=torch.Generator().manual_seed(cfg.seed + 1)
    )[: cfg.jspace_prompts]
    manifest = {
        "config": cfg.to_dict(),
        "source_sha256": source_digest,
        "calibration_indices": calibration_indices.tolist(),
        "model_revision": backbone.revision,
        "data": data_info,
        "device": device,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
    }
    write_json(output / "manifest.json", manifest)
    if on_checkpoint:
        on_checkpoint()
    if any(v.endswith(":jspace") for v in cfg.variants):
        from .jspace import fit_jacobians

        lens_path = output / "jacobians.pt"
        if resume and lens_path.exists():
            backbone.jacobians = torch.load(lens_path, weights_only=True)
        else:
            backbone.jacobians = fit_jacobians(
                backbone,
                train[calibration_indices, : min(128, cfg.context_length)],
                deadline=started + cfg.max_seconds,
                checkpoint_path=output / "jacobian-calibration.pt",
                on_checkpoint=on_checkpoint,
            )
            torch.save(backbone.jacobians, lens_path)
        if on_checkpoint:
            on_checkpoint()
    report = {
        "manifest": manifest,
        "variants": {},
        "teacher": teacher_baseline(backbone, valid, cfg, device),
    }
    cache_path = output / "features.pt"
    if resume and cache_path.exists():
        cache = torch.load(cache_path, weights_only=True)
    else:
        chunks = []
        for batch in train.split(cfg.batch_size):
            memory, _ = backbone.encode(batch[:, : cfg.context_length].to(device))
            chunks.append(memory.cpu())
            if time.monotonic() - started > cfg.max_seconds:
                raise TimeoutError("time budget exhausted during extraction")
        cache = torch.cat(chunks)
        torch.save(cache, cache_path)
        del chunks
    report["feature_cache_bytes"] = cache.numel() * cache.element_size()
    for variant in cfg.variants:
        if time.monotonic() - started > cfg.max_seconds:
            report["stopped"] = "time budget"
            break
        seed_all(cfg.seed)
        head = BlockHead(backbone.hidden_size, cfg, variant).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
        sampler = torch.Generator().manual_seed(cfg.seed)
        path = output / (variant.replace(":", "-") + ".pt")
        step = 0
        if resume and path.exists():
            state = torch.load(path, map_location=device, weights_only=True)
            head.load_state_dict(state["head"])
            optimizer.load_state_dict(state["optimizer"])
            step = state["step"]
            torch.set_rng_state(state["rng"].cpu())
            sampler.set_state(state["sampler_rng"].cpu())
            if state["cuda_rng"]:
                torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        initial_step = step
        initial = evaluate(head, backbone, valid, cfg, device)
        sync(device)
        train_start = time.perf_counter()
        losses = []
        while step < cfg.steps and time.monotonic() - started < cfg.max_seconds:
            head.train()
            indices = torch.randint(len(train), (cfg.batch_size,), generator=sampler)
            memory = cache[indices].to(device)
            target = train[indices, cfg.context_length :].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = loss_and_logits(head, backbone, memory, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss: {variant}, {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            step += 1
            losses.append(float(loss.detach()))
            if step % cfg.eval_every == 0:
                checkpoint(path, head, optimizer, step, cfg, backbone, sampler)
                if on_checkpoint:
                    on_checkpoint()
                print(
                    json.dumps(
                        {
                            "variant": variant,
                            "step": step,
                            "loss": sum(losses[-cfg.eval_every :])
                            / min(len(losses), cfg.eval_every),
                        }
                    ),
                    flush=True,
                )
        sync(device)
        training_seconds = time.perf_counter() - train_start
        checkpoint(path, head, optimizer, step, cfg, backbone, sampler)
        result = {
            "parameters": sum(p.numel() for p in head.parameters() if p.requires_grad),
            "steps": step,
            "completed": step == cfg.steps,
            "context_token_exposures": step * cfg.batch_size * cfg.context_length,
            "target_token_exposures": step * cfg.batch_size * cfg.block_size,
            "training_seconds_this_session": training_seconds,
            "steps_per_second_this_session": (step - initial_step) / max(training_seconds, 1e-9),
            "initial": initial,
            "held_out": evaluate(head, backbone, valid, cfg, device),
        }
        # Only one head at a time, no concurrent timing or models in memory.
        result["latency"] = benchmark(
            head, backbone, valid[:1, : cfg.context_length].to(device), tokens=16
        )
        result["samples"] = []
        for prompt in ("The capital of France is", "Once upon a time"):
            prefix = backbone.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            tokens = generate(head, backbone, prefix, max_new_tokens=16)
            result["samples"].append(
                {"prompt": prompt, "completion": backbone.tokenizer.decode(tokens)}
            )
        report["variants"][variant] = result
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "results.json", report)
        if on_checkpoint:
            on_checkpoint()
        print(
            json.dumps(
                {
                    "variant": variant,
                    "nll": result["held_out"]["nll"],
                    "parameters": result["parameters"],
                    "steps": step,
                }
            ),
            flush=True,
        )
        del optimizer, head
        if device == "cuda":
            torch.cuda.empty_cache()
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_gpu_bytes"] = torch.cuda.max_memory_allocated() if device == "cuda" else None
    write_json(output / "results.json", report)
    if on_checkpoint:
        on_checkpoint()
    return report
