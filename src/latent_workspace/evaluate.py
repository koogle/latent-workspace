import time

import torch
from torch.nn import functional as F

from .models import loss_and_logits, sample_block


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(head, backbone, data, cfg, device):
    head.eval()
    backbone.conditioning = head.condition
    nll = torch.zeros(cfg.block_size, device=device)
    accuracy = torch.zeros_like(nll)
    exact, count, generated_exact = 0, 0, 0
    generated_accuracy = torch.zeros_like(nll)
    per_example = []
    for rows in data.split(cfg.batch_size):
        rows = rows.to(device)
        prefix, target = rows[:, : cfg.context_length], rows[:, cfg.context_length :]
        memory, _ = backbone.encode(prefix)
        # Identical all-masked context-only metric for EVERY architecture.
        _, logits, _ = loss_and_logits(head, backbone, memory, target, all_masked=True)
        ce = F.cross_entropy(logits.float().transpose(1, 2), target, reduction="none")
        hits = logits.argmax(-1) == target
        generated_hits = sample_block(head, backbone, memory) == target
        generated_accuracy += generated_hits.sum(0)
        generated_exact += generated_hits.all(-1).sum().item()
        nll += ce.sum(0)
        accuracy += hits.sum(0)
        exact += hits.all(-1).sum().item()
        count += len(rows)
        per_example.extend(ce.mean(-1).cpu().tolist())
    values = torch.tensor(per_example)
    return {
        "nll": float(nll.sum() / (count * cfg.block_size)),
        "nll_by_position": (nll / count).tolist(),
        "accuracy_by_position": (accuracy / count).tolist(),
        "exact_block_accuracy": exact / count,
        "generated_accuracy_by_position": (generated_accuracy / count).tolist(),
        "generated_exact_block_accuracy": generated_exact / count,
        "examples": count,
        "example_mean_nll": per_example,
        "nll_standard_error": float(values.std(unbiased=True) / count**0.5) if count > 1 else None,
    }


@torch.no_grad()
def teacher_baseline(backbone, data, cfg, device):
    nll = torch.zeros(cfg.block_size, device=device)
    hits = torch.zeros_like(nll)
    for rows in data.split(cfg.batch_size):
        rows = rows.to(device)
        prefix, target = rows[:, : cfg.context_length], rows[:, cfg.context_length :]
        logits = backbone.teacher_logits(prefix, target).float()
        nll += F.cross_entropy(logits.transpose(1, 2), target, reduction="none").sum(0)
        hits += (logits.argmax(-1) == target).sum(0)
    return {
        "nll_by_position": (nll / len(data)).tolist(),
        "accuracy_by_position": (hits / len(data)).tolist(),
        "note": "Teacher-forced AR reference sees true earlier target tokens; not a context-only comparator.",
    }


@torch.no_grad()
def generate(head, backbone, prefix, max_new_tokens=32, temperature=0.0):
    head.eval()
    backbone.conditioning = head.condition
    memory, cache = backbone.encode(prefix)
    generated = []
    eos = backbone.tokenizer.eos_token_id
    while len(generated) < max_new_tokens:
        block = sample_block(head, backbone, memory, temperature)[0]
        tokens = block.tolist()[: max_new_tokens - len(generated)]
        if eos in tokens:
            generated.extend(tokens[: tokens.index(eos) + 1])
            break
        generated.extend(tokens)
        if len(generated) < max_new_tokens:
            fresh, cache = backbone.encode(block[None], past_key_values=cache)
            memory = torch.cat([memory, fresh], dim=2)[:, :, -head.cfg.memory_tokens :]
    return generated


@torch.no_grad()
def benchmark(head, backbone, prefix, tokens=32, repeats=3):
    """Warm, end-to-end latency includes Qwen prefix and committed-block updates."""
    device = prefix.device

    def ar():
        return backbone.model.generate(
            prefix,
            max_new_tokens=tokens,
            min_new_tokens=tokens,
            do_sample=False,
            pad_token_id=backbone.tokenizer.eos_token_id,
        )

    def block():
        return generate(head, backbone, prefix, tokens)

    report = {}
    for name, fn in (("autoregressive", ar), ("block", block)):
        fn()
        times, lengths = [], []
        for _ in range(repeats):
            sync(device)
            start = time.perf_counter()
            output = fn()
            sync(device)
            times.append(time.perf_counter() - start)
            lengths.append(
                output.shape[-1] - prefix.shape[-1] if name == "autoregressive" else len(output)
            )
        report[name] = {
            "seconds": times,
            "output_tokens": lengths,
            "seconds_per_token": sum(times) / sum(lengths),
        }
    report["note"] = (
        "Unverified block quality; timing alone does not establish efficiency at matched quality."
    )
    return report
