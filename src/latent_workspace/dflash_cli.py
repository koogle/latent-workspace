"""Load a trained draft checkpoint and generate through the frozen Qwen verifier."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dflash import DFlashDrafter, DraftConfig, verified_generate
from .dflash_train import precision


def generate_checkpoint(checkpoint, prompt, tokens):
    if not prompt.strip() or tokens <= 0:
        raise ValueError("prompt must be nonempty and tokens positive")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = DraftConfig(**state["config"]).validate()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = (
        AutoModelForCausalLM.from_pretrained(
            cfg.model,
            revision=cfg.model_revision,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision)
    drafter = DFlashDrafter(target.config, cfg).to(device).eval()
    drafter.load_state_dict(state["drafter"])
    prefix = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with precision(device):
        result = verified_generate(
            drafter, target, prefix, tokens, state["mask_token_id"], tokenizer.eos_token_id
        )
    result["text"] = tokenizer.decode(result["tokens"], skip_special_tokens=True)
    result["checkpoint_step"] = state["step"]
    result["note"] = (
        "Greedy target verification; anchor excluded from acceptance. One untuned call is not a speed benchmark."
    )
    return result
