"""The encoder only sees committed prefix tokens, never prediction targets."""

import torch
from torch import nn


class FrozenBackbone(nn.Module):
    def __init__(self, cfg, device="cuda"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.model_revision)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                cfg.model,
                revision=cfg.model_revision,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.hidden_size = self.model.config.hidden_size
        if max(cfg.layers) != self.model.config.num_hidden_layers:
            raise ValueError("last selected layer must be the model's final hidden state")
        self.jacobians = None
        self.revision = self.model.config._commit_hash

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def encode(self, prefix, *, past_key_values=None):
        residuals, attentions, hooks = {}, {}, []

        def capture(destination, index):
            def hook(module, args, output):
                value = output[0] if isinstance(output, tuple) else output
                destination[index] = value[:, -self.cfg.memory_tokens :].detach()

            return hook

        try:
            for index in self.cfg.layers:
                layer = self.model.model.layers[index - 1]
                hooks.append(layer.register_forward_hook(capture(residuals, index)))
                hooks.append(layer.self_attn.register_forward_hook(capture(attentions, index)))
            out = self.model.model(
                input_ids=prefix, use_cache=True, past_key_values=past_key_values
            )
        finally:
            for hook in hooks:
                hook.remove()
        residual = torch.stack([residuals[i] for i in self.cfg.layers], dim=1)
        attention = torch.stack([attentions[i] for i in self.cfg.layers], dim=1)
        parts = [residual, attention]
        if self.jacobians is not None and getattr(self, "conditioning", None) in (None, "jspace"):
            from .jspace import readout

            parts.append(readout(self, residual))
        return torch.cat(parts, dim=1), out.past_key_values

    @torch.no_grad()
    def embed(self, ids):
        return self.model.get_input_embeddings()(ids)

    def project(self, features):
        # Frozen weights, but preserve gradients to the trainable head.
        weight = self.model.get_output_embeddings().weight
        return torch.nn.functional.linear(features.to(weight.dtype), weight)

    @torch.no_grad()
    def teacher_logits(self, prefix, target):
        joined = torch.cat([prefix, target[:, :-1]], dim=1)
        return self.model(joined, use_cache=False).logits[:, prefix.shape[1] - 1 :]
