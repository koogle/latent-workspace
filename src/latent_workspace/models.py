"""Trainable heads; frozen Qwen embedding and output weights stay external."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class Workspace(nn.Module):
    def __init__(self, hidden, cfg):
        super().__init__()
        self.project = nn.Linear(hidden, cfg.width)
        self.depth = nn.Parameter(torch.randn(len(cfg.layers), cfg.width) * 0.02)
        self.position = nn.Parameter(torch.randn(cfg.memory_tokens, cfg.width) * 0.02)
        self.slots = nn.Parameter(torch.randn(cfg.slots, cfg.width) * 0.02)
        self.attend = nn.MultiheadAttention(cfg.width, cfg.heads, batch_first=True)
        self.norm = nn.LayerNorm(cfg.width)
        self.ff = nn.Sequential(nn.Linear(cfg.width, cfg.width * 4), nn.GELU(),
                                nn.Linear(cfg.width * 4, cfg.width))

    def forward(self, memory):
        # [batch, depth, context tail, hidden], all strictly before the target.
        b, depths, positions, _ = memory.shape
        x = self.project(memory.float())
        x = x + self.depth[:depths][None, :, None] + self.position[:positions][None, None]
        x = x.reshape(b, depths * positions, -1)
        slots = self.slots.unsqueeze(0).expand(b, -1, -1)
        slots = self.norm(slots + self.attend(slots, x, x, need_weights=False)[0])
        return slots + self.ff(slots)


class BlockHead(nn.Module):
    def __init__(self, hidden, cfg, variant):
        super().__init__()
        self.cfg, self.variant = cfg, variant
        self.kind, self.condition = variant.split(":")
        self.position = nn.Parameter(torch.randn(cfg.block_size, cfg.width) * 0.02)
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(cfg.width, hidden)
        if self.kind == "probe":
            self.probe = nn.Sequential(nn.Linear(hidden, cfg.width), nn.GELU(),
                                       nn.Linear(cfg.width, cfg.width), nn.GELU())
            return
        self.workspace = Workspace(hidden, cfg)
        self.token_project = nn.Linear(hidden, cfg.width)
        self.mask = nn.Parameter(torch.randn(cfg.width) * 0.02)
        self.time = nn.Sequential(nn.Linear(1, cfg.width), nn.SiLU(),
                                  nn.Linear(cfg.width, cfg.width))
        layer = nn.TransformerDecoderLayer(cfg.width, cfg.heads, cfg.width * 4,
                                           dropout=0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, cfg.decoder_layers)
        # Identical capacity across conditioning ablations.


    def forward(self, memory, token_embeddings=None, masked=None, noise=None):
        b = memory.shape[0]
        if self.kind == "probe":
            index = self.cfg.layers.index(int(self.condition))
            x = self.probe(memory[:, index, -1].float())[:, None] + self.position
        else:
            depth = len(self.cfg.layers)
            if self.condition == "final":
                memory = memory[:, depth - 1:depth]
            elif self.condition == "multi":
                memory = memory[:, :depth]
            elif self.condition == "attention_multi":
                memory = memory[:, depth:2 * depth]
            elif self.condition == "jspace":
                memory = memory[:, 2 * depth:]
            elif self.condition.startswith("layer"):
                index = self.cfg.layers.index(int(self.condition[5:]))
                memory = memory[:, index:index + 1]
            elif self.condition.startswith("attention"):
                index = depth + self.cfg.layers.index(int(self.condition[9:]))
                memory = memory[:, index:index + 1]
            slots = self.workspace(memory)
            x = self.position.unsqueeze(0).expand(b, -1, -1)
            if self.kind != "parallel":
                if token_embeddings is None or masked is None or noise is None:
                    raise ValueError("diffusion needs corrupted tokens, mask, and noise level")
                visible = self.token_project(token_embeddings.float())
                x = x + torch.where(masked[..., None], self.mask, visible)
                x = x + self.time(noise[:, None].float())[:, None]
            for _ in range(self.cfg.refinement_rounds):
                x = self.decoder(x, slots)
        return self.out_norm(self.out(x))


def corrupt(target, generator=None):
    """Uniform nonempty mask count; loss averages masked positions per example.

    Deliberately a masked-denoising objective, not a BD3-LM ELBO reproduction.
    """
    b, length = target.shape
    count = torch.randint(1, length + 1, (b,), device=target.device, generator=generator)
    order = torch.rand(target.shape, device=target.device, generator=generator).argsort(-1)
    mask = order.argsort(-1) < count[:, None]
    return mask, count.float() / length


def loss_and_logits(head, backbone, memory, target, *, all_masked=False):
    if head.kind in ("parallel", "probe"):
        mask = torch.ones_like(target, dtype=torch.bool)
        features = head(memory)
    else:
        mask, noise = corrupt(target)
        if all_masked:
            mask = torch.ones_like(mask)
            noise = torch.ones_like(noise)
        # Masked target IDs cannot leak through embeddings, even accidentally.
        ids = target.masked_fill(mask, 0)
        features = head(memory, backbone.embed(ids), mask, noise)
    logits = backbone.project(features)
    ce = F.cross_entropy(logits.float().transpose(1, 2), target, reduction="none")
    loss = ((ce * mask).sum(-1) / mask.sum(-1)).mean()
    return loss, logits, mask


@torch.no_grad()
def sample_block(head, backbone, memory, temperature=0.0, steps=None):
    b = memory.shape[0]
    length = head.cfg.block_size
    ids = torch.zeros((b, length), dtype=torch.long, device=memory.device)
    if head.kind in ("parallel", "probe"):
        logits = backbone.project(head(memory)).float()
        return draw(logits, temperature)
    mask = torch.ones_like(ids, dtype=torch.bool)
    steps = steps or head.cfg.diffusion_steps
    if steps <= 0:
        raise ValueError("steps must be positive")
    for step in range(steps):
        noise = mask.float().mean(-1)
        logits = backbone.project(head(memory, backbone.embed(ids), mask, noise)).float()
        proposals = draw(logits, temperature)
        confidence = logits.softmax(-1).gather(-1, proposals[..., None]).squeeze(-1)
        confidence = confidence.masked_fill(~mask, -float("inf"))
        remaining = math.ceil(length * (1 - (step + 1) / steps))
        to_commit = (mask.sum(-1) - remaining).clamp(min=0)
        rank = confidence.argsort(-1, descending=True).argsort(-1)
        commit = mask & (rank < to_commit[:, None])
        ids = torch.where(commit, proposals, ids)
        mask &= ~commit
    return ids


def draw(logits, temperature):
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if temperature == 0:
        return logits.argmax(-1)
    shape = logits.shape[:-1]
    probs = (logits / temperature).softmax(-1).reshape(-1, logits.shape[-1])
    return torch.multinomial(probs, 1).reshape(shape)
