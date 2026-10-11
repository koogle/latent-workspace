"""A checkpoint must reproduce the next update, not just load without errors."""

from types import SimpleNamespace

import torch
from test_core import ToyBackbone, tiny_config

from latent_workspace.models import BlockHead, loss_and_logits
from latent_workspace.train import checkpoint


def test_optimizer_and_rng_resume_reproduces_next_update(tmp_path):
    torch.manual_seed(12)
    cfg = tiny_config()
    backbone = ToyBackbone()
    head = BlockHead(12, cfg, "diffusion:multi")
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    sampler = torch.Generator().manual_seed(20)
    memory = torch.randn(5, 4, 3, 12)
    targets = torch.randint(31, (5, 4))

    def update(model, opt, rng):
        idx = torch.randint(5, (2,), generator=rng)
        opt.zero_grad()
        loss, _, _ = loss_and_logits(model, backbone, memory[idx], targets[idx])
        loss.backward()
        opt.step()
        return loss.detach()

    update(head, optimizer, sampler)
    path = tmp_path / "state.pt"
    checkpoint(
        path,
        head,
        optimizer,
        1,
        cfg,
        SimpleNamespace(revision="frozen-test", jacobians=None),
        sampler,
    )
    expected_loss = update(head, optimizer, sampler)
    restored = BlockHead(12, cfg, "diffusion:multi")
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=0.001)
    state = torch.load(path, weights_only=True)
    restored.load_state_dict(state["head"])
    restored_opt.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng"])
    sampler.set_state(state["sampler_rng"])
    actual_loss = update(restored, restored_opt, sampler)
    torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
    for expected, actual in zip(head.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
