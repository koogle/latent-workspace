import torch

from latent_workspace.corpus import DOMAINS, SPLITS, Record, build, record_windows
from latent_workspace.study import nested_indices


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())


def test_balanced_splits_quarantine_duplicates():
    records = [
        Record("code", split, "duplicate-" + split, "shared duplicate content")
        for split in ("train", "test")
    ]
    records += [
        Record(
            domain,
            split,
            domain + split,
            "unique " + domain + split,
            "source " + split if domain == "translation" else "",
        )
        for split in SPLITS
        for domain in DOMAINS
    ]
    data, meta = build(records, Tokenizer(), 8, 4, dict.fromkeys(SPLITS, 1))
    assert all(rows.shape == (3, 12) for rows in data.values())
    assert len(meta["quarantined_duplicate_hashes"]) == 1
    assert not set(meta["window_hashes"]["train"]) & set(meta["window_hashes"]["test"])


def test_translation_labels_are_target_only():
    record = Record("translation", "train", "pair", "German answer", "English source")
    windows = list(record_windows(record, Tokenizer(), 128, 4))
    target = list((" " + record.content).encode())
    assert windows[0][0] == -1
    assert windows[0][-4:] == target[:4]
    assert windows[1][-8:] == target[:8]


def test_learning_curve_subsets_are_balanced_and_nested():
    ranges = {domain: (i * 12, (i + 1) * 12) for i, domain in enumerate(DOMAINS)}
    subsets = nested_indices(ranges, (6, 18, 36), 42)
    assert set(subsets[6].tolist()) < set(subsets[18].tolist()) < set(subsets[36].tolist())
    for size, subset in subsets.items():
        for start, end in ranges.values():
            assert int(((subset >= start) & (subset < end)).sum()) == size // 3
    torch.testing.assert_close(subsets[18], nested_indices(ranges, (18,), 42)[18])


def test_study_runs_all_sizes_without_evaluating_test(tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    from test_backbone import tiny_backbone
    from test_core import tiny_config

    from latent_workspace import study

    cfg = replace(
        tiny_config(),
        corpus="mixed",
        train_examples=12,
        eval_examples=6,
        test_examples=6,
        study_sizes=(6, 12),
        study_checkpoints=(1, 2),
        variants=("diffusion:multi",),
        batch_size=2,
        max_seconds=60,
    )
    data = {
        "train": torch.randint(31, (12, 28)),
        "validation": torch.randint(31, (6, 28)),
        "test": torch.full((6, 28), 99),
    }
    ranges = {
        split: {d: [i * len(rows) // 3, (i + 1) * len(rows) // 3] for i, d in enumerate(DOMAINS)}
        for split, rows in data.items()
    }

    def factory(config, device):
        backbone = tiny_backbone()
        backbone.cfg = config
        backbone.revision = "test"
        backbone.tokenizer = SimpleNamespace(eos_token_id=30, decode=lambda ids: str(ids))
        return backbone

    monkeypatch.setattr(study, "FrozenBackbone", factory)
    monkeypatch.setattr(study, "prepare", lambda *a: (data, {"domain_ranges": ranges}))
    result = study.run_study(cfg, tmp_path)
    assert result["completed"]
    assert "sealed" in result["test_status"]
    for curve in result["curves"].values():
        assert len(curve["checkpoints"]) == 2
        assert curve["checkpoints"][-1]["masked_supervised_tokens_seen"] <= 16
        assert set(curve["validation_samples"]) == set(DOMAINS)
