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
