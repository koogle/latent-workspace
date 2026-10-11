"""Balanced prose/code/translation corpus with frozen train/validation/test splits."""

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass

import torch

from .data import documents

DOMAINS = ("prose", "code", "translation")
SPLITS = ("train", "validation", "test")
SOURCES = {
    "prose": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
    "code": ("codeparrot/codeparrot-clean", None),
    "translation": ("wmt/wmt14", "de-en"),
}


def digest(text):
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def repository_split(repository):
    value = int(digest("latent-workspace-v1:" + repository), 16) % 100
    return "test" if value < 10 else "validation" if value < 20 else "train"


@dataclass
class Record:
    domain: str
    split: str
    group: str
    content: str
    source: str = ""

    @property
    def fingerprint(self):
        # Source-only deduplication prevents translation source leakage even if
        # alternative target translations differ across official partitions.
        return digest(self.source if self.domain == "translation" else self.content)


def record_windows(record, tokenizer, context, block):
    if record.domain != "translation":
        ids = tokenizer.encode(record.content, add_special_tokens=False)
        for start in range(0, len(ids) - context - block + 1, context + block):
            yield ids[start : start + context + block]
    else:
        prompt = "Translate English to German.\nEnglish: " + record.source + "\nGerman:"
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        target = tokenizer.encode(" " + record.content, add_special_tokens=False)
        # Only target-language tokens receive labels; never window source text
        # as a translation target. -1 marks left padding, masked in the backbone.
        for offset in range(0, len(target) - block + 1, block):
            context_ids = (prefix + target[:offset])[-context:]
            yield (
                [-1] * (context - len(context_ids)) + context_ids + target[offset : offset + block]
            )


def build(records, tokenizer, context, block, limits):
    """Collect disjoint groups first, then tokenize; exact duplicates are quarantined."""
    pools = defaultdict(list)
    counts = defaultdict(int)
    fingerprints = {}
    duplicate_hashes = set()
    for record in records:
        key = (record.split, record.domain)
        if counts[key] >= limits[record.split] * 3:
            continue
        fingerprint = record.fingerprint
        previous = fingerprints.get(fingerprint)
        if previous is not None:
            if previous != key:
                duplicate_hashes.add(fingerprint)
            continue
        fingerprints[fingerprint] = key
        from itertools import islice

        windows = list(islice(record_windows(record, tokenizer, context, block), 4))
        if windows:
            pools[key].append((record, windows))
            counts[key] += len(windows)
    tensors, ranges, groups, hashes = {}, {}, {}, {}
    for split in SPLITS:
        rows, ranges[split], groups[split], hashes[split] = [], {}, [], []
        for domain in DOMAINS:
            start = len(rows)
            for record, windows in pools[(split, domain)]:
                if record.fingerprint in duplicate_hashes:
                    continue
                remaining = limits[split] - (len(rows) - start)
                if remaining <= 0:
                    break
                chosen = windows[:remaining]
                rows.extend(chosen)
                groups[split].extend([f"{domain}:{record.group}"] * len(chosen))
                hashes[split].extend([record.fingerprint] * len(chosen))
            end = len(rows)
            if end - start < limits[split]:
                raise ValueError(
                    f"insufficient disjoint {split}/{domain} windows: {end - start}/{limits[split]}"
                )
            ranges[split][domain] = [start, end]
        tensors[split] = torch.tensor(rows, dtype=torch.long)
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(groups[a]) & set(groups[b]) or set(hashes[a]) & set(hashes[b]):
            raise ValueError(f"split leakage: {a}/{b}")
    metadata = {
        "domain_ranges": ranges,
        "window_groups": groups,
        "window_hashes": hashes,
        "split_policy": "mixed-v1",
        "quarantined_duplicate_hashes": sorted(duplicate_hashes),
    }
    return tensors, metadata


def prepare(cfg, tokenizer):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revisions = {domain: HfApi().dataset_info(repo).sha for domain, (repo, _) in SOURCES.items()}
    limits = {
        "train": cfg.train_examples // 3,
        "validation": cfg.eval_examples // 3,
        "test": cfg.test_examples // 3,
    }

    # Oversample before exact-duplicate quarantine; quotas are checked afterwards.
    def records():
        for domain in DOMAINS:
            repo, subset = SOURCES[domain]
            splits = ("train",) if domain == "code" else ("test", "validation", "train")
            for split in splits:
                rows = load_dataset(
                    repo, subset, split=split, revision=revisions[domain], streaming=True
                )
                if domain == "prose":
                    rows = ({"text": text} for text in documents(rows))
                counts = defaultdict(int)
                for index, row in enumerate(rows):
                    if domain == "code":
                        group = row["repo_name"]
                        partition = repository_split(group)
                        record = Record(domain, partition, group, row["content"])
                    elif domain == "translation":
                        pair = row["translation"]
                        record = Record(domain, split, digest(pair["en"]), pair["de"], pair["en"])
                    else:
                        record = Record(domain, split, digest(row["text"]), row["text"])
                    key = record.split
                    quota = limits[key] * 3
                    if counts[key] < quota:
                        # Enough records, not just enough windows; caps prevent
                        # a single huge code repository dominating a split.
                        yield record
                        counts[key] += 1
                    required = SPLITS if domain == "code" else (split,)
                    if all(counts[x] >= limits[x] * 3 for x in required):
                        break
                    if index >= 100000:
                        break

    tensors, metadata = build(records(), tokenizer, cfg.context_length, cfg.block_size, limits)
    metadata["sources"] = {
        domain: {
            "repository": SOURCES[domain][0],
            "config": SOURCES[domain][1],
            "revision": revision,
        }
        for domain, revision in revisions.items()
    }
    metadata["dataset_revision"] = digest(json.dumps(metadata["sources"], sort_keys=True))
    return tensors, metadata
