from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-0.5B"
    model_revision: str = "main"
    dataset: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    dataset_revision: str = "main"
    layers: tuple[int, ...] = (6, 12, 18, 24)
    jspace_prompts: int = 8
    jspace_top_k: int = 25
    jspace_dim_batch: int = 16
    block_size: int = 4
    context_length: int = 128
    memory_tokens: int = 16
    slots: int = 8
    width: int = 512
    heads: int = 8
    decoder_layers: int = 4
    refinement_rounds: int = 2
    diffusion_steps: int = 4
    batch_size: int = 8
    train_examples: int = 8192
    eval_examples: int = 256
    steps: int = 1000
    eval_every: int = 250
    learning_rate: float = 0.0003
    seed: int = 42
    max_seconds: int = 1800
    variants: tuple[str, ...] = (
        "parallel:final", "diffusion:final", "diffusion:layer12",
        "diffusion:multi", "diffusion:attention12", "diffusion:attention_multi",
        "diffusion:jspace",
    )


    def validate(self):
        for key in ("jspace_prompts", "jspace_top_k", "jspace_dim_batch", "block_size", "context_length", "memory_tokens", "slots", "width", "heads",
                    "decoder_layers", "refinement_rounds", "diffusion_steps", "batch_size",
                    "train_examples", "eval_examples", "steps", "eval_every", "max_seconds"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.width % self.heads or self.memory_tokens > self.context_length:
            raise ValueError("width must divide heads; memory must fit context")
        if not self.layers or len(set(self.layers)) != len(self.layers) or min(self.layers) < 1:
            raise ValueError("layers must be unique positive hidden-state indices")
        if not self.variants or len(set(self.variants)) != len(self.variants):
            raise ValueError("variants must be nonempty and unique")
        if not 0 < self.learning_rate < 1:
            raise ValueError("invalid learning rate")
        for variant in self.variants:
            kind, condition = variant.split(":")
            if kind == "probe":
                if int(condition) not in self.layers:
                    raise ValueError("probe layer absent from extraction layers")
            elif kind not in ("parallel", "diffusion") or condition not in ("final", "multi", "attention_multi", "jspace", *[f"layer{i}" for i in self.layers], *[f"attention{i}" for i in self.layers]):
                raise ValueError(f"unknown variant: {variant}")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text())
        for key in ("layers", "variants"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values).validate()
