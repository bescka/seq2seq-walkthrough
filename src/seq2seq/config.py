"""Paper vs mid vs toy hyperparameter presets.

`paper` documents the original experiment; `mid` is a real-WMT GPU run;
`toy` is what we run locally. Same code path — only these knobs change.
Comparison table: docs/configs.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


ConfigName = Literal["paper", "mid", "toy"]


@dataclass
class Seq2SeqConfig:
    name: str = "toy"

    # Model (paper: 4 layers × 1000 cells, 1000-d embeddings)
    num_layers: int = 4
    hidden_size: int = 1000
    embed_size: int = 1000

    # Vocab (paper: 160k source / 80k target)
    src_vocab_size: int = 160_000
    tgt_vocab_size: int = 80_000

    # Data
    dataset: str = "wmt/wmt14"
    dataset_config: str = "fr-en"
    src_lang: str = "en"
    tgt_lang: str = "fr"
    reverse_source: bool = True
    max_src_len: int = 50
    max_tgt_len: int = 50
    max_train_examples: int | None = None
    seed: int = 42

    # Training (paper §3.4)
    batch_size: int = 128
    learning_rate: float = 0.7
    init_range: float = 0.08
    grad_clip: float = 5.0
    epochs: float = 7.5
    lr_hold_epochs: float = 5.0
    lr_halve_every_epochs: float = 0.5
    bucket_width: int = 5

    # Runtime
    device: str = "cpu"
    num_workers: int = 0
    log_every: int = 10
    sample_every: int = 50
    checkpoint_dir: str = "runs/toy"
    max_steps: int | None = None

    # Decode
    beam_size: int = 2
    max_decode_len: int = 50

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def paper_config() -> Seq2SeqConfig:
    """Full paper hyperparams (reference; not for laptop training)."""
    return Seq2SeqConfig(
        name="paper",
        num_layers=4,
        hidden_size=1000,
        embed_size=1000,
        src_vocab_size=160_000,
        tgt_vocab_size=80_000,
        max_src_len=100,
        max_tgt_len=100,
        max_train_examples=None,
        batch_size=128,
        learning_rate=0.7,
        checkpoint_dir="runs/paper",
    )


def mid_config() -> Seq2SeqConfig:
    """Real WMT'14 slice sized for ~30–60 min on a single consumer GPU."""
    return Seq2SeqConfig(
        name="mid",
        num_layers=4,
        hidden_size=256,
        embed_size=256,
        src_vocab_size=20_000,
        tgt_vocab_size=20_000,
        max_src_len=50,
        max_tgt_len=50,
        max_train_examples=150_000,
        batch_size=64,
        learning_rate=0.7,
        epochs=3.0,
        lr_hold_epochs=2.0,
        lr_halve_every_epochs=0.5,
        device="cuda",
        log_every=20,
        sample_every=100,
        checkpoint_dir="runs/mid",
    )


def toy_config() -> Seq2SeqConfig:
    """Small runnable preset that preserves paper topology."""
    return Seq2SeqConfig(
        name="toy",
        num_layers=2,
        hidden_size=128,
        embed_size=128,
        src_vocab_size=8_000,
        tgt_vocab_size=8_000,
        max_src_len=40,
        max_tgt_len=40,
        max_train_examples=5_000,
        batch_size=32,
        learning_rate=0.7,
        epochs=2.0,
        lr_hold_epochs=1.0,
        lr_halve_every_epochs=0.5,
        log_every=5,
        sample_every=25,
        checkpoint_dir="runs/toy",
        max_steps=100,
    )


def get_config(name: ConfigName = "toy") -> Seq2SeqConfig:
    if name == "paper":
        return paper_config()
    if name == "mid":
        return mid_config()
    if name == "toy":
        return toy_config()
    raise ValueError(f"Unknown config: {name!r} (expected 'paper', 'mid', or 'toy')")
