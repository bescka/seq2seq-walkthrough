"""WMT'14 

En→Fr data: tokenize, reverse source, vocabs, length-bucketed batches.

Paper §3.1 / §3.4:
- Fixed vocabularies; OOV → UNK
- Source sentences reversed (targets not reversed)
- Minibatches grouped by similar length (≈2× speedup in the paper)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from seq2seq.config import Seq2SeqConfig

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"
SOS_TOKEN = "<SOS>"  # decoder start; paper uses previous token/ BOS convention

SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, EOS_TOKEN, SOS_TOKEN)


_WORD_RE = re.compile(r"\S+")


def basic_tokenize(text: str) -> list[str]:
    """Paper-era whitespace tokenization (not BPE/SentencePiece)."""
    return _WORD_RE.findall(text.strip().lower())


@dataclass
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    @property
    def sos_id(self) -> int:
        return self.token_to_id[SOS_TOKEN]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(self, tokens: list[str], add_eos: bool = True) -> list[int]:
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        toks: list[str] = []
        special = {self.pad_id, self.eos_id, self.sos_id}
        for i in ids:
            if i < 0 or i >= len(self.id_to_token):
                continue
            if skip_special and i in special:
                if i == self.eos_id:
                    break
                continue
            toks.append(self.id_to_token[i])
        return " ".join(toks)

    @classmethod
    def build(cls, texts: list[list[str]], max_size: int) -> Vocab:
        counts: Counter[str] = Counter()
        for toks in texts:
            counts.update(toks)
        # Reserve specials; fill remaining slots by frequency.
        id_to_token = list(SPECIAL_TOKENS)
        for tok, _ in counts.most_common(max(0, max_size - len(SPECIAL_TOKENS))):
            if tok not in SPECIAL_TOKENS:
                id_to_token.append(tok)
        token_to_id = {t: i for i, t in enumerate(id_to_token)}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)


@dataclass
class ParallelExample:
    src_ids: list[int]  # already reversed + EOS
    tgt_ids: list[int]  # EOS at end; SOS prepended at batch time for inputs
    src_len: int
    tgt_len: int


class ParallelDataset(Dataset):
    def __init__(self, examples: list[ParallelExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ParallelExample:
        return self.examples[idx]


class BucketBatchSampler(Sampler[list[int]]):
    """Group indices by source length so minibatch padding waste is small (§3.4)."""

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        bucket_width: int = 5,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_width = bucket_width
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        buckets: dict[int, list[int]] = {}
        for i, length in enumerate(lengths):
            key = length // bucket_width
            buckets.setdefault(key, []).append(i)
        self.buckets = buckets

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        batches: list[list[int]] = []
        for indices in self.buckets.values():
            idxs = list(indices)
            if self.shuffle:
                perm = torch.randperm(len(idxs), generator=g).tolist()
                idxs = [idxs[i] for i in perm]
            for start in range(0, len(idxs), self.batch_size):
                batch = idxs[start : start + self.batch_size]
                if batch:
                    batches.append(batch)
        if self.shuffle:
            order = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in order]
        yield from batches

    def __len__(self) -> int:
        return sum(
            (len(v) + self.batch_size - 1) // self.batch_size
            for v in self.buckets.values()
        )


def collate_batch(
    batch: list[ParallelExample],
    src_pad: int,
    tgt_pad: int,
    tgt_sos: int,
) -> dict[str, torch.Tensor]:
    """Pad and build teacher-forcing tensors.

    Decoder input:  <SOS> y1 y2 ... y_{T'-1}
    Decoder target: y1 y2 ... y_{T'}   (includes <EOS>)
    """
    src = [torch.tensor(ex.src_ids, dtype=torch.long) for ex in batch]
    tgt = [torch.tensor(ex.tgt_ids, dtype=torch.long) for ex in batch]

    src_pad_t = pad_sequence(src, batch_first=True, padding_value=src_pad)
    tgt_pad_t = pad_sequence(tgt, batch_first=True, padding_value=tgt_pad)

    # Teacher-forcing inputs: SOS + target[:-1]
    bos = torch.full((len(batch), 1), tgt_sos, dtype=torch.long)
    tgt_in = torch.cat([bos, tgt_pad_t[:, :-1]], dim=1)
    tgt_out = tgt_pad_t

    src_lengths = torch.tensor([ex.src_len for ex in batch], dtype=torch.long)
    tgt_lengths = torch.tensor([ex.tgt_len for ex in batch], dtype=torch.long)

    return {
        "src": src_pad_t,
        "tgt_in": tgt_in,
        "tgt_out": tgt_out,
        "src_lengths": src_lengths,
        "tgt_lengths": tgt_lengths,
    }


def _load_raw_pairs(cfg: Seq2SeqConfig) -> list[tuple[str, str]]:
    from datasets import load_dataset

    # Prefer streaming-friendly split; fall back to full train if needed.
    ds = load_dataset(cfg.dataset, cfg.dataset_config, split="train")
    pairs: list[tuple[str, str]] = []
    n = cfg.max_train_examples
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        tr = row["translation"]
        src = tr[cfg.src_lang]
        tgt = tr[cfg.tgt_lang]
        if src and tgt:
            pairs.append((src, tgt))
    return pairs


def prepare_data(
    cfg: Seq2SeqConfig,
) -> tuple[DataLoader, Vocab, Vocab, list[ParallelExample]]:
    """Download/tokenize WMT slice, build vocabs, return bucketed DataLoader."""
    pairs = _load_raw_pairs(cfg)

    src_toks = [basic_tokenize(s) for s, _ in pairs]
    tgt_toks = [basic_tokenize(t) for _, t in pairs]

    # Filter by max length (before reverse; count content tokens).
    filtered: list[tuple[list[str], list[str]]] = []
    for s, t in zip(src_toks, tgt_toks):
        if 1 <= len(s) <= cfg.max_src_len and 1 <= len(t) <= cfg.max_tgt_len:
            filtered.append((s, t))
    if not filtered:
        raise RuntimeError("No examples left after length filtering.")

    src_toks = [s for s, _ in filtered]
    tgt_toks = [t for _, t in filtered]

    src_vocab = Vocab.build(src_toks, cfg.src_vocab_size)
    tgt_vocab = Vocab.build(tgt_toks, cfg.tgt_vocab_size)

    examples: list[ParallelExample] = []
    for s, t in zip(src_toks, tgt_toks):
        # Paper: reverse source word order (not the target).
        s_use = list(reversed(s)) if cfg.reverse_source else list(s)
        src_ids = src_vocab.encode(s_use, add_eos=True)
        tgt_ids = tgt_vocab.encode(t, add_eos=True)
        examples.append(
            ParallelExample(
                src_ids=src_ids,
                tgt_ids=tgt_ids,
                src_len=len(src_ids),
                tgt_len=len(tgt_ids),
            )
        )

    dataset = ParallelDataset(examples)
    sampler = BucketBatchSampler(
        lengths=[ex.src_len for ex in examples],
        batch_size=cfg.batch_size,
        bucket_width=cfg.bucket_width,
        shuffle=True,
        seed=cfg.seed,
    )

    def _collate(batch: list[ParallelExample]) -> dict[str, torch.Tensor]:
        return collate_batch(
            batch, src_vocab.pad_id, tgt_vocab.pad_id, tgt_vocab.sos_id
        )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=_collate,
        num_workers=cfg.num_workers,
    )
    return loader, src_vocab, tgt_vocab, examples


def make_synthetic_loader(
    cfg: Seq2SeqConfig,
    n: int = 64,
) -> tuple[DataLoader, Vocab, Vocab, list[ParallelExample]]:
    """Tiny deterministic data for tests / offline smoke (no HF download)."""
    # Identity-ish parallel pairs over a tiny word list.
    words = [f"w{i}" for i in range(20)]
    pairs_tok = []
    for i in range(n):
        L = 3 + (i % 5)
        s = [words[(i + j) % len(words)] for j in range(L)]
        t = list(s)  # copy — still useful for shape smoke
        pairs_tok.append((s, t))

    src_vocab = Vocab.build([s for s, _ in pairs_tok], cfg.src_vocab_size)
    tgt_vocab = Vocab.build([t for _, t in pairs_tok], cfg.tgt_vocab_size)

    examples: list[ParallelExample] = []
    for s, t in pairs_tok:
        s_use = list(reversed(s)) if cfg.reverse_source else list(s)
        src_ids = src_vocab.encode(s_use, add_eos=True)
        tgt_ids = tgt_vocab.encode(t, add_eos=True)
        examples.append(ParallelExample(src_ids, tgt_ids, len(src_ids), len(tgt_ids)))

    dataset = ParallelDataset(examples)
    sampler = BucketBatchSampler(
        [ex.src_len for ex in examples],
        batch_size=min(cfg.batch_size, 16),
        bucket_width=cfg.bucket_width,
        shuffle=False,
        seed=cfg.seed,
    )

    def _collate(batch: list[ParallelExample]) -> dict[str, torch.Tensor]:
        return collate_batch(
            batch, src_vocab.pad_id, tgt_vocab.pad_id, tgt_vocab.sos_id
        )

    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=_collate)
    return loader, src_vocab, tgt_vocab, examples
