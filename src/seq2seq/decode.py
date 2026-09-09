"""Greedy and beam-search decoding (paper §3.2).

Left-to-right beam search: keep B partial hypotheses, expand by vocab,
prune to top B. Hypotheses that emit <EOS> are completed.
Paper notes beam 1 works; beam 2 captures most of the gain.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from seq2seq.config import get_config
from seq2seq.data import Vocab, basic_tokenize
from seq2seq.model import Seq2Seq


@dataclass
class Hypothesis:
    tokens: list[int]
    score: float  # sum log-probs
    state: tuple[torch.Tensor, torch.Tensor]
    finished: bool = False

    def avg_score(self) -> float:
        # Length-normalized for ranking completed hyps (simple; paper used raw sum in beam).
        n = max(len(self.tokens), 1)
        return self.score / n


@torch.no_grad()
def greedy_decode(
    model: Seq2Seq,
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    sos_id: int,
    eos_id: int,
    max_len: int,
) -> list[list[int]]:
    """src: (B, T) → list of token-id sequences (no SOS; stops at EOS)."""
    model.eval()
    state = model.encode(src, src_lengths)
    batch = src.size(0)
    device = src.device
    y = torch.full((batch,), sos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    outputs: list[list[int]] = [[] for _ in range(batch)]

    for _ in range(max_len):
        logits, state = model.decode_step(y, state)
        y = logits.argmax(dim=-1)
        for b in range(batch):
            if not finished[b]:
                tok = int(y[b].item())
                outputs[b].append(tok)
                if tok == eos_id:
                    finished[b] = True
        if bool(finished.all()):
            break
    return outputs


@torch.no_grad()
def beam_search_decode(
    model: Seq2Seq,
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    sos_id: int,
    eos_id: int,
    max_len: int,
    beam_size: int = 2,
) -> list[list[int]]:
    """Single-sequence or batched beam search; returns best hyp per batch item."""
    model.eval()
    batch = src.size(0)
    results: list[list[int]] = []
    for b in range(batch):
        results.append(
            _beam_one(
                model,
                src[b : b + 1],
                src_lengths[b : b + 1],
                sos_id,
                eos_id,
                max_len,
                beam_size,
            )
        )
    return results


def _beam_one(
    model: Seq2Seq,
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    sos_id: int,
    eos_id: int,
    max_len: int,
    beam_size: int,
) -> list[int]:
    state = model.encode(src, src_lengths)
    # Expand state tensors along batch for beam later.
    hyps = [
        Hypothesis(
            tokens=[],
            score=0.0,
            state=state,
            finished=False,
        )
    ]
    completed: list[Hypothesis] = []

    for _ in range(max_len):
        open_hyps = [h for h in hyps if not h.finished]
        if not open_hyps:
            break
        # Score next token for each open hyp (beam usually small).
        candidates: list[Hypothesis] = []
        for h in open_hyps:
            last = sos_id if not h.tokens else h.tokens[-1]
            y = torch.tensor([last], dtype=torch.long, device=src.device)
            logits, new_state = model.decode_step(y, h.state)
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # (V,)
            topk = torch.topk(log_probs, k=beam_size)
            for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                tok = int(idx)
                new_tokens = h.tokens + [tok]
                done = tok == eos_id
                nh = Hypothesis(
                    tokens=new_tokens,
                    score=h.score + float(score),
                    state=new_state,
                    finished=done,
                )
                if done:
                    completed.append(nh)
                else:
                    candidates.append(nh)
        # Also keep already-finished from previous round (none in open)
        candidates.sort(key=lambda h: h.score, reverse=True)
        hyps = candidates[:beam_size]
        if len(completed) >= beam_size and (
            not hyps or completed[0].score >= hyps[0].score
        ):
            # Early-ish stop when enough completed and best open can't beat
            pass

    pool = completed + hyps
    if not pool:
        return []
    best = max(pool, key=lambda h: h.score)
    # Strip trailing EOS for cleaner decode strings
    toks = best.tokens
    if toks and toks[-1] == eos_id:
        toks = toks[:-1]
    return toks


def encode_source_sentence(
    text: str,
    src_vocab: Vocab,
    reverse: bool = True,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    toks = basic_tokenize(text)
    if reverse:
        toks = list(reversed(toks))
    ids = src_vocab.encode(toks, add_eos=True)
    src = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)
    return src, lengths


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Decode with a trained seq2seq checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--src", type=str, required=True, help="English source sentence")
    parser.add_argument("--beam", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    ckpt = load_checkpoint(args.checkpoint, args.device)
    cfg = get_config(ckpt.get("config_name", "toy"))
    src_vocab: Vocab = ckpt["src_vocab"]
    tgt_vocab: Vocab = ckpt["tgt_vocab"]

    model = Seq2Seq.from_config(cfg, len(src_vocab), len(tgt_vocab), tgt_vocab.pad_id)
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    model.eval()

    src, lengths = encode_source_sentence(
        args.src, src_vocab, reverse=cfg.reverse_source, device=torch.device(args.device)
    )
    if args.beam <= 1:
        ids = greedy_decode(
            model, src, lengths, tgt_vocab.sos_id, tgt_vocab.eos_id, cfg.max_decode_len
        )[0]
    else:
        ids = beam_search_decode(
            model,
            src,
            lengths,
            tgt_vocab.sos_id,
            tgt_vocab.eos_id,
            cfg.max_decode_len,
            beam_size=args.beam,
        )[0]
    print(tgt_vocab.decode(ids) or "<empty>")


if __name__ == "__main__":
    main()
