"""Seq2Seq model: separate deep encoder/decoder LSTMs + naive softmax.

Paper objective (§2):

    p(y_1 .. y_{T'} | x) = ∏_t p(y_t | v, y_<t)

where v is the encoder's final hidden state (we pass full (h,c) over all
layers into the decoder as its initial state — see docs/architecture.md).

No attention. Softmax over the full target vocabulary at each step.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from seq2seq.config import Seq2SeqConfig
from seq2seq.deep_lstm import DeepLSTM


class Seq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        pad_id: int,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.src_embed = nn.Embedding(src_vocab_size, embed_size, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, embed_size, padding_idx=pad_id)

        # Separate LSTMs for source and target (paper §2).
        self.encoder = DeepLSTM(embed_size, hidden_size, num_layers)
        self.decoder = DeepLSTM(embed_size, hidden_size, num_layers)

        # Naive softmax projection: H → |V_tgt|
        self.out_proj = nn.Linear(hidden_size, tgt_vocab_size)

    @classmethod
    def from_config(
        cls, cfg: Seq2SeqConfig, src_vocab_size: int, tgt_vocab_size: int, pad_id: int
    ) -> Seq2Seq:
        return cls(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            embed_size=cfg.embed_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            pad_id=pad_id,
        )

    def init_weights(self, init_range: float = 0.08) -> None:
        """Paper: Uniform(-0.08, 0.08) for all LSTM parameters."""
        for p in self.parameters():
            if p.dim() > 0:
                nn.init.uniform_(p, -init_range, init_range)

    def encode(
        self, src: torch.Tensor, src_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """src: (B, T_src) → encoder final (h, c) each (L, B, H)."""
        emb = self.src_embed(src)
        _, state = self.encoder(emb, lengths=src_lengths)
        return state

    def decode_step(
        self,
        y_t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """One decoder step. y_t: (B,) token ids → logits (B, V), new state."""
        emb = self.tgt_embed(y_t)
        h_top, state = self.decoder.step(emb, state)
        logits = self.out_proj(h_top)
        return logits, state

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_in: torch.Tensor,
        tgt_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced training forward.

        Returns:
            loss: scalar mean NLL over non-pad target tokens
            logits: (B, T_tgt, V) for monitoring / sampling
        """
        state = self.encode(src, src_lengths)
        emb = self.tgt_embed(tgt_in)  # (B, T, E)
        # Run full decoder sequence initialized from encoder state.
        dec_out, _ = self.decoder(emb, lengths=None, state=state)
        logits = self.out_proj(dec_out)  # (B, T, V)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=self.pad_id,
        )
        return loss, logits

    def param_groups_for_monitor(self) -> dict[str, list[nn.Parameter]]:
        """Named groups so TrainingMonitor can show ‖Δθ‖ per block."""
        groups: dict[str, list[nn.Parameter]] = {
            "src_embed": list(self.src_embed.parameters()),
            "tgt_embed": list(self.tgt_embed.parameters()),
            "softmax": list(self.out_proj.parameters()),
        }
        for i, cell in enumerate(self.encoder.cells):
            groups[f"encoder.layer{i}"] = list(cell.parameters())
        for i, cell in enumerate(self.decoder.cells):
            groups[f"decoder.layer{i}"] = list(cell.parameters())
        return groups
