"""Graves-style LSTM cell (Sutskever et al. cite Graves 2013).

For input x_t and previous state (h_{t-1}, c_{t-1}):

    i_t = σ(W_xi x_t + W_hi h_{t-1} + b_i)     # input gate
    f_t = σ(W_xf x_t + W_hf h_{t-1} + b_f)     # forget gate
    g_t = tanh(W_xg x_t + W_hg h_{t-1} + b_g)  # cell candidate
    o_t = σ(W_xo x_t + W_ho h_{t-1} + b_o)     # output gate

    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    h_t = o_t ⊙ tanh(c_t)

We pack the eight weight matrices into two linear maps (input + hidden)
for efficiency, matching the usual "four gates concatenated" layout:
[i | f | g | o].
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMCell(nn.Module):
    """Single-step LSTM; shapes: x (B, input_size) → h,c (B, hidden_size)."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Combined affine: maps to 4 * hidden (i, f, g, o)
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self, init_range: float = 0.08) -> None:
        """Paper-style uniform init (Seq2Seq.init_weights may overwrite)."""
        for p in self.parameters():
            nn.init.uniform_(p, -init_range, init_range)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        # (B, 4H)
        gates = (
            x @ self.weight_ih.T
            + self.bias_ih
            + h_prev @ self.weight_hh.T
            + self.bias_hh
        )
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c
