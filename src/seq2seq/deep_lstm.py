"""Deep (stacked) LSTM built from our Graves LSTMCell.

Manual timestep loop keeps shapes pedagogical:

    For t in 1..T:
      layer_0 input = x_t
      for ℓ in 0..L-1:
        h^ℓ_t, c^ℓ_t = LSTM^ℓ(input, (h^ℓ_{t-1}, c^ℓ_{t-1}))
        input = h^ℓ_t   # feed upward

Encoder final state is the tuple of all layers' (h_T, c_T) — the paper's
fixed-dimensional sentence representation when read at the top layer
(and we pass all layers to the decoder as initial state).
"""

from __future__ import annotations

import torch
from torch import nn

from seq2seq.lstm_cell import LSTMCell


class DeepLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        cells: list[LSTMCell] = []
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            cells.append(LSTMCell(in_size, hidden_size))
        self.cells = nn.ModuleList(cells)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zeros: (num_layers, B, H) for h and c."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype)
        return h, c

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, T, input_size) — already embedded
            lengths: (B,) valid lengths (for packing mask); if None, use full T
            state: optional (h, c) each (L, B, H)

        Returns:
            outputs: (B, T, H) top-layer hidden states per timestep
            final_state: (h, c) each (L, B, H) — last *valid* timestep per layer
        """
        batch, timesteps, _ = x.shape
        device, dtype = x.device, x.dtype
        if state is None:
            h, c = self.initial_state(batch, device, dtype)
        else:
            h, c = state

        if lengths is None:
            lengths = torch.full((batch,), timesteps, device=device, dtype=torch.long)
        else:
            lengths = lengths.to(device)

        # outputs[t] = top-layer h_t
        outputs = x.new_zeros(batch, timesteps, self.hidden_size)

        # Keep last valid state per example (for variable lengths).
        final_h = h.clone()
        final_c = c.clone()

        for t in range(timesteps):
            inp = x[:, t, :]
            new_h_layers = []
            new_c_layers = []
            for layer, cell in enumerate(self.cells):
                h_l, c_l = cell(inp, (h[layer], c[layer]))
                new_h_layers.append(h_l)
                new_c_layers.append(c_l)
                inp = h_l  # next layer input
            h = torch.stack(new_h_layers, dim=0)
            c = torch.stack(new_c_layers, dim=0)
            outputs[:, t, :] = h[-1]

            # Update final state where timestep t is still inside the sequence.
            # lengths are 1-indexed counts; valid indices are 0 .. length-1.
            alive = (t < lengths).view(1, batch, 1)
            final_h = torch.where(alive, h, final_h)
            final_c = torch.where(alive, c, final_c)

        return outputs, (final_h, final_c)

    def step(
        self,
        x_t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single timestep for decoding. x_t: (B, input_size)."""
        h, c = state
        inp = x_t
        new_h_layers = []
        new_c_layers = []
        for layer, cell in enumerate(self.cells):
            h_l, c_l = cell(inp, (h[layer], c[layer]))
            new_h_layers.append(h_l)
            new_c_layers.append(c_l)
            inp = h_l
        h = torch.stack(new_h_layers, dim=0)
        c = torch.stack(new_c_layers, dim=0)
        return h[-1], (h, c)
