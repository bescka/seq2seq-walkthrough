"""Unit tests for Graves LSTMCell vs a reference formulation."""

from __future__ import annotations

import torch
from torch import nn

from seq2seq.deep_lstm import DeepLSTM
from seq2seq.lstm_cell import LSTMCell


def _reference_step(
    x: torch.Tensor,
    h: torch.Tensor,
    c: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gates = x @ weight_ih.T + bias_ih + h @ weight_hh.T + bias_hh
    i, f, g, o = gates.chunk(4, dim=-1)
    i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
    c_new = f * c + i * g
    h_new = o * torch.tanh(c_new)
    return h_new, c_new


def test_lstm_cell_shapes():
    B, I, H = 3, 8, 16
    cell = LSTMCell(I, H)
    x = torch.randn(B, I)
    h = torch.zeros(B, H)
    c = torch.zeros(B, H)
    h2, c2 = cell(x, (h, c))
    assert h2.shape == (B, H)
    assert c2.shape == (B, H)


def test_lstm_cell_matches_reference():
    torch.manual_seed(0)
    B, I, H = 2, 5, 7
    cell = LSTMCell(I, H)
    # Fixed init
    nn.init.uniform_(cell.weight_ih, -0.1, 0.1)
    nn.init.uniform_(cell.weight_hh, -0.1, 0.1)
    nn.init.uniform_(cell.bias_ih, -0.1, 0.1)
    nn.init.uniform_(cell.bias_hh, -0.1, 0.1)

    x = torch.randn(B, I)
    h = torch.randn(B, H)
    c = torch.randn(B, H)

    h_a, c_a = cell(x, (h, c))
    h_b, c_b = _reference_step(
        x, h, c, cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh
    )
    assert torch.allclose(h_a, h_b, atol=1e-6)
    assert torch.allclose(c_a, c_b, atol=1e-6)


def test_lstm_cell_close_to_pytorch():
    """Align our weights with nn.LSTMCell and compare one step."""
    torch.manual_seed(1)
    B, I, H = 4, 6, 9
    ours = LSTMCell(I, H)
    ref = nn.LSTMCell(I, H)

    # Copy weights (PyTorch layout matches: weight_ih/hh are (4H, *) with i,f,g,o)
    with torch.no_grad():
        ours.weight_ih.copy_(ref.weight_ih)
        ours.weight_hh.copy_(ref.weight_hh)
        # nn.LSTMCell has bias_ih and bias_hh
        ours.bias_ih.copy_(ref.bias_ih)
        ours.bias_hh.copy_(ref.bias_hh)

    x = torch.randn(B, I)
    h = torch.randn(B, H)
    c = torch.randn(B, H)
    h_o, c_o = ours(x, (h, c))
    h_r, c_r = ref(x, (h, c))
    assert torch.allclose(h_o, h_r, atol=1e-5)
    assert torch.allclose(c_o, c_r, atol=1e-5)


def test_deep_lstm_shapes_and_final_state():
    torch.manual_seed(2)
    B, T, E, H, L = 3, 5, 8, 10, 2
    rnn = DeepLSTM(E, H, L)
    x = torch.randn(B, T, E)
    lengths = torch.tensor([5, 3, 4])
    out, (h, c) = rnn(x, lengths=lengths)
    assert out.shape == (B, T, H)
    assert h.shape == (L, B, H)
    assert c.shape == (L, B, H)

    # For shorter sequences, final state should equal state at t = length-1
    # Recompute step-by-step for batch item 1 (length 3).
    state = rnn.initial_state(1, x.device, x.dtype)
    h_last = None
    for t in range(3):
        h_last, state = rnn.step(x[1:2, t, :], state)
    assert torch.allclose(h[:, 1:2, :], state[0], atol=1e-5)
