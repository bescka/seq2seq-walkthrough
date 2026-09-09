"""Shape / smoke tests for data, model, train, decode."""

from __future__ import annotations

import torch

from seq2seq.config import toy_config
from seq2seq.data import Vocab, basic_tokenize, collate_batch, make_synthetic_loader
from seq2seq.decode import beam_search_decode, greedy_decode
from seq2seq.model import Seq2Seq
from seq2seq.train import clip_grad_norm_, learning_rate_at_epoch, train


def test_basic_tokenize_and_reverse_intent():
    toks = basic_tokenize("The Cat sat.")
    assert toks == ["the", "cat", "sat."]
    assert list(reversed(toks)) == ["sat.", "cat", "the"]


def test_vocab_unk_and_eos():
    v = Vocab.build([["a", "b"], ["a", "c"]], max_size=10)
    ids = v.encode(["a", "z"], add_eos=True)
    assert ids[-1] == v.eos_id
    assert ids[1] == v.unk_id


def test_collate_teacher_forcing_shapes():
    cfg = toy_config()
    loader, src_vocab, tgt_vocab, _ = make_synthetic_loader(cfg, n=32)
    batch = next(iter(loader))
    B = batch["src"].size(0)
    assert batch["src"].dim() == 2
    assert batch["tgt_in"].shape == batch["tgt_out"].shape
    assert batch["tgt_in"].size(0) == B
    assert batch["src_lengths"].tolist()
    # First decoder input token is SOS
    assert (batch["tgt_in"][:, 0] == tgt_vocab.sos_id).all()


def test_seq2seq_forward_loss_and_init():
    cfg = toy_config()
    cfg.hidden_size = 32
    cfg.embed_size = 32
    cfg.num_layers = 2
    loader, src_vocab, tgt_vocab, _ = make_synthetic_loader(cfg, n=16)
    batch = next(iter(loader))
    model = Seq2Seq.from_config(cfg, len(src_vocab), len(tgt_vocab), tgt_vocab.pad_id)
    model.init_weights(0.08)
    # Check init range
    for p in model.parameters():
        assert p.min() >= -0.08 - 1e-6
        assert p.max() <= 0.08 + 1e-6
    loss, logits = model(
        batch["src"], batch["src_lengths"], batch["tgt_in"], batch["tgt_out"]
    )
    assert loss.ndim == 0
    assert logits.shape[:2] == batch["tgt_out"].shape
    assert logits.size(-1) == len(tgt_vocab)
    loss.backward()


def test_lr_schedule_and_clip():
    cfg = toy_config()
    assert learning_rate_at_epoch(cfg, 0.5) == cfg.learning_rate
    assert learning_rate_at_epoch(cfg, cfg.lr_hold_epochs) < cfg.learning_rate

    p = torch.nn.Parameter(torch.zeros(10))
    p.grad = torch.ones(10) * 10.0
    pre, post, clipped = clip_grad_norm_([p], max_norm=5.0)
    assert clipped
    assert post <= 5.0 + 1e-4


def test_greedy_and_beam_decode():
    cfg = toy_config()
    cfg.hidden_size = 24
    cfg.embed_size = 24
    cfg.num_layers = 2
    loader, src_vocab, tgt_vocab, _ = make_synthetic_loader(cfg, n=8)
    batch = next(iter(loader))
    model = Seq2Seq.from_config(cfg, len(src_vocab), len(tgt_vocab), tgt_vocab.pad_id)
    model.init_weights(0.08)
    model.eval()
    g = greedy_decode(
        model,
        batch["src"][:2],
        batch["src_lengths"][:2],
        tgt_vocab.sos_id,
        tgt_vocab.eos_id,
        max_len=10,
    )
    b = beam_search_decode(
        model,
        batch["src"][:2],
        batch["src_lengths"][:2],
        tgt_vocab.sos_id,
        tgt_vocab.eos_id,
        max_len=10,
        beam_size=2,
    )
    assert len(g) == 2 and len(b) == 2
    assert isinstance(g[0], list)


def test_train_synthetic_smoke(tmp_path):
    cfg = toy_config()
    cfg.hidden_size = 32
    cfg.embed_size = 32
    cfg.num_layers = 2
    cfg.batch_size = 8
    cfg.max_steps = 5
    cfg.log_every = 1
    cfg.sample_every = 5
    cfg.checkpoint_dir = str(tmp_path / "run")
    cfg.epochs = 10
    monitor = train(cfg, synthetic=True, sample_src="w0 w1 w2")
    assert len(monitor.history) >= 1
    assert (tmp_path / "run" / "checkpoint.pt").exists()
