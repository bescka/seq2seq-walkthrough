"""Training loop: paper SGD recipe + TrainingMonitor.

Paper §3.4:
- SGD without momentum, LR 0.7
- After `lr_hold_epochs`, halve LR every `lr_halve_every_epochs`
- Gradient averaged over batch; clip if ‖g‖₂ > 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from seq2seq.config import Seq2SeqConfig, get_config
from seq2seq.data import Vocab, make_synthetic_loader, prepare_data
from seq2seq.decode import encode_source_sentence, greedy_decode
from seq2seq.model import Seq2Seq
from seq2seq.monitor import TrainingMonitor


def learning_rate_at_epoch(cfg: Seq2SeqConfig, epoch: float) -> float:
    """Piecewise LR: flat until lr_hold_epochs, then halve every lr_halve_every_epochs."""
    lr = cfg.learning_rate
    if epoch < cfg.lr_hold_epochs:
        return lr
    # Number of halving intervals since hold ended.
    past = epoch - cfg.lr_hold_epochs
    n_halves = int(past / cfg.lr_halve_every_epochs) + 1
    return lr / (2**n_halves)


def total_grad_norm(params: list[nn.Parameter]) -> float:
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2).item()


def clip_grad_norm_(params: list[nn.Parameter], max_norm: float) -> tuple[float, float, bool]:
    """Hard constraint: if s=‖g‖₂ > max_norm, g ← max_norm * g / s."""
    pre = total_grad_norm(params)
    clipped = pre > max_norm
    if clipped and pre > 0:
        scale = max_norm / pre
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)
    post = total_grad_norm(params)
    return pre, post, clipped


def save_checkpoint(
    path: Path,
    model: Seq2Seq,
    cfg: Seq2SeqConfig,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.to_dict(),
            "config_name": cfg.name,
            "src_vocab": src_vocab,
            "tgt_vocab": tgt_vocab,
            "step": step,
        },
        path,
    )


def train(
    cfg: Seq2SeqConfig,
    *,
    synthetic: bool = False,
    sample_src: str = "the cat sat on the mat",
) -> TrainingMonitor:
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        device = torch.device("cpu")

    if synthetic:
        loader, src_vocab, tgt_vocab, _ = make_synthetic_loader(cfg)
    else:
        loader, src_vocab, tgt_vocab, _ = prepare_data(cfg)

    # Vocab may be smaller than configured max — use actual sizes.
    model = Seq2Seq.from_config(cfg, len(src_vocab), len(tgt_vocab), tgt_vocab.pad_id)
    model.init_weights(cfg.init_range)
    model.to(device)

    # SGD without momentum; loss already mean-reduced → grads are batch-averaged.
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.0)

    def sample_fn() -> str:
        model.eval()
        src_t, lens = encode_source_sentence(
            sample_src, src_vocab, reverse=cfg.reverse_source, device=device
        )
        # On synthetic data the English sentence may be all UNK — still shows pipeline.
        ids = greedy_decode(
            model, src_t, lens, tgt_vocab.sos_id, tgt_vocab.eos_id, cfg.max_decode_len
        )[0]
        out = tgt_vocab.decode(ids)
        model.train()
        return f"{sample_src!r} → {out!r}"

    monitor = TrainingMonitor(
        model,
        model.param_groups_for_monitor(),
        log_every=cfg.log_every,
        sample_every=cfg.sample_every,
        sample_fn=sample_fn,
    )

    steps_per_epoch = max(len(loader), 1)
    global_step = 0
    model.train()
    epoch_idx = 0
    max_steps = cfg.max_steps

    while True:
        # BucketBatchSampler epoch seed
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch_idx)

        pbar = tqdm(loader, desc=f"epoch {epoch_idx}", leave=False)
        for batch in pbar:
            epoch_frac = epoch_idx + (global_step % steps_per_epoch) / steps_per_epoch
            if epoch_frac >= cfg.epochs:
                break
            if max_steps is not None and global_step >= max_steps:
                break

            lr = learning_rate_at_epoch(cfg, epoch_frac)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)
            src_lengths = batch["src_lengths"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(src, src_lengths, tgt_in, tgt_out)
            loss.backward()

            params = [p for p in model.parameters() if p.requires_grad]
            pre, post, clipped = clip_grad_norm_(params, cfg.grad_clip)

            monitor.snapshot_params()
            optimizer.step()

            global_step += 1
            monitor.maybe_log(
                step=global_step,
                epoch=epoch_frac,
                loss=float(loss.item()),
                lr=lr,
                grad_norm_pre=pre,
                grad_norm_post=post,
                clipped=clipped,
            )
            pbar.set_postfix(loss=float(loss.item()), lr=lr)

        epoch_idx += 1
        done_epochs = epoch_idx >= cfg.epochs
        done_steps = max_steps is not None and global_step >= max_steps
        if done_epochs or done_steps:
            break

    ckpt_dir = Path(cfg.checkpoint_dir)
    save_checkpoint(ckpt_dir / "checkpoint.pt", model, cfg, src_vocab, tgt_vocab, global_step)
    hist_path = ckpt_dir / "monitor_history.json"
    hist_path.write_text(json.dumps(monitor.as_dict_history(), indent=2))
    print(f"Saved checkpoint → {ckpt_dir / 'checkpoint.pt'}")
    monitor.maybe_log(
        step=global_step,
        epoch=min(epoch_idx, cfg.epochs),
        loss=monitor.history[-1].loss if monitor.history else 0.0,
        lr=learning_rate_at_epoch(cfg, min(epoch_idx, cfg.epochs)),
        grad_norm_pre=0.0,
        grad_norm_post=0.0,
        clipped=False,
        force=True,
    )
    return monitor


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Sutskever seq2seq (pedagogical)")
    parser.add_argument("--config", type=str, default="toy", choices=["toy", "mid", "paper"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (no Hugging Face download)",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args(argv)

    cfg = get_config(args.config)  # type: ignore[arg-type]
    if args.device:
        cfg.device = args.device
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
        # Don't let the epoch budget stop a short smoke before max_steps.
        cfg.epochs = max(cfg.epochs, 1_000.0)
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir

    train(cfg, synthetic=args.synthetic)


if __name__ == "__main__":
    main()
