"""TrainingMonitor: what updated, by how much, and sample translations.

Emits structured events the CLI and notebook can print or plot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn


@dataclass
class StepReport:
    step: int
    epoch: float
    loss: float
    ppl: float
    lr: float
    grad_norm_pre: float
    grad_norm_post: float
    clipped: bool
    delta_by_group: dict[str, float] = field(default_factory=dict)
    sample: str | None = None

    def summary(self) -> str:
        top = sorted(self.delta_by_group.items(), key=lambda kv: -kv[1])[:4]
        top_s = ", ".join(f"{k}={v:.3e}" for k, v in top)
        clip = " clipped" if self.clipped else ""
        base = (
            f"step={self.step} epoch={self.epoch:.3f} loss={self.loss:.4f} "
            f"ppl={self.ppl:.2f} lr={self.lr:.4g} "
            f"|g|={self.grad_norm_pre:.3f}→{self.grad_norm_post:.3f}{clip} "
            f"Δθ[{top_s}]"
        )
        if self.sample:
            return base + f"\n  sample: {self.sample}"
        return base


class TrainingMonitor:
    def __init__(
        self,
        model: nn.Module,
        param_groups: dict[str, list[nn.Parameter]],
        log_every: int = 10,
        sample_every: int = 50,
        sample_fn: Callable[[], str] | None = None,
        history: list[StepReport] | None = None,
    ):
        self.model = model
        self.param_groups = param_groups
        self.log_every = log_every
        self.sample_every = sample_every
        self.sample_fn = sample_fn
        self.history: list[StepReport] = history if history is not None else []
        self._snapshots: dict[str, list[torch.Tensor]] | None = None

    def snapshot_params(self) -> None:
        """Call before optimizer.step() to measure ‖Δθ‖ afterward."""
        self._snapshots = {
            name: [p.detach().clone() for p in params if p.requires_grad]
            for name, params in self.param_groups.items()
        }

    def group_deltas(self) -> dict[str, float]:
        if self._snapshots is None:
            return {}
        out: dict[str, float] = {}
        for name, old_list in self._snapshots.items():
            params = [p for p in self.param_groups[name] if p.requires_grad]
            total = 0.0
            for p, old in zip(params, old_list):
                total += torch.norm(p.detach() - old).item() ** 2
            out[name] = total**0.5
        return out

    def maybe_log(
        self,
        step: int,
        epoch: float,
        loss: float,
        lr: float,
        grad_norm_pre: float,
        grad_norm_post: float,
        clipped: bool,
        force: bool = False,
    ) -> StepReport | None:
        if not force and step % self.log_every != 0:
            return None
        sample = None
        if self.sample_fn is not None and (
            force or step % self.sample_every == 0
        ):
            with torch.no_grad():
                sample = self.sample_fn()
        report = StepReport(
            step=step,
            epoch=epoch,
            loss=loss,
            ppl=float(torch.exp(torch.tensor(min(loss, 20.0)))),
            lr=lr,
            grad_norm_pre=grad_norm_pre,
            grad_norm_post=grad_norm_post,
            clipped=clipped,
            delta_by_group=self.group_deltas(),
            sample=sample,
        )
        self.history.append(report)
        print(report.summary(), flush=True)
        return report

    def as_dict_history(self) -> list[dict[str, Any]]:
        return [
            {
                "step": r.step,
                "epoch": r.epoch,
                "loss": r.loss,
                "ppl": r.ppl,
                "lr": r.lr,
                "grad_norm_pre": r.grad_norm_pre,
                "grad_norm_post": r.grad_norm_post,
                "clipped": r.clipped,
                "delta_by_group": r.delta_by_group,
                "sample": r.sample,
            }
            for r in self.history
        ]
