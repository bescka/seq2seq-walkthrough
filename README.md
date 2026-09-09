# Seq2Seq — Sutskever et al. 2014

Pedagogical PyTorch replica of [*Sequence to Sequence Learning with Neural Networks*](https://arxiv.org/abs/1409.3215) ([HTML](https://arxiv.org/html/1409.3215v3)).

| Doc | Content |
|-----|---------|
| [docs/architecture.md](docs/architecture.md) | Objective, reversal, LSTM, train/decode |
| [docs/configs.md](docs/configs.md) | `paper` / `mid` / `toy` |
| [notebooks/01_walkthrough.ipynb](notebooks/01_walkthrough.ipynb) | Runnable walkthrough |

**In scope:** reversed-source encoder → fixed \(v\) → decoder LM; Graves LSTM; no attention.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

GPU images that already ship CUDA PyTorch — do not let pip replace it:

```bash
pip install -e ".[dev]" --no-deps
pip install datasets numpy tqdm pytest
```

## Commands

```bash
python -m seq2seq.train --config toy --synthetic --max-steps 50
python -m seq2seq.train --config mid --device cuda
python -m seq2seq.decode --checkpoint runs/mid/checkpoint.pt --beam 2 \
  --src "the cat sat on the mat" --device cuda
```

## Layout

```
src/seq2seq/   # implementation
tests/
docs/
notebooks/
scripts/run_mid.sh
```

## Reference

Sutskever, Vinyals, Le. *Sequence to Sequence Learning with Neural Networks*. NeurIPS 2014. [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)
