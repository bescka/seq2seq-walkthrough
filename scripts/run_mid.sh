#!/usr/bin/env bash
# Mid-scale WMT'14 En→Fr on one GPU. Prefer --no-deps if the image already has CUDA torch.
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -e ".[dev]" --no-deps -q
pip install datasets numpy tqdm pytest -q
python -m seq2seq.train --config mid --device cuda "$@"
