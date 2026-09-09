# Configs: paper vs mid vs toy

Same code path (`get_config` → `train` / `decode`). Scale only.
Paper numbers: [Sutskever et al. 2014](https://arxiv.org/abs/1409.3215) §2–§3.4.

| | paper | mid | toy |
|--|-------|-----|-----|
| Role | document full run | 1-GPU WMT slice | laptop / notebook |
| L × H | 4 × 1000 | 4 × 256 | 2 × 128 |
| Embed | 1000 | 256 | 128 |
| \|V_src\| / \|V_tgt\| | 160k / 80k | 20k / 20k | 8k / 8k |
| Pairs | full WMT’14 | 150k | 5k or synthetic |
| Max len | 100 | 50 | 40 |
| Batch / epochs | 128 / 7.5 | 64 / 3 | 32 / 2 |
| LR hold | 5 ep | 2 ep | 1 ep |

**Kept in mid/toy:** reverse source, separate encoder/decoder, Graves cell, SGD+clip 5,
init ±0.08, length buckets, no attention.

**Dropped vs paper run:** ~384M scale, multi-GPU parallel, ensemble, SMT rescoring, BLEU target.

```bash
python -m seq2seq.train --config toy --synthetic --max-steps 20
python -m seq2seq.train --config mid --device cuda
```
