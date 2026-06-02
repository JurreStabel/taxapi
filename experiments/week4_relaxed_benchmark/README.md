# Week 4 — Relaxed benchmark (Stage B)

Compares the strict "gold" benchmark with a relaxed one that also accepts valuations
up to +30 days after the ground-truth date (`builders.*(relaxed=True)`). Reports the
coverage gain and RF accuracy (GroupKFold). Finding: +35% rows, MAPE unchanged.

Prereq: build both benchmark chains first:

```bash
python -m taxapi.pipelines.train             # gold
python -m taxapi.pipelines.train --relaxed   # relaxed
```

Run (from repo root):

```bash
python -m experiments.week4_relaxed_benchmark.scripts.relaxed_compare
```

Outputs: `results/relaxed_vs_gold.csv`, `plots/{coverage,mape_stability}.png`.
