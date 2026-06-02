# Week 4 — Partial sources (Stage C)

How well can we value a property when only some sources are available
(e.g. WOZ only, source_1 + WOZ)? Two views: production cohorts (each scenario on its
available rows) and ablation (every scenario on identical all-source rows, to isolate
each source's marginal value). Finding: `source_1 + WOZ` is the coverage/accuracy
sweet spot.

Run (from repo root):

```bash
python -m experiments.week4_partial_sources.scripts.partial_sources
```

Outputs: `results/partial_sources.csv`, `plots/partial_sources.png`.
