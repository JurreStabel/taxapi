# Week 2 — WOZ model

First RandomForest that adds the official WOZ valuation to sources 1-3 (no source 4).
Reuses the shared modeling pipeline (`taxapi.core.modeling`) and reports feature
importance — WOZ immediately becomes a dominant feature.

Run (from repo root):

```bash
python -m experiments.week2_woz_model.scripts.woz_model
```

Outputs: `results/woz_feature_importance.csv`, `plots/feature_importance.png`.
See `progress_week2.docx` for the original write-up.
