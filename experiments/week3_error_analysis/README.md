# Week 3 — Error analysis (+ source 4)

Adds source 4 and studies where the RandomForest errs: a with/without-source-4
comparison, feature importance, and error sliced by price range, object type, and
municipality, plus an overall residual analysis.

Run (from repo root), e.g.:

```bash
python -m experiments.week3_error_analysis.scripts.source4_models
python -m experiments.week3_error_analysis.scripts.feature_importance
python -m experiments.week3_error_analysis.scripts.error_by_price_range
python -m experiments.week3_error_analysis.scripts.error_by_object_type
python -m experiments.week3_error_analysis.scripts.error_by_municipality
python -m experiments.week3_error_analysis.scripts.residuals
```

Outputs: `results/*.csv`, `plots/*.png`.
