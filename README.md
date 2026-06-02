# taxapi

Benchmarking model-based valuations of Dutch residential property against ground
truth (WOZ + model sources + ground truth). A shared library builds a leakage-safe
benchmark; each week's analysis lives as a self-contained experiment.

## Layout

```
taxapi/                         shared library
  core/        paths, metrics, modeling, features, splitting, matching, runner
  builders/    clean.py, woz.py, source4.py   (build the benchmark datasets)
  pipelines/   train.py                        (run the builders end to end)
  exploration/ inspect_data.py                 (profile the raw data)
experiments/                    one folder per chapter; each has scripts/ plots/ results/ README
  week1_source_evaluation/      sources 1-3 as standalone predictors
  week2_woz_model/              first RandomForest with WOZ
  week3_error_analysis/         + source 4; error by price/type/municipality, residuals
  week4_relaxed_benchmark/      Stage B: relaxed (+30d) benchmark, coverage vs accuracy
  week4_partial_sources/        Stage C: partial-source scenarios
data/                           gitignored: raw/ + benchmark/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the raw files in `data/raw/` (`ground_truth_data.csv`, `model_values_data.csv`,
`woz_values.csv`, `model_values_source_4_data.csv`).

## Build the benchmark

```bash
python -m taxapi.pipelines.train             # gold / strict   -> data/benchmark/
python -m taxapi.pipelines.train --relaxed   # relaxed +30d     -> data/benchmark/relaxed/
```

## Run an experiment

All experiments run as modules from the repo root, e.g.:

```bash
python -m experiments.week4_relaxed_benchmark.scripts.relaxed_compare
```

Each experiment writes its CSVs to its own `results/` and figures to its own `plots/`;
see the README inside each experiment folder. The data pipeline is shared in `taxapi/`.

## Headline results

- **Benchmark:** 31,140 leakage-safe rows; relaxing the date window to +30 days adds
  +35% rows at the same accuracy (week 4 / relaxed).
- **Coverage:** `source_1 + WOZ` already reaches near-best accuracy across most of the
  housing stock (week 4 / partial sources).
