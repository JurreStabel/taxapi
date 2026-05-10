# taxapi

Exploration of a Dutch property valuation dataset (WOZ + model values + ground truth).

## Data

The `full_dataset_model_values_assignment/` directory (gitignored) contains:

| File | Rows | Size | Description |
| --- | ---: | ---: | --- |
| `ground_truth_data.csv` | 1,107,924 | 141 MB | Property records with full attributes and a ground-truth `value`. Column `gt_source` identifies the source. |
| `model_values_data.csv` | 117,346 | 14 MB | Model-predicted values for properties. Multiple rows per property (one per `mv_source`). |
| `woz_values.csv` | 97,815,511 | 2.7 GB | Historical official WOZ valuations keyed by `(postal_code, house_number, letter)` and `reference_date`. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Inspect the data

```bash
python scripts/inspect_data.py
```
