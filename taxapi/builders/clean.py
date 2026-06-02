# Build the clean benchmark from raw ground truth + model values (leakage-safe).
# For each ground-truth valuation event (gt_source 1/2/3, one row per property+date)
# attach the closest model value per source within the leakage window, then keep
# events with at least one source. window_days=0 = strict (gold); 30 = relaxed.
# in : data/raw/ground_truth_data.csv, data/raw/model_values_data.csv
# out: data/benchmark[/relaxed]/benchmark_dataset_clean.csv
import pandas as pd

from taxapi.core import matching, paths

GT_FILE = paths.RAW_DIR / "ground_truth_data.csv"
MV_FILE = paths.RAW_DIR / "model_values_data.csv"

GT_SOURCES = [1, 2, 3]
MODEL_SOURCES = [1, 2, 3]
RELAXED_WINDOW_DAYS = 30

COLUMNS = [
    "province", "municipality", "city", "street", "postal_code", "house_number",
    "letter", "latitude", "longitude", "gt_value", "gt_date", "object_type",
    "plot_area", "floor_area", "build_year", "volume", "energy_label", "gt_source",
    "property_key", "source_1", "source_2", "source_3",
]


def make_property_key(df):
    letter = df["letter"].fillna("").astype(str).str.strip()
    return (
        df["postal_code"].astype(str)
        + "_"
        + df["house_number"].astype(str)
        + "_"
        + letter
    )


def main(relaxed=False):
    window = RELAXED_WINDOW_DAYS if relaxed else 0
    output_file = paths.benchmark_dir(relaxed) / "benchmark_dataset_clean.csv"

    print("Loading ground truth...")
    gt = pd.read_csv(GT_FILE, low_memory=False)
    gt = gt[gt["gt_source"].isin(GT_SOURCES)].copy()
    gt["gt_date"] = pd.to_datetime(gt["valuation_date"], errors="coerce")
    gt["property_key"] = make_property_key(gt)
    gt = gt.rename(columns={"value": "gt_value"}).drop(columns=["valuation_date"])
    gt = gt.dropna(subset=["gt_date"])
    gt = (
        gt.drop_duplicates(subset=["property_key", "gt_date"], keep="first")
        .reset_index(drop=True)
    )
    gt["event_id"] = gt.index
    print(f"Ground-truth events (gt_source 1/2/3): {len(gt)}  (window +{window}d)")

    print("Loading model values...")
    mv = pd.read_csv(MV_FILE, low_memory=False)
    mv["mv_date"] = pd.to_datetime(mv["valuation_date"], errors="coerce")
    mv["property_key"] = make_property_key(mv)

    events = gt[["event_id", "property_key", "gt_date"]]
    benchmark = gt
    for s in MODEL_SOURCES:
        cand = (
            mv[mv["mv_source"] == s][["property_key", "mv_date", "value"]]
            .rename(columns={"value": f"source_{s}"})
            .dropna(subset=["mv_date"])
        )
        matched = matching.attach_nearest(
            events, cand, f"source_{s}", "mv_date", window_days=window
        )
        benchmark = benchmark.merge(
            matched[["event_id", f"source_{s}"]], on="event_id", how="left"
        )

    source_cols = [f"source_{s}" for s in MODEL_SOURCES]
    before = len(benchmark)
    benchmark = benchmark[benchmark[source_cols].notna().any(axis=1)].copy()
    print(f"Kept events with >=1 source: {len(benchmark)} (dropped {before - len(benchmark)})")
    for s in source_cols:
        print(f"  {s} coverage:", benchmark[s].notna().sum())

    benchmark = benchmark[COLUMNS]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}  ({len(benchmark)} rows)")


if __name__ == "__main__":
    main()
