# Attach the nearest leakage-safe source-4 valuation to the WOZ benchmark.
# window_days=0 = strict (gold); 30 = relaxed.
# in : data/benchmark[/relaxed]/benchmark_dataset_with_woz.csv, data/raw/model_values_source_4_data.csv
# out: data/benchmark[/relaxed]/benchmark_dataset_with_woz_and_source4.csv
import pandas as pd

from taxapi.core import matching, paths

SOURCE4_FILE = paths.RAW_DIR / "model_values_source_4_data.csv"
RELAXED_WINDOW_DAYS = 30


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
    bdir = paths.benchmark_dir(relaxed)
    benchmark_file = bdir / "benchmark_dataset_with_woz.csv"
    output_file = bdir / "benchmark_dataset_with_woz_and_source4.csv"

    print("Loading benchmark dataset...")
    benchmark = pd.read_csv(benchmark_file, low_memory=False)
    benchmark["gt_date"] = pd.to_datetime(benchmark["gt_date"], errors="coerce")
    benchmark["property_key"] = make_property_key(benchmark)
    benchmark = benchmark.reset_index(drop=True)
    benchmark["event_id"] = benchmark.index
    print(f"Benchmark rows: {len(benchmark)}  (window +{window}d)")

    source4 = pd.read_csv(SOURCE4_FILE, low_memory=False)
    source4["source_4_date"] = pd.to_datetime(source4["valuation_date"], errors="coerce")
    source4["property_key"] = make_property_key(source4)
    source4 = source4.rename(columns={"value": "source_4"})[
        ["property_key", "source_4_date", "source_4"]
    ].dropna(subset=["source_4_date"])

    events = benchmark[["event_id", "property_key", "gt_date"]]
    matched = matching.attach_nearest(
        events, source4, "source_4", "source_4_date",
        window_days=window, diff_col="source_4_days_diff",
    )
    final_df = benchmark.merge(matched, on="event_id", how="left").drop(columns=["event_id"])
    print("Rows with source 4:", final_df["source_4"].notna().sum(), "/", len(final_df))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
