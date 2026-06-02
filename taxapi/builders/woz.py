# Attach the nearest leakage-safe WOZ valuation to the clean benchmark, scanning the
# large WOZ history file in chunks. window_days=0 = strict (gold); 30 = relaxed.
# in : data/benchmark[/relaxed]/benchmark_dataset_clean.csv, data/raw/woz_values.csv
# out: data/benchmark[/relaxed]/benchmark_dataset_with_woz.csv
import pandas as pd
from tqdm import tqdm

from taxapi.core import matching, paths

WOZ_FILE = paths.RAW_DIR / "woz_values.csv"
CHUNK_SIZE = 500_000
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
    benchmark_file = bdir / "benchmark_dataset_clean.csv"
    output_file = bdir / "benchmark_dataset_with_woz.csv"

    print("Loading benchmark dataset...")
    benchmark = pd.read_csv(benchmark_file, low_memory=False)
    benchmark["gt_date"] = pd.to_datetime(benchmark["gt_date"], errors="coerce")
    benchmark["property_key"] = make_property_key(benchmark)
    benchmark = benchmark.reset_index(drop=True)
    benchmark["event_id"] = benchmark.index
    benchmark_keys = set(benchmark["property_key"].unique())
    print(f"Benchmark rows: {len(benchmark)}  (window +{window}d)")

    matched_chunks = []
    reader = pd.read_csv(WOZ_FILE, chunksize=CHUNK_SIZE, low_memory=False)
    for chunk in tqdm(reader, desc="WOZ chunks", unit="chunk"):
        chunk["property_key"] = make_property_key(chunk)
        keep = chunk[chunk["property_key"].isin(benchmark_keys)].copy()
        if len(keep):
            matched_chunks.append(keep)

    if not matched_chunks:
        raise ValueError("No matching WOZ rows found for benchmark properties.")
    woz = pd.concat(matched_chunks, ignore_index=True)
    woz["woz_date"] = pd.to_datetime(woz["reference_date"], errors="coerce")
    woz = woz.rename(columns={"value": "woz_value"})[
        ["property_key", "woz_date", "woz_value"]
    ].dropna(subset=["woz_date", "woz_value"])
    print("Matched WOZ rows:", len(woz))

    events = benchmark[["event_id", "property_key", "gt_date"]]
    matched = matching.attach_nearest(
        events, woz, "woz_value", "woz_date", window_days=window, diff_col="woz_days_diff"
    )
    out = benchmark.merge(matched, on="event_id", how="left").drop(columns=["event_id"])
    print("Rows with WOZ:", out["woz_value"].notna().sum(), "/", len(out))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
