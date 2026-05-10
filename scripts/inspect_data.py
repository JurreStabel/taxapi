from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parent.parent / "full_dataset_model_values_assignment"


def header(title):
    print(f"\n=== {title} ===")


def summarize(name, lf):
    header(name)
    schema = lf.collect_schema()
    print(f"columns ({len(schema)}): {list(schema.names())}")
    print(f"dtypes: {dict(schema)}")
    print(f"rows: {lf.select(pl.len()).collect().item():,}")
    print("head:")
    print(lf.head(5).collect())


def value_counts(name, lf, col, top=10):
    header(f"{name} — top {top} values of `{col}`")
    print(
        lf.group_by(col)
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(top)
        .collect()
    )


def numeric_stats(name, lf, col):
    header(f"{name} — `{col}` stats")
    print(lf.select(
        pl.col(col).min().alias("min"),
        pl.col(col).max().alias("max"),
        pl.col(col).mean().alias("mean"),
        pl.col(col).median().alias("median"),
        pl.col(col).null_count().alias("nulls"),
    ).collect())


def main():
    gt = pl.scan_csv(DATA_DIR / "ground_truth_data.csv")
    mv = pl.scan_csv(DATA_DIR / "model_values_data.csv")
    woz = pl.scan_csv(DATA_DIR / "woz_values.csv")

    summarize("ground_truth_data.csv", gt)
    value_counts("ground_truth", gt, "gt_source")
    value_counts("ground_truth", gt, "object_type")
    numeric_stats("ground_truth", gt, "value")

    summarize("model_values_data.csv", mv)
    value_counts("model_values", mv, "mv_source")
    numeric_stats("model_values", mv, "value")

    summarize("woz_values.csv", woz)
    value_counts("woz_values", woz, "reference_date")
    numeric_stats("woz_values", woz, "value")


if __name__ == "__main__":
    main()
