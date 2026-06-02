import polars as pl

from taxapi.core import paths

DATA_DIR = paths.RAW_DIR
ADDR_KEYS = ["postal_code", "house_number", "letter"]

GT_CSV = "ground_truth_data.csv"
MV_CSV = "model_values_data.csv"
WOZ_CSV = "woz_values.csv"


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


def model_source_coverage():
    header("model_values — per-source coverage and overlaps (unique addresses)")
    mv = (
        pl.scan_csv(DATA_DIR / MV_CSV)
        .with_columns(pl.col("letter").fill_null(""))
        .select([*ADDR_KEYS, "mv_source"])
        .unique()
        .group_by(ADDR_KEYS)
        .agg(pl.col("mv_source").sort().alias("sources"))
        .collect()
    )
    total = len(mv)
    print(f"unique addresses with ≥1 model prediction: {total:,}\n")

    print("per-source coverage:")
    for src in (1, 2, 3):
        n = mv.filter(pl.col("sources").list.contains(src)).height
        print(f"  source {src}: {n:>7,}  ({n/total*100:5.1f}%)")

    print("\npairwise overlaps:")
    for a, b in ((1, 2), (1, 3), (2, 3)):
        n = mv.filter(
            pl.col("sources").list.contains(a) & pl.col("sources").list.contains(b)
        ).height
        print(f"  source {a} ∩ {b}: {n:>7,}")

    n_all = mv.filter(
        pl.col("sources").list.contains(1)
        & pl.col("sources").list.contains(2)
        & pl.col("sources").list.contains(3)
    ).height
    print(f"\nall three sources: {n_all:>7,}  ({n_all/total*100:5.1f}%)")

    print("\nexclusive (covered by only one source):")
    for src in (1, 2, 3):
        n = mv.filter(
            (pl.col("sources").list.len() == 1)
            & pl.col("sources").list.contains(src)
        ).height
        print(f"  only source {src}: {n:>7,}")


def gt_vs_mv_overlap():
    header("ground truth vs model values overlap (unique addresses)")
    gt = (
        pl.scan_csv(DATA_DIR / GT_CSV)
        .with_columns(pl.col("letter").fill_null(""))
        .select(ADDR_KEYS)
        .unique()
        .collect()
    )
    mv = (
        pl.scan_csv(DATA_DIR / MV_CSV)
        .with_columns(pl.col("letter").fill_null(""))
        .select(ADDR_KEYS)
        .unique()
        .collect()
    )
    overlap = gt.join(mv, on=ADDR_KEYS, how="inner")
    print(f"  unique gt addresses:        {len(gt):>9,}")
    print(f"  unique mv addresses:        {len(mv):>9,}")
    print(f"  gt ∩ mv:                    {len(overlap):>9,}")
    print(f"  gt only (no mv prediction): {len(gt) - len(overlap):>9,}")
    print(f"  mv only (no gt label):      {len(mv) - len(overlap):>9,}")


def main():
    gt = pl.scan_csv(DATA_DIR / GT_CSV)
    mv = pl.scan_csv(DATA_DIR / MV_CSV)
    woz = pl.scan_csv(DATA_DIR / WOZ_CSV)

    summarize(GT_CSV, gt)
    value_counts("ground_truth", gt, "gt_source")
    value_counts("ground_truth", gt, "object_type")
    numeric_stats("ground_truth", gt, "value")

    summarize(MV_CSV, mv)
    value_counts("model_values", mv, "mv_source")
    numeric_stats("model_values", mv, "value")

    summarize(WOZ_CSV, woz)
    value_counts("woz_values", woz, "reference_date")
    numeric_stats("woz_values", woz, "value")

    model_source_coverage()
    gt_vs_mv_overlap()


if __name__ == "__main__":
    main()
