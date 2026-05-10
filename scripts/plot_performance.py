from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "full_dataset_model_values_assignment"
OUT_DIR = REPO / "plots"
OUT_DIR.mkdir(exist_ok=True)

ADDR_KEYS = ["postal_code", "house_number", "letter"]
SOURCES = (1, 2, 3)


def load_keyed(path, extra_cols):
    return (
        pl.scan_csv(path)
        .with_columns(pl.col("letter").fill_null(""))
        .select([*ADDR_KEYS, *extra_cols])
    )


def build_pairs():
    mv = load_keyed(
        DATA_DIR / "model_values_data.csv",
        ["value", "mv_source", "province", "object_type"],
    ).rename({"value": "model_value"})

    gt = (
        load_keyed(DATA_DIR / "ground_truth_data.csv", ["value"])
        .group_by(ADDR_KEYS)
        .agg(pl.col("value").median().alias("gt_value"))
    )

    return (
        mv.join(gt, on=ADDR_KEYS, how="inner")
        .with_columns(
            ((pl.col("model_value") - pl.col("gt_value")) / pl.col("gt_value"))
            .alias("rel_err")
        )
        .collect()
    )


def print_metrics(pairs):
    metrics = (
        pairs.group_by("mv_source")
        .agg(
            pl.col("rel_err").abs().mean().alias("MAPE"),
            pl.col("rel_err").abs().median().alias("MdAPE"),
            (pl.col("rel_err").abs() < 0.10).mean().alias("within_10pct"),
            pl.len().alias("n"),
        )
        .sort("mv_source")
    )
    print(metrics)


def plot_scatter(pairs):
    cap = 1_500_000
    fig, ax = plt.subplots(figsize=(7, 7))
    for src in SOURCES:
        sub = pairs.filter(pl.col("mv_source") == src)
        sub = sub.sample(n=min(5000, sub.height), seed=0)
        ax.scatter(sub["gt_value"], sub["model_value"],
                   s=4, alpha=0.3, label=f"source {src}")
    ax.plot([0, cap], [0, cap], "k--", lw=1)
    ax.set_xlim(0, cap); ax.set_ylim(0, cap)
    ax.set_xlabel("ground truth (€)")
    ax.set_ylabel("model value (€)")
    ax.set_title("Model vs ground truth")
    ax.legend()
    out = OUT_DIR / "01_pred_vs_gt.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_error_hist(pairs):
    bins = np.linspace(-0.5, 0.5, 61)
    fig, ax = plt.subplots(figsize=(8, 5))
    for src in SOURCES:
        e = pairs.filter(pl.col("mv_source") == src)["rel_err"].to_numpy()
        ax.hist(np.clip(e, -0.5, 0.5), bins=bins, alpha=0.45, label=f"source {src}")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("(model − gt) / gt   (clipped to ±50%)")
    ax.set_ylabel("count")
    ax.set_title("Relative error per source")
    ax.legend()
    out = OUT_DIR / "02_error_distribution.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_by_value_bin(pairs):
    edges = [250_000, 400_000, 600_000, 1_000_000]
    labels = ["<250k", "250–400k", "400–600k", "600k–1M", ">1M"]

    summary = (
        pairs.with_columns(
            pl.col("gt_value").cut(edges, labels=labels).alias("price_bin")
        )
        .group_by(["price_bin", "mv_source"])
        .agg(
            pl.col("rel_err").median().alias("median_signed"),
            pl.col("rel_err").abs().median().alias("median_abs"),
            pl.len().alias("n"),
        )
        .sort(["mv_source", "price_bin"])
    )
    print("\nperformance by value bin:")
    print(summary)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    x = np.arange(len(labels))
    for src in SOURCES:
        rows = {
            r["price_bin"]: r
            for r in summary.filter(pl.col("mv_source") == src).iter_rows(named=True)
        }
        signed = [rows[lbl]["median_signed"] if lbl in rows else None for lbl in labels]
        absolute = [rows[lbl]["median_abs"] if lbl in rows else None for lbl in labels]
        axes[0].plot(x, signed, "o-", label=f"source {src}")
        axes[1].plot(x, absolute, "o-", label=f"source {src}")

    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_title("Bias  —  median signed relative error")
    axes[0].set_ylabel("(model − gt) / gt")
    axes[1].set_title("Accuracy  —  median |relative error|")
    axes[1].set_ylabel("|model − gt| / gt")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("ground truth value bin")
        ax.legend()

    out = OUT_DIR / "03_error_by_value_bin.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_by_province(pairs):
    summary = (
        pairs.group_by(["province", "mv_source"])
        .agg(
            pl.col("rel_err").median().alias("median_signed"),
            pl.col("rel_err").abs().median().alias("median_abs"),
            pl.len().alias("n"),
        )
        .sort(["mv_source", "province"])
    )
    print("\nperformance by province:")
    print(summary)

    provinces = sorted(summary["province"].unique().to_list())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    x = np.arange(len(provinces))
    for src in SOURCES:
        rows = {
            r["province"]: r
            for r in summary.filter(pl.col("mv_source") == src).iter_rows(named=True)
        }
        signed = [rows[p]["median_signed"] if p in rows else None for p in provinces]
        absolute = [rows[p]["median_abs"] if p in rows else None for p in provinces]
        axes[0].plot(x, signed, "o-", label=f"source {src}")
        axes[1].plot(x, absolute, "o-", label=f"source {src}")

    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_title("Bias  —  median signed relative error")
    axes[0].set_ylabel("(model − gt) / gt")
    axes[1].set_title("Accuracy  —  median |relative error|")
    axes[1].set_ylabel("|model − gt| / gt")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(provinces, rotation=45, ha="right")
        ax.set_xlabel("province")
        ax.legend()

    out = OUT_DIR / "04_error_by_province.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_by_object_type(pairs):
    types = (
        pairs.drop_nulls("object_type")
        .group_by("object_type")
        .agg(pl.len().alias("n_total"))
        .sort("n_total", descending=True)["object_type"]
        .to_list()
    )

    summary = (
        pairs.drop_nulls("object_type")
        .group_by(["object_type", "mv_source"])
        .agg(
            pl.col("rel_err").median().alias("median_signed"),
            pl.col("rel_err").abs().median().alias("median_abs"),
            pl.len().alias("n"),
        )
        .sort(["mv_source", "object_type"])
    )
    print("\nperformance by object_type:")
    print(summary)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    x = np.arange(len(types))
    for src in SOURCES:
        rows = {
            r["object_type"]: r
            for r in summary.filter(pl.col("mv_source") == src).iter_rows(named=True)
        }
        signed = [rows[t]["median_signed"] if t in rows else None for t in types]
        absolute = [rows[t]["median_abs"] if t in rows else None for t in types]
        axes[0].plot(x, signed, "o-", label=f"source {src}")
        axes[1].plot(x, absolute, "o-", label=f"source {src}")

    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_title("Bias  —  median signed relative error")
    axes[0].set_ylabel("(model − gt) / gt")
    axes[1].set_title("Accuracy  —  median |relative error|")
    axes[1].set_ylabel("|model − gt| / gt")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=45, ha="right")
        ax.set_xlabel("object type (sorted by n)")
        ax.legend()

    out = OUT_DIR / "05_error_by_object_type.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_ensemble_comparison(pairs):
    wide = (
        pairs.pivot(
            on="mv_source",
            index=[*ADDR_KEYS, "gt_value"],
            values="model_value",
            aggregate_function="median",
        )
        .rename({"1": "m1", "2": "m2", "3": "m3"})
        .drop_nulls(["m1", "m2", "m3"])
    )

    combos = {
        "1": pl.col("m1"),
        "2": pl.col("m2"),
        "3": pl.col("m3"),
        "mean(1,2)": (pl.col("m1") + pl.col("m2")) / 2,
        "mean(1,3)": (pl.col("m1") + pl.col("m3")) / 2,
        "mean(2,3)": (pl.col("m2") + pl.col("m3")) / 2,
        "mean(1,2,3)": (pl.col("m1") + pl.col("m2") + pl.col("m3")) / 3,
        "median(1,2,3)": pl.concat_list(["m1", "m2", "m3"]).list.median(),
    }

    def rel_err(pred_expr):
        return (pred_expr - pl.col("gt_value")) / pl.col("gt_value")

    errs = wide.select(
        [rel_err(expr).alias(name) for name, expr in combos.items()]
    )

    rows = []
    for name in combos:
        s = errs[name]
        rows.append({
            "combo": name,
            "median_signed": float(s.median()),
            "median_abs": float(s.abs().median()),
        })
    summary = pl.DataFrame(rows)
    print(f"\nensemble comparison (n={len(wide):,} addresses with all 3 models):")
    print(summary)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = summary["combo"].to_list()
    x = np.arange(len(names))
    colors = ["#1f77b4" if n in ("1", "2", "3") else "#2ca02c" for n in names]

    axes[0].bar(x, summary["median_signed"].to_numpy(), color=colors)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_title("Bias  —  median signed relative error")
    axes[0].set_ylabel("(pred − gt) / gt")

    axes[1].bar(x, summary["median_abs"].to_numpy(), color=colors)
    axes[1].set_title("Accuracy  —  median |relative error|")
    axes[1].set_ylabel("|pred − gt| / gt")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")

    fig.suptitle("blue = single source, green = ensemble")
    out = OUT_DIR / "06_ensemble_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    pairs = build_pairs()
    print(f"paired rows: {len(pairs):,}\n")
    print_metrics(pairs)
    plot_scatter(pairs)
    plot_error_hist(pairs)
    plot_by_value_bin(pairs)
    plot_by_province(pairs)
    plot_by_object_type(pairs)
    plot_ensemble_comparison(pairs)


if __name__ == "__main__":
    main()
