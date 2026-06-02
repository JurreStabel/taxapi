# How well can we value a property when only some sources are available?
# Trains a model on each source combination and measures its error. The 'view'
# column has two modes: each combination scored on the properties that have it,
# and all combinations scored on one shared set so they compare fairly.
# in : data/benchmark/benchmark_dataset_with_woz_and_source4.csv
# out: results/partial_sources.csv, partial_sources.png
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

from taxapi.core import metrics, modeling, splitting

EXP = Path(__file__).resolve().parents[1]
OUTPUT_FILE = EXP / "results" / "partial_sources.csv"
PLOT_DIR = EXP / "plots"

WOZ = ["woz_value", "woz_days_diff"]
SCENARIOS = {
    "WOZ only": WOZ,
    "source_1": ["source_1"],
    "source_1 + WOZ": ["source_1"] + WOZ,
    "source_1 + source_2": ["source_1", "source_2"],
    "sources 1+2+3": ["source_1", "source_2", "source_3"],
    "sources 1+2+3 + WOZ": ["source_1", "source_2", "source_3"] + WOZ,
    "all + source_4": ["source_1", "source_2", "source_3", "source_4"] + WOZ + ["source_4_days_diff"],
}
ALL_PRESENT = ["source_1", "source_2", "source_3", "source_4", "woz_value"]


def presence_cols(cols):
    return [c for c in cols if not c.endswith("days_diff")]


def kfold_mape(df, numeric, categorical):
    X, y = df[numeric + categorical], df["gt_value"]
    scores = []
    for train_idx, test_idx in splitting.grouped_kfold(df):
        model = modeling.build_model(numeric, categorical).fit(X.loc[train_idx], y.loc[train_idx])
        pred = pd.Series(model.predict(X.loc[test_idx]), index=test_idx)
        scores.append(metrics.evaluate(y.loc[test_idx], pred)["MAPE"])
    return float(np.mean(scores)), float(np.std(scores))


def run_view(df, view):
    cat = modeling.CATEGORICAL_FEATURES
    common = df[df[ALL_PRESENT].notna().all(axis=1)]
    rows = []
    for name, cols in SCENARIOS.items():
        sub = common if view == "same_rows" else df[df[presence_cols(cols)].notna().all(axis=1)]
        mean, std = kfold_mape(sub, cols + modeling.NUMERIC_PROPERTY_FEATURES, cat)
        rows.append(
            {
                "view": view,
                "scenario": name,
                "sources": len(presence_cols(cols)),
                "n": len(sub),
                "MAPE_mean": mean,
                "MAPE_std": std,
            }
        )
    return rows


def plot_same_rows(table):
    ab = table[table["view"] == "same_rows"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ab))
    ax.bar(x, ab["MAPE_mean"], yerr=ab["MAPE_std"], capsize=4, color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(ab["scenario"], rotation=30, ha="right")
    ax.set_ylabel("MAPE (GroupKFold, %)")
    ax.set_title(
        f"Marginal value of each source (same {int(ab['n'].iloc[0])} rows, all sources present)"
    )
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "partial_sources.png", dpi=120)
    plt.close(fig)


def main():
    df = modeling.load_benchmark()
    table = pd.DataFrame(run_view(df, "by_availability") + run_view(df, "same_rows"))

    for view in ("by_availability", "same_rows"):
        print(f"\n=== {view} ===")
        print(
            table[table["view"] == view][["scenario", "sources", "n", "MAPE_mean", "MAPE_std"]]
            .to_string(index=False, float_format=lambda v: f"{v:.2f}")
        )

    plot_same_rows(table)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE} + partial_sources.png")


if __name__ == "__main__":
    main()
