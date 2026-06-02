# Strict vs relaxed (+30 day) benchmark: how many more rows we get and whether
# accuracy holds. Run after building both benchmark versions.
# out: results/relaxed_vs_gold.csv, plots/{coverage,mape_stability}.png
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from taxapi.core import metrics, modeling, paths, splitting

EXP = Path(__file__).resolve().parents[1]
RESULTS, PLOTS = EXP / "results", EXP / "plots"
BLUE, ORANGE = "#1f77b4", "#ff7f0e"

NUMERIC = (
    modeling.SOURCE_FEATURES
    + modeling.NUMERIC_PROPERTY_FEATURES
    + modeling.WOZ_FEATURES_WITH_SOURCE4
)
CAT = modeling.CATEGORICAL_FEATURES
SOURCES = ["source_1", "source_2", "source_3", "source_4", "woz_value"]


def load(relaxed):
    path = paths.benchmark_dir(relaxed) / "benchmark_dataset_with_woz_and_source4.csv"
    return pd.read_csv(path, low_memory=False)


def track_a(df):
    return df[df[modeling.SOURCE_FEATURES_NO_4].notna().all(axis=1)].copy()


def rf_kfold_mape(ta):
    X, y = ta[NUMERIC + CAT], ta["gt_value"]
    scores = []
    for train_idx, test_idx in splitting.grouped_kfold(ta):
        model = modeling.build_model(NUMERIC, CAT).fit(X.loc[train_idx], y.loc[train_idx])
        pred = pd.Series(model.predict(X.loc[test_idx]), index=test_idx)
        scores.append(metrics.evaluate(y.loc[test_idx], pred)["MAPE"])
    return float(np.mean(scores)), float(np.std(scores))


def make_plots(table):
    PLOTS.mkdir(parents=True, exist_ok=True)
    g = table.set_index("dataset")

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    w = 0.38
    gold = [g.loc["gold", "rows"], g.loc["gold", "track_a_rows"]]
    relx = [g.loc["relaxed", "rows"], g.loc["relaxed", "track_a_rows"]]
    ax.bar(x - w / 2, gold, w, label="gold", color=BLUE)
    ax.bar(x + w / 2, relx, w, label="relaxed", color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(["Total rows", "rows with sources 1-3"])
    ax.set_ylabel("rows")
    ax.set_title("Coverage gold vs relaxed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "coverage.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    means = [g.loc["gold", "RF_MAPE_mean"], g.loc["relaxed", "RF_MAPE_mean"]]
    stds = [g.loc["gold", "RF_MAPE_std"], g.loc["relaxed", "RF_MAPE_std"]]
    ax.bar([0, 1], means, yerr=stds, capsize=10, width=0.5, color=[BLUE, ORANGE])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["gold", "relaxed"])
    ax.set_ylim(0, max(means) * 1.4)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("RF MAPE gold vs relaxed")
    fig.tight_layout()
    fig.savefig(PLOTS / "mape_stability.png", dpi=120)
    plt.close(fig)


def main():
    rows = []
    for name, relaxed in [("gold", False), ("relaxed", True)]:
        df = load(relaxed)
        ta = track_a(df)
        mape_mean, mape_std = rf_kfold_mape(ta)
        row = {
            "dataset": name,
            "rows": len(df),
            "track_a_rows": len(ta),
            "RF_MAPE_mean": mape_mean,
            "RF_MAPE_std": mape_std,
        }
        for s in SOURCES:
            row[f"{s}_cov%"] = df[s].notna().mean() * 100
        rows.append(row)

    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "relaxed_vs_gold.csv", index=False)
    make_plots(table)
    print(f"\nSaved: {RESULTS / 'relaxed_vs_gold.csv'} + plots/")


if __name__ == "__main__":
    main()
