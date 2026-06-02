# Does adding source 4 help? Compares the model with and without it, on the same rows.
# in : data/benchmark/benchmark_dataset_with_woz_and_source4.csv
# out: results/source4_benchmark_results.csv
import numpy as np
import pandas as pd

from pathlib import Path

from taxapi.core import metrics, modeling, splitting

EXP = Path(__file__).resolve().parents[1]
OUTPUT_FILE = EXP / "results" / "source4_benchmark_results.csv"

NUM_NO4 = (
    modeling.SOURCE_FEATURES_NO_4 + modeling.NUMERIC_PROPERTY_FEATURES + modeling.WOZ_FEATURES
)
NUM_4 = modeling.SOURCE_FEATURES + modeling.NUMERIC_PROPERTY_FEATURES + modeling.WOZ_FEATURES
CAT = modeling.CATEGORICAL_FEATURES


def main():
    df = modeling.load_benchmark()
    df = df[df["source_4"].notna()].copy()
    print("Rows with source 4:", df.shape)
    y = df["gt_value"]

    rows, deltas = [], []
    for fold, (tr, te) in enumerate(splitting.grouped_kfold(df)):
        y_test = y.loc[te]

        m_wo = modeling.build_model(NUM_NO4, CAT).fit(df.loc[tr, NUM_NO4 + CAT], y.loc[tr])
        m_w = modeling.build_model(NUM_4, CAT).fit(df.loc[tr, NUM_4 + CAT], y.loc[tr])
        e_wo = metrics.evaluate(y_test, pd.Series(m_wo.predict(df.loc[te, NUM_NO4 + CAT]), index=te))
        e_w = metrics.evaluate(y_test, pd.Series(m_w.predict(df.loc[te, NUM_4 + CAT]), index=te))
        rows.append({"method": "rf_without_source4", **e_wo})
        rows.append({"method": "rf_with_source4", **e_w})
        deltas.append(e_wo["MAPE"] - e_w["MAPE"])

        # baselines on the same fold test rows
        baselines = {
            "woz_value": df.loc[te, "woz_value"],
            "source_1": df.loc[te, "source_1"],
            "source_4": df.loc[te, "source_4"],
            "mean(1,4)": df.loc[te, ["source_1", "source_4"]].mean(axis=1),
        }
        for name, pred in baselines.items():
            rows.append({"method": name, **metrics.evaluate(y_test, pred)})

    summary = (
        pd.DataFrame(rows)
        .groupby("method")
        .agg(
            n=("rows", "mean"),
            MAPE_mean=("MAPE", "mean"),
            MAPE_std=("MAPE", "std"),
            MdAPE=("MdAPE", "mean"),
            within_10pct=("within_10pct", "mean"),
            Bias=("Bias", "mean"),
        )
        .reset_index()
        .sort_values("MAPE_mean")
    )
    print("\nSOURCE 4 BENCHMARK (GroupKFold, mean over folds)")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    delta = np.array(deltas)
    print(
        f"\nsource_4 improvement (MAPE without - with): {delta.mean():+.2f} +- {delta.std():.2f} pp "
        f"({int((delta > 0).sum())}/{len(delta)} folds positive)"
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
