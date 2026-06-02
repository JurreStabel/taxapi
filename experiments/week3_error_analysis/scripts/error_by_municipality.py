# Model error broken down by municipality.
# in : data/benchmark/benchmark_dataset_with_woz_and_source4.csv
# out: results/error_analysis_by_municipality.csv,
#      plots/06_municipality_best.png, 07_..._worst.png, 08_..._bias.png
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

from taxapi.core import metrics, modeling, splitting

EXP = Path(__file__).resolve().parents[1]
OUTPUT_FILE = EXP / "results" / "error_analysis_by_municipality.csv"
PLOT_DIR = EXP / "plots"
MIN_ROWS = 20


def main():
    df = modeling.load_benchmark()
    print("Original dataset shape:", df.shape)

    df = df[df[modeling.SOURCE_FEATURES_NO_4].notna().all(axis=1)].copy()
    print("rows with sources 1-3:", df.shape)

    y = df["gt_value"]
    numeric_features = (
        modeling.SOURCE_FEATURES
        + modeling.NUMERIC_PROPERTY_FEATURES
        + modeling.WOZ_FEATURES_WITH_SOURCE4
    )
    categorical_features = modeling.CATEGORICAL_FEATURES
    X = df[numeric_features + categorical_features]

    train_idx, test_idx = splitting.grouped_split(df)
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    test_municipalities = X_test["municipality"].copy()

    print("\nTraining final Random Forest model...")
    model = modeling.build_model(numeric_features, categorical_features)
    model.fit(X_train, y_train)
    print("Training completed.")

    results = pd.DataFrame(
        {
            "municipality": test_municipalities,
            "gt_value": y_test,
            "prediction": model.predict(X_test),
        }
    )
    metrics.add_error_columns(results)

    summary = metrics.error_summary(results, "municipality")
    summary = summary[summary["rows"] >= MIN_ROWS].copy()
    print("\nMUNICIPALITY ERROR ANALYSIS")
    print(summary.sort_values("MAPE"))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    best = summary.sort_values("MAPE").head(15)
    plt.figure(figsize=(12, 6))
    plt.bar(best["municipality"], best["MAPE"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("MAPE (%)")
    plt.xlabel("Municipality")
    plt.title("Best Performing Municipalities")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "06_municipality_best.png", dpi=120)
    plt.close()

    worst = summary.sort_values("MAPE", ascending=False).head(15)
    plt.figure(figsize=(12, 6))
    plt.bar(worst["municipality"], worst["MAPE"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("MAPE (%)")
    plt.xlabel("Municipality")
    plt.title("Worst Performing Municipalities")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "07_municipality_worst.png", dpi=120)
    plt.close()

    bias_sorted = summary.reindex(
        summary["Bias"].abs().sort_values(ascending=False).index
    ).head(15)
    plt.figure(figsize=(12, 6))
    plt.bar(bias_sorted["municipality"], bias_sorted["Bias"])
    plt.axhline(0, linestyle="--")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Bias (%)")
    plt.xlabel("Municipality")
    plt.title("Largest Municipality Biases")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "08_municipality_bias.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
