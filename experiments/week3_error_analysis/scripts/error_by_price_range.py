# Model error broken down by price range.
# in : data/benchmark/benchmark_dataset_with_woz_and_source4.csv
# out: results/error_analysis_by_price_range.csv,
#      plots/02_error_by_price_range_mape.png, 03_..._bias.png
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

from taxapi.core import metrics, modeling, splitting

EXP = Path(__file__).resolve().parents[1]
OUTPUT_FILE = EXP / "results" / "error_analysis_by_price_range.csv"
PLOT_DIR = EXP / "plots"

PRICE_BINS = [0, 250000, 400000, 600000, 1000000, np.inf]
PRICE_LABELS = ["<250k", "250k-400k", "400k-600k", "600k-1M", ">1M"]


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

    print("\nTraining final Random Forest model...")
    model = modeling.build_model(numeric_features, categorical_features)
    model.fit(X_train, y_train)
    print("Training completed.")

    test_results = X_test.copy()
    test_results["gt_value"] = y_test
    test_results["prediction"] = model.predict(X_test)
    metrics.add_error_columns(test_results)

    test_results["price_bin"] = pd.cut(
        test_results["gt_value"], bins=PRICE_BINS, labels=PRICE_LABELS
    )

    summary = metrics.error_summary(test_results, "price_bin")
    print("\nERROR ANALYSIS BY PRICE RANGE")
    print(summary)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.bar(summary["price_bin"].astype(str), summary["MAPE"])
    plt.xlabel("Ground Truth Price Range")
    plt.ylabel("MAPE (%)")
    plt.title("Model Error by Property Price Range")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_error_by_price_range_mape.png", dpi=120)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(summary["price_bin"].astype(str), summary["Bias"])
    plt.axhline(0, linestyle="--")
    plt.xlabel("Ground Truth Price Range")
    plt.ylabel("Bias (%)")
    plt.title("Model Bias by Property Price Range")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_error_by_price_range_bias.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
