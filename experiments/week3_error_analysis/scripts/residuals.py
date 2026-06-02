# Overall error distribution of the model on the test set.
# in : data/benchmark/benchmark_dataset_with_woz_and_source4.csv
# out: results/residual_analysis_results.csv, plots/09_..12_residuals_*.png
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

from taxapi.core import modeling, splitting

EXP = Path(__file__).resolve().parents[1]
OUTPUT_FILE = EXP / "results" / "residual_analysis_results.csv"
PLOT_DIR = EXP / "plots"


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

    results = pd.DataFrame({"gt_value": y_test, "prediction": model.predict(X_test)})
    results["error"] = results["prediction"] - results["gt_value"]
    results["absolute_error"] = results["error"].abs()
    results["relative_error"] = results["error"] / results["gt_value"]
    results["absolute_relative_error"] = results["relative_error"].abs()
    results["relative_error_percent"] = results["relative_error"] * 100
    results["absolute_relative_error_percent"] = results["absolute_relative_error"] * 100

    print("\nOVERALL ERROR SUMMARY")
    print("Rows:", len(results))
    print("Mean error (€):", results["error"].mean())
    print("Median error (€):", results["error"].median())
    print("Mean absolute error (€):", results["absolute_error"].mean())
    print("Median absolute error (€):", results["absolute_error"].median())
    print("MAPE (%):", results["absolute_relative_error_percent"].mean())
    print(
        "Median absolute percentage error (%):",
        results["absolute_relative_error_percent"].median(),
    )
    print("Bias (%):", results["relative_error_percent"].mean())

    print("\nERROR PERCENTILES")
    print(
        results[["absolute_error", "absolute_relative_error_percent"]].quantile(
            [0.5, 0.75, 0.9, 0.95, 0.99]
        )
    )

    print("\nEXTREME ERROR COUNTS")
    for threshold in (10, 20, 30):
        n = (results["absolute_relative_error_percent"] > threshold).sum()
        print(f"Errors > {threshold}%:", n)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.hist(results["relative_error_percent"], bins=60)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Relative Error (%)")
    plt.ylabel("Count")
    plt.title("Distribution of Relative Errors")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "09_residuals_relative_error_hist.png", dpi=120)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(results["absolute_relative_error_percent"], bins=60)
    plt.xlabel("Absolute Percentage Error (%)")
    plt.ylabel("Count")
    plt.title("Distribution of Absolute Percentage Errors")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "10_residuals_abs_pct_error_hist.png", dpi=120)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(results["gt_value"], results["prediction"], alpha=0.4)
    min_value = min(results["gt_value"].min(), results["prediction"].min())
    max_value = max(results["gt_value"].max(), results["prediction"].max())
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.xlabel("Ground Truth Value (€)")
    plt.ylabel("Predicted Value (€)")
    plt.title("Prediction vs Ground Truth")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "11_residuals_pred_vs_gt.png", dpi=120)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.scatter(results["gt_value"], results["relative_error_percent"], alpha=0.4)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Ground Truth Value (€)")
    plt.ylabel("Relative Error (%)")
    plt.title("Relative Error vs Property Value")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "12_residuals_error_vs_value.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
