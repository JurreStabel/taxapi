# Model error broken down by property type.
# in : data/benchmark/benchmark_dataset_with_woz_and_source4.csv
# out: results/error_analysis_by_object_type.csv,
#      plots/04_error_by_object_type_mape.png, 05_..._bias.png
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

from taxapi.core import metrics, modeling, splitting

EXP = Path(__file__).resolve().parents[1]
OUTPUT_FILE = EXP / "results" / "error_analysis_by_object_type.csv"
PLOT_DIR = EXP / "plots"
MIN_ROWS = 30


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
    test_object_types = X_test["object_type"].copy()

    print("\nTraining final Random Forest model...")
    model = modeling.build_model(numeric_features, categorical_features)
    model.fit(X_train, y_train)
    print("Training completed.")

    results = pd.DataFrame(
        {
            "object_type": test_object_types,
            "gt_value": y_test,
            "prediction": model.predict(X_test),
        }
    )
    metrics.add_error_columns(results)

    summary = metrics.error_summary(results, "object_type")
    summary = summary[summary["rows"] >= MIN_ROWS].sort_values("MAPE")
    print("\nERROR ANALYSIS BY OBJECT TYPE")
    print(summary)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.bar(summary["object_type"], summary["MAPE"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("MAPE (%)")
    plt.xlabel("Object Type")
    plt.title("Model Error by Object Type")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_error_by_object_type_mape.png", dpi=120)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(summary["object_type"], summary["Bias"])
    plt.axhline(0, linestyle="--")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Bias (%)")
    plt.xlabel("Object Type")
    plt.title("Model Bias by Object Type")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_error_by_object_type_bias.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
