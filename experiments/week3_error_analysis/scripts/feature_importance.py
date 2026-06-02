# Which features the model relies on most.
# in : data/benchmark/benchmark_dataset_with_woz_and_source4.csv
# out: results/source4_feature_importance.csv, plots/01_feature_importance.png
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

from taxapi.core import modeling, splitting

EXP = Path(__file__).resolve().parents[1]
OUTPUT_FILE = EXP / "results" / "source4_feature_importance.csv"
PLOT_DIR = EXP / "plots"


def main():
    df = modeling.load_benchmark()
    print("Original dataset shape:", df.shape)

    df = df[df["source_4"].notna()].copy()
    print("Rows with source 4:", df.shape)

    y = df["gt_value"]
    numeric_features = (
        modeling.SOURCE_FEATURES
        + modeling.NUMERIC_PROPERTY_FEATURES
        + modeling.WOZ_FEATURES
    )
    categorical_features = modeling.CATEGORICAL_FEATURES
    X = df[numeric_features + categorical_features]

    train_idx, _ = splitting.grouped_split(df)
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]

    print("\nTraining Random Forest with source 4...")
    rf_pipeline = modeling.build_model(numeric_features, categorical_features)
    rf_pipeline.fit(X_train, y_train)
    print("Training completed.")

    # expand one-hot names so importances line up with the encoded columns
    ohe = (
        rf_pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
    )
    feature_names = numeric_features + list(
        ohe.get_feature_names_out(categorical_features)
    )

    rf_model = rf_pipeline.named_steps["model"]
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": rf_model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    print("\nTOP 25 FEATURES")
    print(importance_df.head(25))

    top20 = importance_df.head(20)
    plt.figure(figsize=(12, 8))
    plt.barh(top20["Feature"][::-1], top20["Importance"][::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 20 Feature Importances WITH Source 4")
    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_DIR / "01_feature_importance.png", dpi=120)
    plt.close()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
