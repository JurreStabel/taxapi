import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    "benchmark_dataset_with_woz_and_source4.csv",
    low_memory=False
)

print("Original dataset shape:")
print(df.shape)


# =========================================================
# KEEP ONLY ROWS WITH SOURCE 4
# =========================================================

df = df[df["source_4"].notna()].copy()

print("\nRows with source 4:")
print(df.shape)


# =========================================================
# EVALUATION FUNCTION THAT HANDLES NaN
# =========================================================

def evaluate(y_true, y_pred):
    temp = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    }).dropna()

    y_true_clean = temp["y_true"]
    y_pred_clean = temp["y_pred"]

    mae = mean_absolute_error(y_true_clean, y_pred_clean)

    rmse = np.sqrt(
        mean_squared_error(y_true_clean, y_pred_clean)
    )

    mape = (
        np.abs(y_true_clean - y_pred_clean) / y_true_clean
    ).mean() * 100

    bias = (
        (y_pred_clean - y_true_clean) / y_true_clean
    ).mean() * 100

    return {
        "rows": len(temp),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Bias": bias
    }


# =========================================================
# TARGET
# =========================================================

y = df["gt_value"]


# =========================================================
# SIMPLE SOURCE BENCHMARKS
# =========================================================

results = {}

results["source_1"] = evaluate(
    y,
    df["source_1"]
)

results["source_2"] = evaluate(
    y,
    df["source_2"]
)

results["source_3"] = evaluate(
    y,
    df["source_3"]
)

results["source_4"] = evaluate(
    y,
    df["source_4"]
)

results["mean_1_4"] = evaluate(
    y,
    (
        df["source_1"]
        + df["source_4"]
    ) / 2
)


# =========================================================
# FEATURE SETS
# =========================================================

source_features_without_4 = [
    "source_1",
    "source_2",
    "source_3"
]

source_features_with_4 = [
    "source_1",
    "source_2",
    "source_3",
    "source_4"
]

numeric_property_features = [
    "latitude",
    "longitude",
    "plot_area",
    "floor_area",
    "build_year",
    "volume"
]

woz_features = [
    "woz_value",
    "woz_days_diff"
]

categorical_features = [
    "province",
    "municipality",
    "object_type",
    "energy_label"
]


# =========================================================
# MODELING DATA
# =========================================================

X_base = df[
    source_features_without_4
    + numeric_property_features
    + woz_features
    + categorical_features
]

X_s4 = df[
    source_features_with_4
    + numeric_property_features
    + woz_features
    + categorical_features
]


# =========================================================
# SAME TRAIN/TEST SPLIT FOR BOTH MODELS
# =========================================================

Xb_train, Xb_test, y_train, y_test = train_test_split(
    X_base,
    y,
    test_size=0.2,
    random_state=42
)

Xs_train, Xs_test, _, _ = train_test_split(
    X_s4,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# PREPROCESSING
# =========================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="missing"
            )
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# =========================================================
# BASELINE RF WITHOUT SOURCE 4
# =========================================================

base_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            source_features_without_4
            + numeric_property_features
            + woz_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

base_model = Pipeline(
    steps=[
        ("preprocessor", base_preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                max_depth=12
            )
        )
    ]
)

print("\nTraining baseline RF without source 4...")

base_model.fit(Xb_train, y_train)

base_pred = base_model.predict(Xb_test)

results["rf_without_source4"] = evaluate(
    y_test,
    base_pred
)


# =========================================================
# RF WITH SOURCE 4
# =========================================================

s4_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            source_features_with_4
            + numeric_property_features
            + woz_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

s4_model = Pipeline(
    steps=[
        ("preprocessor", s4_preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                max_depth=12
            )
        )
    ]
)

print("\nTraining RF with source 4...")

s4_model.fit(Xs_train, y_train)

s4_pred = s4_model.predict(Xs_test)

results["rf_with_source4"] = evaluate(
    y_test,
    s4_pred
)


# =========================================================
# RESULTS
# =========================================================

results_df = pd.DataFrame(results).T

print("\nSOURCE 4 BENCHMARK RESULTS")
print(results_df.sort_values("MAPE"))


# =========================================================
# IMPROVEMENT
# =========================================================

baseline_mape = results_df.loc[
    "rf_without_source4",
    "MAPE"
]

source4_mape = results_df.loc[
    "rf_with_source4",
    "MAPE"
]

print("\nMAPE improvement from source 4:")
print(baseline_mape - source4_mape)


# =========================================================
# SAVE
# =========================================================

results_df.to_csv(
    "source4_benchmark_results.csv"
)

print("\nSaved: source4_benchmark_results.csv")