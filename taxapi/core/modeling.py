import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from taxapi.core import paths

TARGET = "gt_value"

SOURCE_FEATURES = ["source_1", "source_2", "source_3", "source_4"]
SOURCE_FEATURES_NO_4 = ["source_1", "source_2", "source_3"]

NUMERIC_PROPERTY_FEATURES = [
    "latitude", "longitude", "plot_area", "floor_area", "build_year", "volume",
]

WOZ_FEATURES = ["woz_value", "woz_days_diff"]
WOZ_FEATURES_WITH_SOURCE4 = ["woz_value", "woz_days_diff", "source_4_days_diff"]

CATEGORICAL_FEATURES = ["province", "municipality", "object_type", "energy_label"]

RF_PARAMS = {"n_estimators": 200, "random_state": 42, "n_jobs": -1, "max_depth": 12}


def load_benchmark(name="benchmark_dataset_with_woz_and_source4.csv"):
    return pd.read_csv(paths.BENCHMARK_DIR / name, low_memory=False)


def build_model(numeric_features, categorical_features, estimator=None, **rf_kwargs):
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_features)),
            ("cat", categorical_transformer, list(categorical_features)),
        ]
    )
    model = estimator if estimator is not None else RandomForestRegressor(**{**RF_PARAMS, **rf_kwargs})
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def split(X, y, test_size=0.2, random_state=42):
    # random split for now; a time-aware split is the planned follow-up
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
