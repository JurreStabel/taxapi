import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate(y_true, y_pred):
    # drop rows where either value is missing, then summarise
    temp = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    y_true_clean = temp["y_true"]
    y_pred_clean = temp["y_pred"]

    abs_rel = np.abs(y_pred_clean - y_true_clean) / y_true_clean
    signed_rel = (y_pred_clean - y_true_clean) / y_true_clean

    return {
        "rows": len(temp),
        "MAE": mean_absolute_error(y_true_clean, y_pred_clean),
        "RMSE": np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
        "MAPE": abs_rel.mean() * 100,
        "MdAPE": abs_rel.median() * 100,
        "within_10pct": (abs_rel < 0.10).mean() * 100,
        "Bias": signed_rel.mean() * 100,
    }


def add_error_columns(df, truth="gt_value", pred="prediction"):
    df["absolute_error"] = (df[pred] - df[truth]).abs()
    df["relative_error"] = (df[pred] - df[truth]) / df[truth]
    df["absolute_relative_error"] = df["relative_error"].abs()
    return df


def error_summary(df, group_col):
    return (
        df.groupby(group_col, observed=False)
        .agg(
            rows=("gt_value", "count"),
            mean_gt_value=("gt_value", "mean"),
            MAE=("absolute_error", "mean"),
            MAPE=("absolute_relative_error", lambda x: x.mean() * 100),
            Bias=("relative_error", lambda x: x.mean() * 100),
        )
        .reset_index()
    )
