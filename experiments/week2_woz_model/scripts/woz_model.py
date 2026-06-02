# Week 2: first RandomForest with WOZ (sources 1-3 + WOZ, no source 4) and its
# feature importance. Reuses the shared modeling pipeline (taxapi.core.modeling).
# in : data/benchmark/benchmark_dataset_with_woz.csv
# out: results/woz_feature_importance.csv, plots/feature_importance.png
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from taxapi.core import modeling

EXP = Path(__file__).resolve().parents[1]
RESULTS, PLOTS = EXP / "results", EXP / "plots"

NUMERIC = (
    modeling.SOURCE_FEATURES_NO_4
    + modeling.NUMERIC_PROPERTY_FEATURES
    + modeling.WOZ_FEATURES
)
CAT = modeling.CATEGORICAL_FEATURES


def main():
    df = modeling.load_benchmark("benchmark_dataset_with_woz.csv")
    df = df[df[modeling.SOURCE_FEATURES_NO_4].notna().all(axis=1)].copy()
    print("Week-2 (sources 1-3 + WOZ) shape:", df.shape)

    X, y = df[NUMERIC + CAT], df["gt_value"]
    model = modeling.build_model(NUMERIC, CAT).fit(X, y)

    ohe = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    names = NUMERIC + list(ohe.get_feature_names_out(CAT))
    imp = pd.DataFrame(
        {"Feature": names, "Importance": model.named_steps["model"].feature_importances_}
    )

    def base(f):
        for c in CAT:
            if f.startswith(c + "_"):
                return c
        return f

    agg = (
        imp.assign(group=imp["Feature"].map(base))
        .groupby("group")["Importance"].sum()
        .sort_values(ascending=False)
    )
    print(agg.head(12))

    RESULTS.mkdir(parents=True, exist_ok=True)
    agg.rename("importance").rename_axis("feature").reset_index().to_csv(
        RESULTS / "woz_feature_importance.csv", index=False
    )

    PLOTS.mkdir(parents=True, exist_ok=True)
    top = agg.head(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index[::-1], top.values[::-1], color="#1f77b4")
    ax.set_xlabel("importance")
    ax.set_title("Feature importance (WOZ model)")
    fig.tight_layout()
    fig.savefig(PLOTS / "feature_importance.png", dpi=120)
    plt.close(fig)
    print("Saved results + plots")


if __name__ == "__main__":
    main()
