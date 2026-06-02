# Train/test splitters. Each returns (train_index, test_index) as pandas Index
# labels over the passed frame, so every method can be scored on identical rows.
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split


def random_split(df, test_size=0.2, seed=42):
    train, test = train_test_split(df.index, test_size=test_size, random_state=seed)
    return train, test


def grouped_split(df, group="property_key", test_size=0.2, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df[group]))
    return df.index[tr], df.index[te]


def grouped_kfold(df, group="property_key", n_splits=5):
    # yields (train_index, test_index) per fold; every property tested exactly once
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(df, groups=df[group]):
        yield df.index[tr], df.index[te]


def time_split(df, date_col="gt_date", test_size=0.2):
    # train on the oldest rows, test on the newest fraction
    order = pd.to_datetime(df[date_col], errors="coerce").sort_values(kind="stable").index
    n_test = int(round(len(order) * test_size))
    return order[:-n_test], order[-n_test:]


def expanding_time_splits(df, date_col="gt_date", n_splits=5):
    # forward-chaining: train on all earlier blocks, test on the next one
    order = pd.to_datetime(df[date_col], errors="coerce").sort_values(kind="stable").index
    blocks = np.array_split(np.array(order), n_splits + 1)
    for i in range(1, n_splits + 1):
        yield pd.Index(np.concatenate(blocks[:i])), pd.Index(blocks[i])
