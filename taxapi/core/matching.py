# Leakage-safe nearest-valuation matcher shared by the benchmark builders.
import pandas as pd


def attach_nearest(events, candidates, value_col, date_col, window_days=0, diff_col=None):
    # events: rows with columns [event_id, property_key, gt_date].
    # candidates: rows with columns [property_key, date_col, value_col].
    # For each event, picks the candidate closest in time to gt_date, keeping only
    # candidates dated on/before gt_date (window_days=0, strict) or up to window_days
    # after it (relaxed). Ties prefer the prior (less look-ahead). Returns one row per
    # matched event_id with [value_col, date_col, diff_col] where diff = gt_date - date
    # (positive = prior).
    diff_col = diff_col or f"{value_col}_days_diff"
    j = events.merge(candidates, on="property_key", how="inner")
    j["_days"] = (j["gt_date"] - j[date_col]).dt.days
    j = j[j["_days"] >= -window_days].copy()
    j["_abs"] = j["_days"].abs()
    j = j.sort_values(
        ["event_id", "_abs", "_days"], ascending=[True, True, False]
    ).groupby("event_id", as_index=False).first()
    return j[["event_id", value_col, date_col]].assign(**{diff_col: j["_days"].values})
