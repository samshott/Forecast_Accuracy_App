from __future__ import annotations

from typing import Callable, Iterable, List, Optional

import numpy as np
import pandas as pd


def mae(errors: np.ndarray) -> float:
    return float(np.mean(np.abs(errors))) if errors.size else float("nan")


def rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(errors)))) if errors.size else float("nan")


def bias(errors: np.ndarray) -> float:
    return float(np.mean(errors)) if errors.size else float("nan")


METRICS: dict[str, Callable[[np.ndarray], float]] = {
    "mae": mae,
    "rmse": rmse,
    "bias": bias,
}


def score_pairs(
    pairs_df: pd.DataFrame,
    *,
    metrics: Optional[Iterable[str]] = None,
    group_fields: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    if pairs_df.empty:
        metric_cols = list(metrics or METRICS.keys())
        columns = list(group_fields or ["station_id", "var", "lead_hours"]) + metric_cols + ["n"]
        return pd.DataFrame(columns=columns)

    metric_funcs = {name: METRICS[name] for name in (metrics or METRICS.keys())}
    group_cols = list(group_fields or ["station_id", "var", "lead_hours"])
    rows: List[dict] = []
    for keys, group in pairs_df.groupby(group_cols):
        record = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        errors = (group["f_value"] - group["o_value"]).to_numpy(dtype=float)
        for name, func in metric_funcs.items():
            record[name] = func(errors)
        record["n"] = int(len(group))
        rows.append(record)
    result = pd.DataFrame(rows)
    return result.sort_values(group_cols).reset_index(drop=True)


__all__ = ["mae", "rmse", "bias", "score_pairs"]
