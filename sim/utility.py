from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float
    cv: float


def summarize_metrics(values: Iterable[float]) -> MetricSummary:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return MetricSummary(mean=0.0, std=0.0, cv=0.0)
    mean = float(array.mean())
    std = float(array.std(ddof=0))
    cv = float(std / mean) if mean else 0.0
    return MetricSummary(mean=mean, std=std, cv=cv)


_ALPHA = 0.92
_DATA_COLLECTION_FREQ_SEC = 10.0  # seconds per sample


def default_sensing_utility(
    grid_stats: pd.DataFrame,
    grid_gdf: pd.DataFrame,
    alpha: float = _ALPHA,
    freq_sec: float = _DATA_COLLECTION_FREQ_SEC,
) -> float:
    """Compute sensing utility using the MSRP exponential saturation model.

    U = 1 - sum_g w_g * exp(-alpha * N_g)

    where:
      w_g  = residents_g / total_residents  (normalised population weight)
      N_g  = mean_total_duration_g / freq_sec  (number of samples collected)
      alpha = saturation control parameter (default 0.92)
    """
    if grid_stats.empty:
        return 0.0
    merged = grid_stats.merge(
        grid_gdf[["grid_id", "residents"]], on="grid_id", how="left"
    ).fillna({"residents": 0, "mean_total_duration": 0.0})

    population_total = float(max(grid_gdf["residents"].sum(), 1.0))
    weights = merged["residents"].to_numpy(dtype=float) / population_total

    duration = merged["mean_total_duration"].to_numpy(dtype=float)
    N = duration / freq_sec

    return float(1.0 - np.dot(weights, np.exp(-alpha * N)))
