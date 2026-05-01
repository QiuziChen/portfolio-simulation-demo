from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from itertools import combinations
from typing import Callable, Dict, List, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from .utility import MetricSummary, default_sensing_utility, summarize_metrics


UtilityFunction = Callable[[pd.DataFrame, pd.DataFrame], float]

# Module-level result cache: keyed on (activity_df id, stay_df id, grid_gdf id, vehicle tuple).
# DataFrames are lru_cached at load time so their id() is stable within a session.
_postal_eval_cache: Dict[Tuple, "PostalSimulationResult"] = {}

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


@lru_cache(maxsize=8)
def load_grid_and_population(
    grid_path: str | Path = "data/grid_100m.gpkg",
    population_path: str | Path = "data/swiss_population.csv",
) -> gpd.GeoDataFrame:
    grid = gpd.read_file(resolve_path(grid_path))
    population = pd.read_csv(resolve_path(population_path))
    grid = grid.merge(population, on=["easting", "northing"], how="left").fillna({"residents": 0})
    if "grid_id" not in grid.columns:
        grid = grid.reset_index().rename(columns={"index": "grid_id"})
    return grid


@lru_cache(maxsize=8)
def load_postal_activity(
    trajectory_path: str | Path = "data/agg_grid_1H.csv",
    stay_path: str | Path = "data/agg_grid_stay_1H.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trajectory = pd.read_csv(resolve_path(trajectory_path))
    stay = pd.read_csv(resolve_path(stay_path))
    if "datetime" in trajectory.columns:
        trajectory["datetime"] = pd.to_datetime(trajectory["datetime"])
    if "datetime" in stay.columns:
        stay["datetime"] = pd.to_datetime(stay["datetime"])
    return trajectory, stay


@dataclass(frozen=True)
class PostalSimulationResult:
    composition: Tuple[str, ...]
    vehicle_ids: Tuple[str, ...]
    summary: MetricSummary
    coverage_rate: float
    population_coverage_rate: float
    per_grid_statistics: pd.DataFrame


@dataclass(eq=False)
class VehicleGridMatrix:
    """Precomputed daily-average sensing time matrix (vehicles × grid cells).

    ``data[v, g]`` is the average total duration (in data units, typically seconds)
    that vehicle ``v`` spent in grid ``g`` per day, averaged across all observed dates.
    Built once from ``agg_grid_1H.csv`` and optionally persisted to disk.
    """
    vehicle_ids: Tuple[str, ...]
    grid_ids: np.ndarray             # shape (G,) int
    data: np.ndarray                 # shape (V, G) float64
    vehicle_index: Dict[str, int]    # vehicle_id -> row index
    grid_index: Dict[int, int]       # grid_id   -> col index

    def lookup(self, selected_ids: Sequence[str]) -> pd.DataFrame:
        """Sum daily-avg times across selected vehicles → per-grid stats DataFrame."""
        indices = [self.vehicle_index[vid] for vid in selected_ids]
        times = self.data[indices].sum(axis=0)  # shape (G,)
        return pd.DataFrame({"grid_id": self.grid_ids, "mean_total_duration": times})


def _normalize_vehicle_id(value) -> str:
    if pd.isna(value):
        raise ValueError("Postal vehicle ID cannot be missing.")
    if isinstance(value, str):
        return value
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def enumerate_combinations(vehicle_ids: Sequence[str], fleet_size: int, max_combinations: int = 5000):
    vehicle_ids = tuple(vehicle_ids)
    total = int(math.comb(len(vehicle_ids), fleet_size)) if fleet_size <= len(vehicle_ids) else 0
    if total == 0:
        return []
    if total <= max_combinations:
        return list(combinations(vehicle_ids, fleet_size))
    rng = np.random.default_rng(42)
    sampled = set()
    target = max_combinations
    while len(sampled) < target:
        sample = tuple(sorted(rng.choice(vehicle_ids, size=fleet_size, replace=False).tolist()))
        sampled.add(sample)
    return sorted(sampled)


def _prepare_grid_statistics(activity_subset: pd.DataFrame, stay_subset: pd.DataFrame) -> pd.DataFrame:
    """Build per-grid statistics using only movement (trajectory) data from agg_grid_1H."""
    if activity_subset.empty:
        return pd.DataFrame(columns=["grid_id", "mean_total_duration", "num_records"])

    activity_frame = activity_subset.copy()
    duration_source = None
    for candidate in ("total_duration", "duration", "num"):
        if candidate in activity_frame.columns:
            duration_source = candidate
            break
    if duration_source is None:
        activity_frame["mean_total_duration"] = 0.0
        duration_source = "mean_total_duration"
    result = activity_frame.groupby("grid_id").agg(
        mean_total_duration=(duration_source, "mean"),
        num_records=("grid_id", "size"),
    ).reset_index()
    result = result.fillna({"mean_total_duration": 0.0, "num_records": 0})
    return result[["grid_id", "mean_total_duration", "num_records"]]


def build_vehicle_grid_matrix(
    activity_df: pd.DataFrame,
    grid_gdf: pd.DataFrame,
    stay_df: pd.DataFrame | None = None,
    save_path: str | Path | None = None,
) -> VehicleGridMatrix:
    """Build the (vehicles × grids) daily-average sensing time matrix.

    ``data[v, g]`` = daily-average movement duration + daily-average stay duration
    for vehicle *v* in grid *g*.  Both components are averaged over the **total number
    of days that vehicle operated** (not just days it visited that grid), so a vehicle
    active for 100 days that visited a grid on only 5 of them contributes
    ``total_time_in_grid / 100`` for each source.

    Pass *stay_df* (from ``agg_grid_stay_1H.csv``) to include stationary stay time on
    top of the driving/movement duration from *activity_df*.

    Pass *save_path* to persist the result as a ``.npz`` file for fast reuse.
    """
    dur_col = next(
        (c for c in ("total_duration", "duration", "num") if c in activity_df.columns),
        None,
    )
    if dur_col is None:
        raise ValueError(
            "No duration column found in activity_df "
            "(expected one of: total_duration, duration, num)."
        )

    df = activity_df[["n", "grid_id", dur_col]].copy()
    df["n"] = df["n"].map(_normalize_vehicle_id)

    vehicle_days = None
    if "datetime" in activity_df.columns and pd.api.types.is_datetime64_any_dtype(activity_df["datetime"]):
        df["date"] = activity_df["datetime"].dt.date.values
        # Total operating days per vehicle (denominator is the same for all grids of that vehicle)
        vehicle_days = df.groupby("n", sort=False)["date"].nunique().rename("n_days")
        # Sum duration within each (vehicle, grid, date), then sum across dates
        daily = df.groupby(["n", "grid_id", "date"], sort=False)[dur_col].sum().reset_index()
        total = daily.groupby(["n", "grid_id"], sort=False)[dur_col].sum().reset_index()
        # Divide by total operating days, not by days the vehicle visited that grid
        total = total.join(vehicle_days, on="n")
        total[dur_col] = total[dur_col] / total["n_days"]
        agg = total[["n", "grid_id", dur_col]]
    else:
        agg = df.groupby(["n", "grid_id"], sort=False)[dur_col].sum().reset_index()

    # Add stay duration (total_stay_duration) aligned to the same daily-average basis
    if stay_df is not None and not stay_df.empty and "total_stay_duration" in stay_df.columns:
        stay = stay_df[["n", "grid_id", "total_stay_duration"]].copy()
        stay["n"] = stay["n"].map(_normalize_vehicle_id)
        if (
            "datetime" in stay_df.columns
            and pd.api.types.is_datetime64_any_dtype(stay_df["datetime"])
            and vehicle_days is not None
        ):
            stay["date"] = stay_df["datetime"].dt.date.values
            daily_stay = stay.groupby(["n", "grid_id", "date"], sort=False)["total_stay_duration"].sum().reset_index()
            total_stay = daily_stay.groupby(["n", "grid_id"], sort=False)["total_stay_duration"].sum().reset_index()
            # Reuse the same vehicle_days denominator from activity_df
            total_stay = total_stay.join(vehicle_days, on="n")
            total_stay["total_stay_duration"] = total_stay["total_stay_duration"] / total_stay["n_days"].clip(lower=1)
            agg_stay = total_stay[["n", "grid_id", "total_stay_duration"]]
        else:
            agg_stay = stay.groupby(["n", "grid_id"], sort=False)["total_stay_duration"].sum().reset_index()
        agg = agg.merge(agg_stay, on=["n", "grid_id"], how="outer").fillna(0.0)
        agg[dur_col] = agg[dur_col] + agg["total_stay_duration"]
        agg = agg[["n", "grid_id", dur_col]]

    all_vehicle_ids = sorted(agg["n"].unique())
    all_grid_ids = sorted(int(g) for g in grid_gdf["grid_id"].unique())

    pivot = (
        agg.pivot(index="n", columns="grid_id", values=dur_col)
        .reindex(index=all_vehicle_ids, columns=all_grid_ids, fill_value=0.0)
        .fillna(0.0)
    )

    data = pivot.to_numpy(dtype=float)
    grid_ids_arr = np.array(all_grid_ids, dtype=int)
    vehicle_index = {v: i for i, v in enumerate(all_vehicle_ids)}
    grid_index = {g: i for i, g in enumerate(all_grid_ids)}

    if save_path is not None:
        p = resolve_path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            data=data,
            vehicle_ids=np.array(list(all_vehicle_ids)),
            grid_ids=grid_ids_arr,
        )

    return VehicleGridMatrix(
        vehicle_ids=tuple(all_vehicle_ids),
        grid_ids=grid_ids_arr,
        data=data,
        vehicle_index=vehicle_index,
        grid_index=grid_index,
    )


def _load_vehicle_grid_matrix(path: Path) -> VehicleGridMatrix:
    """Load a VehicleGridMatrix from a previously saved .npz file."""
    f = np.load(path)
    vehicle_ids = tuple(str(v) for v in f["vehicle_ids"])
    grid_ids = f["grid_ids"].astype(int)
    data = f["data"].astype(float)
    vehicle_index = {v: i for i, v in enumerate(vehicle_ids)}
    grid_index = {int(g): i for i, g in enumerate(grid_ids)}
    return VehicleGridMatrix(
        vehicle_ids=vehicle_ids,
        grid_ids=grid_ids,
        data=data,
        vehicle_index=vehicle_index,
        grid_index=grid_index,
    )


@lru_cache(maxsize=4)
def get_vehicle_grid_matrix(
    matrix_path: str | Path = "data/vehicle_grid_matrix.npz",
) -> VehicleGridMatrix:
    """Load the precomputed vehicle-grid matrix from disk.

    The result is cached in-process after the first call.
    """
    p = resolve_path(matrix_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Precomputed vehicle-grid matrix not found at {p}. "
            "Ensure data/vehicle_grid_matrix.npz is present in the repository."
        )
    return _load_vehicle_grid_matrix(p)


def evaluate_postal_composition(
    activity_df: pd.DataFrame,
    stay_df: pd.DataFrame,
    grid_gdf: pd.DataFrame,
    selected_vehicle_ids: Sequence[str],
    utility_fn: UtilityFunction | None = None,
    vehicle_grid_matrix: VehicleGridMatrix | None = None,
) -> PostalSimulationResult:
    selected_vehicle_ids = tuple(sorted(_normalize_vehicle_id(vehicle_id) for vehicle_id in selected_vehicle_ids))

    # Cache lookup: only cache when using the default utility function, since results are deterministic
    # per vehicle subset and the DataFrames / matrix are lru_cached (stable id) within a session.
    if utility_fn is None:
        if vehicle_grid_matrix is not None:
            cache_key = (id(vehicle_grid_matrix), selected_vehicle_ids)
        else:
            cache_key = (id(activity_df), id(grid_gdf), selected_vehicle_ids)
        cached = _postal_eval_cache.get(cache_key)
        if cached is not None:
            return cached

    if vehicle_grid_matrix is not None:
        per_grid_statistics = vehicle_grid_matrix.lookup(selected_vehicle_ids)
    else:
        vehicle_mask = activity_df["n"].isin(selected_vehicle_ids)
        activity_subset = activity_df.loc[vehicle_mask].copy()
        if "n" in stay_df.columns:
            stay_subset = stay_df[stay_df["n"].isin(selected_vehicle_ids)].copy()
        else:
            stay_subset = stay_df.copy()
        per_grid_statistics = _prepare_grid_statistics(activity_subset, stay_subset)
    coverage_rate = float(per_grid_statistics["grid_id"].nunique() / max(len(grid_gdf), 1))
    if "residents" in grid_gdf.columns and not per_grid_statistics.empty:
        visited = per_grid_statistics.merge(grid_gdf[["grid_id", "residents"]], on="grid_id", how="left").fillna({"residents": 0})
        population_coverage_rate = float(visited["residents"].sum() / max(grid_gdf["residents"].sum(), 1.0))
    else:
        population_coverage_rate = coverage_rate

    utility_fn = utility_fn or default_sensing_utility
    summary_value = utility_fn(per_grid_statistics, grid_gdf)
    summary = summarize_metrics([summary_value])

    result = PostalSimulationResult(
        composition=selected_vehicle_ids,
        vehicle_ids=selected_vehicle_ids,
        summary=summary,
        coverage_rate=coverage_rate,
        population_coverage_rate=population_coverage_rate,
        per_grid_statistics=per_grid_statistics,
    )
    # Store in cache (default utility_fn only — custom functions bypass cache)
    if utility_fn is default_sensing_utility:
        if vehicle_grid_matrix is not None:
            _postal_eval_cache[(id(vehicle_grid_matrix), selected_vehicle_ids)] = result
        else:
            _postal_eval_cache[(id(activity_df), id(grid_gdf), selected_vehicle_ids)] = result
    return result


def run_postal_fleet_search(
    activity_df: pd.DataFrame,
    stay_df: pd.DataFrame,
    grid_gdf: pd.DataFrame,
    fleet_size: int,
    max_combinations: int = 5000,
    utility_fn: UtilityFunction | None = None,
) -> pd.DataFrame:
    vehicle_ids = available_postal_vehicle_ids(activity_df)
    results: List[dict] = []
    for composition in enumerate_combinations(vehicle_ids, fleet_size=fleet_size, max_combinations=max_combinations):
        simulation = evaluate_postal_composition(activity_df, stay_df, grid_gdf, composition, utility_fn=utility_fn)
        results.append(
            {
                "composition": composition,
                "expected_utility": simulation.summary.mean,
                "utility_std": simulation.summary.std,
                "cv": simulation.summary.cv,
                "coverage_rate": simulation.coverage_rate,
                "population_coverage_rate": simulation.population_coverage_rate,
                "fleet_size": fleet_size,
            }
        )
    return pd.DataFrame(results)


def available_postal_vehicle_ids(activity_df: pd.DataFrame) -> List[str]:
    values = activity_df["n"].dropna().map(_normalize_vehicle_id).unique().tolist()
    return sorted(values)


def sample_postal_run(
    activity_df: pd.DataFrame,
    stay_df: pd.DataFrame,
    grid_gdf: pd.DataFrame,
    n_postal: int,
    rng: np.random.Generator,
    vehicle_ids: Sequence[str] | None = None,
    utility_fn: UtilityFunction | None = None,
    vehicle_grid_matrix: VehicleGridMatrix | None = None,
) -> PostalSimulationResult:
    vehicle_ids = tuple(vehicle_ids) if vehicle_ids is not None else tuple(available_postal_vehicle_ids(activity_df))
    if n_postal <= 0:
        empty_stats = pd.DataFrame(columns=["grid_id", "mean_total_duration", "num_records"])
        return PostalSimulationResult(
            composition=tuple(),
            vehicle_ids=tuple(),
            summary=summarize_metrics([0.0]),
            coverage_rate=0.0,
            population_coverage_rate=0.0,
            per_grid_statistics=empty_stats,
        )
    if n_postal > len(vehicle_ids):
        raise ValueError(f"Requested n_postal={n_postal} but only {len(vehicle_ids)} postal vehicles are available.")
    selected = tuple(sorted(_normalize_vehicle_id(v) for v in rng.choice(vehicle_ids, size=n_postal, replace=False).tolist()))
    return evaluate_postal_composition(
        activity_df, stay_df, grid_gdf,
        selected_vehicle_ids=selected,
        utility_fn=utility_fn,
        vehicle_grid_matrix=vehicle_grid_matrix,
    )
