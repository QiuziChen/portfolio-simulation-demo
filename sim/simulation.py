from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .postal_sim import load_grid_and_population as load_postal_grid_and_population
from .postal_sim import sample_postal_run, VehicleGridMatrix, get_vehicle_grid_matrix
from .ridehailing_sim import RideHailingContext, RoadGraph, build_ridehailing_context, load_road_graph, simulate_ridehailing_composition
from .utility import MetricSummary, default_sensing_utility, summarize_metrics


ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class CompositionJob:
    n_postal: int
    n_ridehailing: int
    m_runs: int
    labor_time_limit_hours: float
    break_min_minutes: float
    break_max_minutes: float
    grid_gdf: pd.DataFrame
    postal_vehicle_ids: Sequence[str]
    seed: int
    activity_df: Optional[pd.DataFrame] = None
    stay_df: Optional[pd.DataFrame] = None
    road_graph: Optional[RoadGraph] = None
    ride_context: Optional[RideHailingContext] = None
    ridehailing_pool_matrix: Optional["RideHailingPoolMatrix"] = None
    vehicle_grid_matrix: Optional[VehicleGridMatrix] = None


def _run_composition_job(job: CompositionJob) -> Tuple[Tuple[int, int], SingleCompositionStats]:
    stats = _run_single_composition_with_data(
        n_postal=job.n_postal,
        n_ridehailing=job.n_ridehailing,
        m_runs=job.m_runs,
        labor_time_limit_hours=job.labor_time_limit_hours,
        break_min_minutes=job.break_min_minutes,
        break_max_minutes=job.break_max_minutes,
        grid_gdf=job.grid_gdf,
        postal_vehicle_ids=job.postal_vehicle_ids,
        seed=job.seed,
        activity_df=job.activity_df,
        stay_df=job.stay_df,
        road_graph=job.road_graph,
        ride_context=job.ride_context,
        ridehailing_pool_matrix=job.ridehailing_pool_matrix,
        vehicle_grid_matrix=job.vehicle_grid_matrix,
    )
    return (int(job.n_postal), int(job.n_ridehailing)), stats


@dataclass(frozen=True)
class RideHailingPoolJob:
    n_ridehailing: int
    seed: int
    labor_time_limit_hours: float
    break_min_minutes: float
    break_max_minutes: float
    road_graph: RoadGraph
    grid_gdf: pd.DataFrame
    ride_context: RideHailingContext


def _run_ridehailing_pool_job(job: RideHailingPoolJob) -> pd.DataFrame:
    res = simulate_ridehailing_composition(
        vehicle_ids=list(range(job.n_ridehailing)),
        road_graph=job.road_graph,
        grid_gdf=job.grid_gdf,
        context=job.ride_context,
        labor_time_limit_hours=job.labor_time_limit_hours,
        break_min_minutes=job.break_min_minutes,
        break_max_minutes=job.break_max_minutes,
        seed=job.seed,
    )
    return _matrix_from_ridehailing_result(res)


@dataclass(frozen=True)
class ParetoProtocol:
    mean_column: str = "expected_utility"
    variance_column: str = "utility_variance"
    var_column: str = "value_at_risk"
    composition_column: str = "composition"


@dataclass(frozen=True)
class SingleCompositionStats:
    n_postal: int
    n_ridehailing: int
    total_vehicles: int
    m_runs: int
    expected_utility: float
    utility_std: float
    cv: float
    utility_q05: float
    utility_q95: float
    utility_summary: MetricSummary
    mean_travel_time_matrix: pd.DataFrame
    std_travel_time_matrix: pd.DataFrame
    cv_travel_time_matrix: pd.DataFrame
    q05_travel_time_matrix: pd.DataFrame
    q95_travel_time_matrix: pd.DataFrame


def _matrix_from_postal_result(postal_result) -> pd.DataFrame:
    if postal_result.per_grid_statistics.empty:
        return pd.DataFrame(columns=["grid_id", "postal_time"])
    return postal_result.per_grid_statistics[["grid_id", "mean_total_duration"]].rename(
        columns={"mean_total_duration": "postal_time"}
    )


def _matrix_from_ridehailing_result(ride_result) -> pd.DataFrame:
    if ride_result.per_grid_travel_time.empty:
        return pd.DataFrame(columns=["grid_id", "ridehailing_time"])
    return ride_result.per_grid_travel_time[["grid_id", "mean_running_time"]].rename(
        columns={"mean_running_time": "ridehailing_time"}
    )


def _combine_matrix(grid_gdf: pd.DataFrame, postal_matrix: pd.DataFrame, ride_matrix: pd.DataFrame) -> pd.DataFrame:
    combined = grid_gdf[["grid_id"]].copy()
    combined = combined.merge(postal_matrix, on="grid_id", how="left")
    combined = combined.merge(ride_matrix, on="grid_id", how="left")
    combined = combined.fillna({"postal_time": 0.0, "ridehailing_time": 0.0}).infer_objects(copy=False)
    combined["total_time"] = combined["postal_time"] + combined["ridehailing_time"]
    return combined


def _utility_from_combined_matrix(combined_matrix: pd.DataFrame, grid_gdf: pd.DataFrame) -> float:
    """Compute sensing utility from the combined postal+ridehailing grid matrix.

    Delegates to :func:`default_sensing_utility` using the ``total_time`` column
    as the duration signal, keeping the MC utility consistent with the MSRP
    exponential formulation used everywhere else.
    """
    grid_stats = combined_matrix[["grid_id", "total_time"]].rename(
        columns={"total_time": "mean_total_duration"}
    )
    return default_sensing_utility(grid_stats, grid_gdf)


def _summarize_matrices(run_matrices: List[pd.DataFrame], grid_ids: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix_values = np.vstack([
        frame.set_index("grid_id").reindex(grid_ids, fill_value=0.0)["total_time"].to_numpy(dtype=float)
        for frame in run_matrices
    ])
    mean_values = matrix_values.mean(axis=0)
    std_values = matrix_values.std(axis=0)
    nonzero = mean_values != 0
    cv_values = np.zeros_like(mean_values)
    np.divide(std_values, mean_values, out=cv_values, where=nonzero)

    mean_df = pd.DataFrame({"grid_id": grid_ids, "total_time": mean_values})
    std_df = pd.DataFrame({"grid_id": grid_ids, "total_time": std_values})
    cv_df = pd.DataFrame({"grid_id": grid_ids, "total_time": cv_values})
    return mean_df, std_df, cv_df


def _quantile_index(values: Sequence[float], quantile: float) -> int:
    if not values:
        return 0
    order = np.argsort(np.asarray(values, dtype=float))
    idx = int(np.clip(round((len(order) - 1) * quantile), 0, len(order) - 1))
    return int(order[idx])


@dataclass(eq=False)
class RideHailingPoolMatrix:
    """Precomputed ride-hailing single-vehicle simulation pool as a (P × G) matrix.

    ``data[p, g]`` is the total ``ridehailing_time`` (seconds) that pool sample *p*
    accumulated in grid cell *g* during its simulated shift.  Zero means the cell
    was not visited.

    At simulation time, drawing *n_ridehailing* vehicles is a pure numpy operation::

        rows = rng.integers(0, P, size=n)
        combined_times = data[rows].sum(axis=0)   # shape (G,)
    """
    grid_ids: np.ndarray          # shape (G,) int
    data: np.ndarray              # shape (P, G) float64
    grid_index: Dict[int, int]    # grid_id -> column index

    @property
    def pool_size(self) -> int:
        return int(self.data.shape[0])

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return summed ridehailing times for *n* randomly drawn pool samples.

        Returns a zero array when *n* <= 0.  Output shape: (G,).
        """
        if n <= 0:
            return np.zeros(len(self.grid_ids), dtype=float)
        indices = rng.integers(0, self.pool_size, size=n)
        return self.data[indices].sum(axis=0)

    def to_dataframe(self, times: np.ndarray) -> pd.DataFrame:
        """Wrap a (G,) time array as a DataFrame with columns [grid_id, ridehailing_time]."""
        return pd.DataFrame({"grid_id": self.grid_ids, "ridehailing_time": times})


def build_ridehailing_pool_matrix(
    pool: List[pd.DataFrame],
    grid_gdf: pd.DataFrame,
) -> "RideHailingPoolMatrix":
    """Convert a list of single-vehicle DataFrames into a dense (P × G) matrix.

    Each entry ``data[p, g]`` is the ``ridehailing_time`` for pool sample *p* in
    grid cell *g*.  Grid cells not visited by a sample receive 0.
    """
    all_grid_ids = np.array(sorted(int(g) for g in grid_gdf["grid_id"].unique()), dtype=int)
    grid_index: Dict[int, int] = {int(g): i for i, g in enumerate(all_grid_ids)}
    G = len(all_grid_ids)
    data = np.zeros((len(pool), G), dtype=float)
    for p, df in enumerate(pool):
        if df.empty:
            continue
        gids = df["grid_id"].to_numpy(dtype=int)
        times = df["ridehailing_time"].to_numpy(dtype=float)
        valid = np.array([g in grid_index for g in gids])
        if valid.any():
            col_indices = np.array([grid_index[int(g)] for g in gids[valid]])
            data[p, col_indices] = times[valid]
    return RideHailingPoolMatrix(grid_ids=all_grid_ids, data=data, grid_index=grid_index)


def save_ridehailing_pool_matrix(matrix: "RideHailingPoolMatrix", path: str | Path) -> None:
    """Persist a RideHailingPoolMatrix to disk as a compressed .npz file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, data=matrix.data, grid_ids=matrix.grid_ids)


def load_ridehailing_pool_matrix(path: str | Path) -> "RideHailingPoolMatrix":
    """Load a RideHailingPoolMatrix previously saved with :func:`save_ridehailing_pool_matrix`."""
    f = np.load(Path(path))
    grid_ids = f["grid_ids"].astype(int)
    data = f["data"].astype(float)
    grid_index: Dict[int, int] = {int(g): i for i, g in enumerate(grid_ids)}
    return RideHailingPoolMatrix(grid_ids=grid_ids, data=data, grid_index=grid_index)


def _build_ridehailing_pool(
    road_graph: RoadGraph,
    grid_gdf: pd.DataFrame,
    ride_context: RideHailingContext,
    pool_size: int,
    seed: int,
    labor_time_limit_hours: float,
    break_min_minutes: float,
    break_max_minutes: float,
    max_workers: Optional[int] = None,
) -> List[pd.DataFrame]:
    """Generate a pool of single-vehicle ride-hailing simulations in parallel."""
    rng_pool = np.random.default_rng(seed)
    pool_jobs = [
        RideHailingPoolJob(
            n_ridehailing=1,
            seed=int(rng_pool.integers(0, 2**31 - 1)),
            labor_time_limit_hours=labor_time_limit_hours,
            break_min_minutes=break_min_minutes,
            break_max_minutes=break_max_minutes,
            road_graph=road_graph,
            grid_gdf=grid_gdf,
            ride_context=ride_context,
        )
        for _ in range(pool_size)
    ]
    pool: List[pd.DataFrame] = []
    use_parallel = len(pool_jobs) > 1 and (max_workers is None or max_workers != 1)
    if use_parallel:
        worker_count = max_workers or min(len(pool_jobs), os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for future in as_completed({executor.submit(_run_ridehailing_pool_job, job) for job in pool_jobs}):
                pool.append(future.result())
    else:
        for job in pool_jobs:
            pool.append(_run_ridehailing_pool_job(job))
    return pool


def _run_single_composition_with_data(
    n_postal: int,
    n_ridehailing: int,
    m_runs: int,
    labor_time_limit_hours: float,
    break_min_minutes: float,
    break_max_minutes: float,
    grid_gdf: pd.DataFrame,
    postal_vehicle_ids: Sequence[str],
    seed: int,
    activity_df: Optional[pd.DataFrame] = None,
    stay_df: Optional[pd.DataFrame] = None,
    road_graph: Optional[RoadGraph] = None,
    ride_context: Optional[RideHailingContext] = None,
    progress_callback: Optional[ProgressCallback] = None,
    progress_offset: int = 0,
    progress_total: int = 0,
    composition_label: str = "composition",
    ridehailing_pool_matrix: Optional["RideHailingPoolMatrix"] = None,
    vehicle_grid_matrix: Optional[VehicleGridMatrix] = None,
) -> SingleCompositionStats:
    rng = np.random.default_rng(seed)
    utilities: List[float] = []
    run_matrices: List[pd.DataFrame] = []

    for run_number in range(1, m_runs + 1):
        if progress_callback:
            progress_callback(
                progress_offset + run_number - 1,
                progress_total,
                f"{composition_label}: Monte Carlo run {run_number}/{m_runs} starting",
            )
        run_seed = int(rng.integers(0, 2**31 - 1))
        postal_result = sample_postal_run(
            activity_df=activity_df if activity_df is not None else pd.DataFrame(),
            stay_df=stay_df if stay_df is not None else pd.DataFrame(),
            grid_gdf=grid_gdf,
            n_postal=n_postal,
            rng=np.random.default_rng(run_seed),
            vehicle_ids=postal_vehicle_ids,
            vehicle_grid_matrix=vehicle_grid_matrix,
        )

        if ridehailing_pool_matrix is not None:
            times = ridehailing_pool_matrix.sample(n_ridehailing, rng)
            ride_matrix = ridehailing_pool_matrix.to_dataframe(times)
        else:
            ride_vehicle_ids = list(range(max(0, int(n_ridehailing))))
            ride_result = simulate_ridehailing_composition(
                vehicle_ids=ride_vehicle_ids,
                road_graph=road_graph,
                grid_gdf=grid_gdf,
                context=ride_context,
                labor_time_limit_hours=labor_time_limit_hours,
                break_min_minutes=break_min_minutes,
                break_max_minutes=break_max_minutes,
                seed=run_seed + 1,
            )
            ride_matrix = _matrix_from_ridehailing_result(ride_result)

        combined_matrix = _combine_matrix(
            grid_gdf,
            _matrix_from_postal_result(postal_result),
            ride_matrix,
        )
        utilities.append(_utility_from_combined_matrix(combined_matrix, grid_gdf))
        run_matrices.append(combined_matrix[["grid_id", "total_time"]])

        if progress_callback:
            progress_callback(
                progress_offset + run_number,
                progress_total,
                f"{composition_label}: Monte Carlo run {run_number}/{m_runs} completed",
            )

    utility_summary = summarize_metrics(utilities)
    grid_ids = grid_gdf["grid_id"].to_numpy(dtype=int)
    mean_matrix, std_matrix, cv_matrix = _summarize_matrices(run_matrices, grid_ids)

    q05_idx = _quantile_index(utilities, 0.05)
    q95_idx = _quantile_index(utilities, 0.95)
    q05_matrix = run_matrices[q05_idx].copy().reset_index(drop=True)
    q95_matrix = run_matrices[q95_idx].copy().reset_index(drop=True)

    return SingleCompositionStats(
        n_postal=n_postal,
        n_ridehailing=n_ridehailing,
        total_vehicles=n_postal + n_ridehailing,
        m_runs=m_runs,
        expected_utility=utility_summary.mean,
        utility_std=utility_summary.std,
        cv=utility_summary.cv,
        utility_q05=float(np.quantile(np.asarray(utilities, dtype=float), 0.05)),
        utility_q95=float(np.quantile(np.asarray(utilities, dtype=float), 0.95)),
        utility_summary=utility_summary,
        mean_travel_time_matrix=mean_matrix,
        std_travel_time_matrix=std_matrix,
        cv_travel_time_matrix=cv_matrix,
        q05_travel_time_matrix=q05_matrix,
        q95_travel_time_matrix=q95_matrix,
    )


def run_single_composition_simulation(
    n_postal: int,
    n_ridehailing: int,
    m_runs: int = 1000,
    labor_time_limit_hours: float = 5.0,
    avg_speed_kmh: float = 50.0,
    break_min_minutes: float = 0.0,
    break_max_minutes: float = 10.0,
    roads_path: str | Path = "data/lausanne_roads_encoded.gpkg",
    grid_path: str | Path = "data/grid_100m.gpkg",
    population_path: str | Path = "data/swiss_population.csv",
    seed: int = 42,
) -> SingleCompositionStats:
    if m_runs <= 0:
        raise ValueError("m_runs must be positive.")

    grid_gdf = load_postal_grid_and_population(grid_path=grid_path, population_path=population_path)
    vehicle_grid_matrix = get_vehicle_grid_matrix()
    postal_vehicle_ids = list(vehicle_grid_matrix.vehicle_ids)
    road_graph = load_road_graph(path=roads_path, default_speed_kmh=avg_speed_kmh)
    ride_context = build_ridehailing_context(road_graph, grid_gdf)

    return _run_single_composition_with_data(
        n_postal=n_postal,
        n_ridehailing=n_ridehailing,
        m_runs=m_runs,
        labor_time_limit_hours=labor_time_limit_hours,
        break_min_minutes=break_min_minutes,
        break_max_minutes=break_max_minutes,
        grid_gdf=grid_gdf,
        road_graph=road_graph,
        ride_context=ride_context,
        postal_vehicle_ids=postal_vehicle_ids,
        vehicle_grid_matrix=vehicle_grid_matrix,
        seed=seed,
    )


def evaluate_composition_result(
    frame: pd.DataFrame,
    protocol: ParetoProtocol | None = None,
) -> pd.DataFrame:
    protocol = protocol or ParetoProtocol()
    if frame.empty:
        return frame.copy()

    result = frame.copy()
    if protocol.mean_column not in result.columns:
        raise KeyError(f"Missing required column: {protocol.mean_column}")
    if protocol.variance_column not in result.columns and "utility_std" in result.columns:
        result[protocol.variance_column] = result["utility_std"] ** 2
    elif protocol.variance_column not in result.columns:
        result[protocol.variance_column] = 0.0
    if protocol.var_column not in result.columns and "utility_q05" in result.columns:
        result[protocol.var_column] = result["utility_q05"]
    elif protocol.var_column not in result.columns:
        result[protocol.var_column] = 0.0
    return result


def pareto_frontier(
    frame: pd.DataFrame,
    protocol: ParetoProtocol | None = None,
    max_points: int | None = 10,
) -> pd.DataFrame:
    """Return the 2D Pareto frontier (maximize EU and VaR), capped at *max_points*.

    When *max_points* is set and the non-dominated set is smaller than the cap,
    the remaining slots are filled with dominated compositions evenly spaced
    along the EU axis so that the caller gets a richer picture of the trade-off
    space without being flooded with all compositions.
    """
    protocol = protocol or ParetoProtocol()
    if frame.empty:
        return frame.copy()

    # True 2D Pareto: maximize expected_utility AND maximize value_at_risk.
    # Sort by EU desc then VaR desc; keep a point only if its VaR strictly
    # exceeds the best VaR seen among all higher-EU points.
    candidate = frame.sort_values([protocol.mean_column, protocol.var_column], ascending=[False, False]).copy()
    best_var = -np.inf
    pareto_indices: list = []
    for index, row in candidate.iterrows():
        if float(row[protocol.var_column]) > best_var:
            pareto_indices.append(index)
            best_var = float(row[protocol.var_column])

    if max_points is None:
        keep_rows = pareto_indices
    else:
        # Always keep all non-dominated points (up to max_points)
        keep_set = set(pareto_indices)
        keep_rows = pareto_indices[:max_points]
        remaining = max_points - len(keep_rows)
        if remaining > 0:
            # Fill with dominated points, evenly spaced along the EU axis
            dominated = [i for i in candidate.index if i not in keep_set]
            if dominated:
                step = max(1, len(dominated) // remaining)
                keep_rows = keep_rows + dominated[::step][:remaining]

    frontier = candidate.loc[keep_rows].sort_values(protocol.mean_column, ascending=False).reset_index(drop=True)
    if protocol.composition_column in frontier.columns:
        frontier[protocol.composition_column] = frontier[protocol.composition_column].astype(object)
    return frontier


def generate_compositions_for_total(
    total_vehicles: int,
    max_postal: int,
) -> List[Tuple[int, int]]:
    max_postal_for_total = min(max_postal, total_vehicles)
    return [(n_postal, total_vehicles - n_postal) for n_postal in range(0, max_postal_for_total + 1)]


def run_budget_range_pareto(
    total_vehicle_min: int,
    total_vehicle_max: int,
    m_runs: int = 1000,
    labor_time_limit_hours: float = 5.0,
    avg_speed_kmh: float = 50.0,
    break_min_minutes: float = 0.0,
    break_max_minutes: float = 10.0,
    roads_path: str | Path = "data/lausanne_roads_encoded.gpkg",
    grid_path: str | Path = "data/grid_100m.gpkg",
    population_path: str | Path = "data/swiss_population.csv",
    seed: int = 42,
    progress_callback: Optional[ProgressCallback] = None,
    max_workers: Optional[int] = None,
    ridehailing_pool_matrix_path: Optional[str | Path] = "data/ridehailing_pool_matrix.npz",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[int, int], SingleCompositionStats]]:
    if total_vehicle_min > total_vehicle_max:
        raise ValueError("total_vehicle_min must be <= total_vehicle_max")
    if total_vehicle_min < 0:
        raise ValueError("total_vehicle_min must be non-negative")

    grid_gdf = load_postal_grid_and_population(grid_path=grid_path, population_path=population_path)

    postal_matrix = get_vehicle_grid_matrix()
    postal_ids = list(postal_matrix.vehicle_ids)
    max_postal = min(len(postal_ids), 10)

    compositions: List[Tuple[int, int]] = []
    for total in range(int(total_vehicle_min), int(total_vehicle_max) + 1):
        compositions.extend(generate_compositions_for_total(total, max_postal=max_postal))

    steps_per_composition = max(int(m_runs), 0) + 1
    total_steps = len(compositions) * steps_per_composition + 1
    progress_cursor = 0
    if progress_callback:
        progress_callback(progress_cursor, total_steps, "Loading shared inputs")
    progress_cursor += 1

    if progress_callback:
        progress_callback(progress_cursor, total_steps, "Precomputing ride-hailing pools")

    ridehailing_pm: Optional[RideHailingPoolMatrix] = None
    any_ride = any(n > 0 for _, n in compositions)
    if any_ride:
        pool_path = Path(ridehailing_pool_matrix_path) if ridehailing_pool_matrix_path else None
        if pool_path is not None and not pool_path.is_absolute():
            pool_path = Path(roads_path).resolve().parents[1] / pool_path
        if pool_path is not None and pool_path.exists():
            ridehailing_pm = load_ridehailing_pool_matrix(pool_path)
        else:
            # Only load road graph and build context when pool must be generated
            road_graph = load_road_graph(path=roads_path, default_speed_kmh=avg_speed_kmh)
            ride_context = build_ridehailing_context(road_graph, grid_gdf)
            pool_size = min(int(m_runs), 100)
            single_vehicle_pool = _build_ridehailing_pool(
                road_graph=road_graph,
                grid_gdf=grid_gdf,
                ride_context=ride_context,
                pool_size=pool_size,
                seed=seed,
                labor_time_limit_hours=labor_time_limit_hours,
                break_min_minutes=break_min_minutes,
                break_max_minutes=break_max_minutes,
                max_workers=max_workers,
            )
            ridehailing_pm = build_ridehailing_pool_matrix(single_vehicle_pool, grid_gdf)
            if pool_path is not None:
                save_ridehailing_pool_matrix(ridehailing_pm, pool_path)

    rows: List[dict] = []
    stats_by_composition: Dict[Tuple[int, int], SingleCompositionStats] = {}
    jobs = [
        CompositionJob(
            n_postal=n_postal,
            n_ridehailing=n_ridehailing,
            m_runs=m_runs,
            labor_time_limit_hours=labor_time_limit_hours,
            break_min_minutes=break_min_minutes,
            break_max_minutes=break_max_minutes,
            grid_gdf=grid_gdf,
            postal_vehicle_ids=postal_ids,
            seed=seed + idx,
            ridehailing_pool_matrix=ridehailing_pm,
            vehicle_grid_matrix=postal_matrix,
        )
        for idx, (n_postal, n_ridehailing) in enumerate(compositions, start=1)
    ]

    use_parallel = len(jobs) > 1 and (max_workers is None or max_workers != 1)
    if use_parallel:
        worker_count = max_workers or min(len(jobs), os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(_run_composition_job, job): job for job in jobs}
            for completed_index, future in enumerate(as_completed(future_map), start=1):
                composition, stats = future.result()
                job = future_map[future]
                composition_label = f"composition ({job.n_postal} postal, {job.n_ridehailing} ride-hailing)"
                if progress_callback:
                    progress_callback(progress_cursor, total_steps, f"{composition_label}: completed")
                progress_cursor += steps_per_composition
                stats_by_composition[composition] = stats
                rows.append(
                    {
                        "composition": composition,
                        "n_postal": int(job.n_postal),
                        "n_ridehailing": int(job.n_ridehailing),
                        "total_vehicles": int(job.n_postal + job.n_ridehailing),
                        "expected_utility": stats.expected_utility,
                        "utility_std": stats.utility_std,
                        "utility_variance": stats.utility_std ** 2,
                        "cv": stats.cv,
                        "utility_q05": stats.utility_q05,
                        "utility_q95": stats.utility_q95,
                        "value_at_risk": stats.utility_q05,
                        "m_runs": int(m_runs),
                    }
                )
                if progress_callback:
                    progress_callback(progress_cursor, total_steps, f"{composition_label}: stored result {completed_index}/{len(jobs)}")
    else:
        for idx, job in enumerate(jobs, start=1):
            composition_label = f"composition {idx}/{len(jobs)} ({job.n_postal} postal, {job.n_ridehailing} ride-hailing)"
            if progress_callback:
                progress_callback(progress_cursor, total_steps, f"{composition_label}: starting")
            composition, stats = _run_composition_job(job)
            stats_by_composition[composition] = stats
            rows.append(
                {
                    "composition": composition,
                    "n_postal": int(job.n_postal),
                    "n_ridehailing": int(job.n_ridehailing),
                    "total_vehicles": int(job.n_postal + job.n_ridehailing),
                    "expected_utility": stats.expected_utility,
                    "utility_std": stats.utility_std,
                    "utility_variance": stats.utility_std ** 2,
                    "cv": stats.cv,
                    "utility_q05": stats.utility_q05,
                    "utility_q95": stats.utility_q95,
                    "value_at_risk": stats.utility_q05,
                    "m_runs": int(m_runs),
                }
            )
            if progress_callback:
                progress_callback(progress_cursor + steps_per_composition, total_steps, f"{composition_label}: completed")
            progress_cursor += steps_per_composition

    full_results = pd.DataFrame(rows)
    if full_results.empty:
        if progress_callback:
            progress_callback(total_steps, total_steps, "Completed")
        return full_results, full_results.copy(), stats_by_composition

    full_results = full_results.sort_values(["total_vehicles", "n_postal", "n_ridehailing"]).reset_index(drop=True)

    frontier_frames: List[pd.DataFrame] = []
    for total in sorted(full_results["total_vehicles"].unique()):
        subset = full_results[full_results["total_vehicles"] == total].copy()
        frontier_frames.append(pareto_frontier(evaluate_composition_result(subset), max_points=None))
    frontier = pd.concat(frontier_frames, ignore_index=True) if frontier_frames else full_results.iloc[0:0].copy()

    if progress_callback:
        progress_callback(total_steps, total_steps, "Completed")
    return full_results, frontier, stats_by_composition
