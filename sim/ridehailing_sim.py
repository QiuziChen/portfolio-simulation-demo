from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from weakref import WeakValueDictionary
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import networkx as nx
from shapely.geometry import LineString, MultiLineString, Point

from .utility import MetricSummary, summarize_metrics


ProgressCallback = Callable[[int, int, str], None]


_GRAPH_REGISTRY: WeakValueDictionary[int, nx.Graph] = WeakValueDictionary()
_EDGE_TO_GRID_REGISTRY: Dict[int, Dict] = {}


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


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


def _default_travel_time_seconds(geometry: LineString, default_speed_mps: float) -> float:
    length = float(geometry.length)
    if default_speed_mps <= 0:
        default_speed_mps = 10.0
    return length / default_speed_mps


@dataclass(frozen=True)
class RoadGraph:
    graph: nx.Graph
    node_positions: Dict[int, Tuple[float, float]]
    crs: object
    source_path: Path


@dataclass(frozen=True)
class RouteResult:
    path: List[int]
    travel_time: float
    edge_grid_times: Dict[int, float]


def _extract_line_geometries(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if frame.empty:
        raise ValueError("Road graph file is empty.")
    if "geometry" not in frame.columns:
        raise ValueError("Road graph file must contain a geometry column.")
    line_mask = frame.geometry.geom_type.isin(["LineString", "MultiLineString"])
    edges = frame.loc[line_mask].copy()
    if edges.empty:
        raise ValueError("Road graph file does not contain line geometries.")
    return edges


def _normalise_endpoint(point: Tuple[float, float], precision: int = 6) -> Tuple[float, float]:
    return (round(float(point[0]), precision), round(float(point[1]), precision))


def _edge_weight(row, geometry: LineString, default_speed_mps: float) -> float:
    for column in ("travel_time", "running_time", "time", "duration", "seconds", "cost"):
        if column in row and pd.notna(row[column]):
            return float(row[column])
    return _default_travel_time_seconds(geometry, default_speed_mps)


def build_road_graph(path: str | Path, default_speed_mps: float = 50.0 / 3.6) -> RoadGraph:
    source_path = Path(path)
    frame = gpd.read_file(source_path)
    if frame.crs is not None and getattr(frame.crs, "is_geographic", False):
        frame = frame.to_crs("EPSG:2056")
    edges = _extract_line_geometries(frame)

    graph = nx.Graph()
    node_positions: Dict[int, Tuple[float, float]] = {}
    coord_to_node: Dict[Tuple[float, float], int] = {}
    node_index = 0

    for _, row in edges.iterrows():
        geometry = row.geometry
        if isinstance(geometry, MultiLineString):
            geometries = list(geometry.geoms)
        else:
            geometries = [geometry]

        for line in geometries:
            if line.is_empty:
                continue
            start = _normalise_endpoint(line.coords[0])
            end = _normalise_endpoint(line.coords[-1])
            if start not in coord_to_node:
                coord_to_node[start] = node_index
                node_positions[node_index] = start
                graph.add_node(node_index, x=start[0], y=start[1])
                node_index += 1
            if end not in coord_to_node:
                coord_to_node[end] = node_index
                node_positions[node_index] = end
                graph.add_node(node_index, x=end[0], y=end[1])
                node_index += 1

            u = coord_to_node[start]
            v = coord_to_node[end]
            travel_time = _edge_weight(row, line, default_speed_mps)
            attrs = row.drop(labels=["geometry"]).to_dict()
            attrs.update({"travel_time": float(travel_time), "length": float(line.length)})
            graph.add_edge(u, v, **attrs)

    _GRAPH_REGISTRY[id(graph)] = graph
    return RoadGraph(graph=graph, node_positions=node_positions, crs=frame.crs, source_path=source_path)


@lru_cache(maxsize=8)
def _cached_node_lookup(node_positions_key: Tuple[Tuple[int, float, float], ...]):
    node_ids = np.array([item[0] for item in node_positions_key], dtype=int)
    xs = np.array([item[1] for item in node_positions_key], dtype=float)
    ys = np.array([item[2] for item in node_positions_key], dtype=float)
    return node_ids, xs, ys


def nearest_node_id(point: Point, road_graph: RoadGraph) -> int:
    node_ids = tuple((node_id, coord[0], coord[1]) for node_id, coord in road_graph.node_positions.items())
    cached_ids, xs, ys = _cached_node_lookup(node_ids)
    dx = xs - float(point.x)
    dy = ys - float(point.y)
    distances = dx * dx + dy * dy
    return int(cached_ids[int(np.argmin(distances))])


def assign_edge_times_to_grids(
    road_graph: RoadGraph,
    grid_gdf: pd.DataFrame,
) -> Dict[Tuple[int, int], int]:
    grid_centroids = grid_gdf.geometry.centroid
    centroid_x = grid_centroids.x.to_numpy(dtype=float)
    centroid_y = grid_centroids.y.to_numpy(dtype=float)
    grid_ids = grid_gdf["grid_id"].to_numpy(dtype=int)
    edges = list(road_graph.graph.edges())
    if not edges:
        return {}
    pos = road_graph.node_positions
    mid_x = np.array([(pos[u][0] + pos[v][0]) * 0.5 for u, v in edges], dtype=float)
    mid_y = np.array([(pos[u][1] + pos[v][1]) * 0.5 for u, v in edges], dtype=float)
    dx = mid_x[:, None] - centroid_x[None, :]  # (E, N_grids)
    dy = mid_y[:, None] - centroid_y[None, :]
    nearest_grid_indices = np.argmin(dx * dx + dy * dy, axis=1)
    return {(u, v): int(grid_ids[nearest_grid_indices[i]]) for i, (u, v) in enumerate(edges)}


@lru_cache(maxsize=250000)
def _cached_shortest_path(
    graph_id: int,
    origin_node: int,
    destination_node: int,
    edge_weight: str,
) -> Tuple[Tuple[int, ...], float]:
    graph = _GRAPH_REGISTRY.get(int(graph_id))
    if graph is None:
        return tuple(), float("inf")
    try:
        path = nx.shortest_path(graph, origin_node, destination_node, weight=edge_weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return tuple(), float("inf")
    travel_time = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge_data = graph.get_edge_data(u, v)
        if edge_data is None:
            continue
        if isinstance(edge_data, dict) and 0 in edge_data:
            edge_data = edge_data[0]
        travel_time += float(edge_data.get(edge_weight, edge_data.get("travel_time", 0.0)))
    return tuple(path), float(travel_time)


@lru_cache(maxsize=250000)
def _cached_edge_grid_times(
    graph_id: int,
    origin_node: int,
    destination_node: int,
    edge_weight: str,
    edge_to_grid_id: int,
) -> Tuple[Tuple[int, float], ...]:
    """Cache the per-grid time breakdown for a given route, so the path walk is done only once."""
    graph = _GRAPH_REGISTRY.get(graph_id)
    edge_to_grid = _EDGE_TO_GRID_REGISTRY.get(edge_to_grid_id)
    if graph is None or edge_to_grid is None:
        return tuple()
    path, _ = _cached_shortest_path(graph_id, origin_node, destination_node, edge_weight)
    if not path:
        return tuple()
    result: Dict[int, float] = {}
    for u, v in zip(path[:-1], path[1:]):
        edge_data = graph.get_edge_data(u, v)
        if edge_data is None:
            continue
        if isinstance(edge_data, dict) and 0 in edge_data:
            edge_data = edge_data[0]
        weight = float(edge_data.get(edge_weight, edge_data.get("travel_time", 0.0)))
        gid = edge_to_grid.get((u, v)) or edge_to_grid.get((v, u))
        if gid is not None:
            result[gid] = result.get(gid, 0.0) + weight
    return tuple(result.items())


def shortest_path_grid_times(
    road_graph: RoadGraph,
    origin_node: int,
    destination_node: int,
    edge_to_grid: Dict[Tuple[int, int], int],
    edge_weight: str = "travel_time",
) -> RouteResult:
    graph_id = id(road_graph.graph)
    _GRAPH_REGISTRY[graph_id] = road_graph.graph
    edge_to_grid_id = id(edge_to_grid)
    _EDGE_TO_GRID_REGISTRY[edge_to_grid_id] = edge_to_grid
    path, travel_time = _cached_shortest_path(graph_id, int(origin_node), int(destination_node), edge_weight)
    if not path:
        return RouteResult(path=[], travel_time=float("inf"), edge_grid_times={})
    edge_grid_items = _cached_edge_grid_times(graph_id, int(origin_node), int(destination_node), edge_weight, edge_to_grid_id)
    return RouteResult(path=list(path), travel_time=float(travel_time), edge_grid_times=dict(edge_grid_items))


@lru_cache(maxsize=8)
def load_road_graph(path: str | Path = "data/lausanne_roads_encoded.gpkg", default_speed_kmh: float = 50.0):
    return build_road_graph(resolve_path(path), default_speed_mps=float(default_speed_kmh) / 3.6)


@dataclass(frozen=True)
class RideRequest:
    request_id: int
    origin_grid_id: int
    destination_grid_id: int
    request_time: float


@dataclass(frozen=True)
class RideVehicleState:
    vehicle_id: int
    available_time: float
    current_node: int
    total_travel_time: float
    visited_grids: Tuple[int, ...]


@dataclass(frozen=True)
class RideHailingSimulationResult:
    composition: Tuple[int, ...]
    summary: MetricSummary
    coverage_rate: float
    population_coverage_rate: float
    wait_time_summary: MetricSummary
    service_rate: float
    per_grid_travel_time: pd.DataFrame


@dataclass(frozen=True)
class RideHailingContext:
    road_graph: RoadGraph
    grid_ids: np.ndarray
    centroid_x: np.ndarray
    centroid_y: np.ndarray
    residents: np.ndarray
    grid_to_node: Dict[int, int]
    edge_to_grid: Dict[Tuple[int, int], int]
    grid_id_to_index: Dict[int, int]   # O(1) grid_id -> array-index lookup
    nearby_indices: np.ndarray         # (N_grids, K) precomputed k-nearest grid indices
    nearby_distances: np.ndarray       # (N_grids, K) corresponding euclidean distances


def _grid_lookup_arrays(grid_gdf: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid_ids = grid_gdf["grid_id"].to_numpy(dtype=int)
    centroids = grid_gdf.geometry.centroid
    centroid_x = centroids.x.to_numpy(dtype=float)
    centroid_y = centroids.y.to_numpy(dtype=float)
    if "residents" in grid_gdf.columns:
        residents = grid_gdf["residents"].to_numpy(dtype=float)
    else:
        residents = np.ones(len(grid_gdf), dtype=float)
    return grid_ids, centroid_x, centroid_y, residents


def build_ridehailing_context(road_graph: RoadGraph, grid_gdf: pd.DataFrame, nearby_k: int = 12) -> RideHailingContext:
    working_grid_gdf = grid_gdf
    if getattr(grid_gdf, "crs", None) is not None and road_graph.crs is not None and grid_gdf.crs != road_graph.crs:
        working_grid_gdf = grid_gdf.to_crs(road_graph.crs)
    grid_ids, centroid_x, centroid_y, residents = _grid_lookup_arrays(working_grid_gdf)

    # Vectorized nearest-node lookup: avoids iterrows + per-cell argmin
    node_ids_arr = np.array(list(road_graph.node_positions.keys()), dtype=int)
    node_x = np.array([road_graph.node_positions[n][0] for n in node_ids_arr], dtype=float)
    node_y = np.array([road_graph.node_positions[n][1] for n in node_ids_arr], dtype=float)
    dx_cn = centroid_x[:, None] - node_x[None, :]  # (N_grids, N_nodes)
    dy_cn = centroid_y[:, None] - node_y[None, :]
    nearest_node_idx = np.argmin(dx_cn * dx_cn + dy_cn * dy_cn, axis=1)
    grid_to_node = {int(grid_ids[i]): int(node_ids_arr[nearest_node_idx[i]]) for i in range(len(grid_ids))}

    edge_to_grid = assign_edge_times_to_grids(road_graph, working_grid_gdf)
    grid_id_to_index = {int(gid): i for i, gid in enumerate(grid_ids)}

    # Precompute k-nearest grid neighbors in chunks to bound memory usage
    n_grids = len(grid_ids)
    k = min(nearby_k, n_grids)
    nearby_indices_arr = np.empty((n_grids, k), dtype=np.int32)
    nearby_distances_arr = np.empty((n_grids, k), dtype=np.float32)
    chunk_size = min(512, n_grids)
    for start in range(0, n_grids, chunk_size):
        end = min(start + chunk_size, n_grids)
        dx_gg = centroid_x[start:end, None] - centroid_x[None, :]  # (chunk, N_grids)
        dy_gg = centroid_y[start:end, None] - centroid_y[None, :]
        dist_sq = dx_gg * dx_gg + dy_gg * dy_gg
        # argpartition gives top-k without full sort
        part = np.argpartition(dist_sq, min(k, n_grids - 1), axis=1)[:, :k]
        part_dist_sq = np.take_along_axis(dist_sq, part, axis=1)
        sort_order = np.argsort(part_dist_sq, axis=1)
        nearby_indices_arr[start:end] = np.take_along_axis(part, sort_order, axis=1)
        nearby_distances_arr[start:end] = np.sqrt(np.take_along_axis(part_dist_sq, sort_order, axis=1))

    return RideHailingContext(
        road_graph=road_graph,
        grid_ids=grid_ids,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        residents=residents,
        grid_to_node=grid_to_node,
        edge_to_grid=edge_to_grid,
        grid_id_to_index=grid_id_to_index,
        nearby_indices=nearby_indices_arr,
        nearby_distances=nearby_distances_arr,
    )


def _sample_weighted_grid(grid_ids: np.ndarray, weights: np.ndarray, rng: np.random.Generator) -> int:
    safe_weights = np.asarray(weights, dtype=float)
    safe_weights = np.where(np.isfinite(safe_weights) & (safe_weights > 0), safe_weights, 0.0)
    if safe_weights.sum() <= 0:
        safe_weights = np.ones_like(safe_weights, dtype=float)
    safe_weights = safe_weights / safe_weights.sum()
    return int(rng.choice(grid_ids, p=safe_weights))


def _sample_nearby_origin_grid(
    current_grid_id: int,
    context: RideHailingContext,
    rng: np.random.Generator,
) -> int:
    current_index = context.grid_id_to_index.get(int(current_grid_id))
    if current_index is None:
        return _sample_weighted_grid(context.grid_ids, context.residents, rng)
    candidate_indices = context.nearby_indices[current_index]          # shape (K,)
    candidate_distances = context.nearby_distances[current_index].astype(float)
    median_dist = max(float(np.median(candidate_distances)), 1.0)
    local_weights = np.maximum(context.residents[candidate_indices], 1.0) * np.exp(-candidate_distances / median_dist)
    return _sample_weighted_grid(context.grid_ids[candidate_indices], local_weights, rng)


def _sample_destination_grid(
    origin_grid_id: int,
    grid_ids: np.ndarray,
    residents: np.ndarray,
    rng: np.random.Generator,
) -> int:
    destination_weights = np.maximum(residents, 1.0).astype(float)
    origin_matches = np.where(grid_ids == int(origin_grid_id))[0]
    if origin_matches.size:
        destination_weights[int(origin_matches[0])] = 0.0
    return _sample_weighted_grid(grid_ids, destination_weights, rng)


def _simulate_vehicle_sequence(
    vehicle_id: int,
    context: RideHailingContext,
    labor_time_limit_seconds: float,
    break_min_minutes: float,
    break_max_minutes: float,
    rng: np.random.Generator,
) -> Tuple[List[float], Dict[int, float], int, int]:
    current_grid_id = _sample_weighted_grid(context.grid_ids, context.residents, rng)
    current_node = int(context.grid_to_node[current_grid_id])
    current_time = 0.0
    wait_times: List[float] = []
    per_grid_time: Dict[int, float] = {}
    served = 0
    attempted = 0

    while current_time < labor_time_limit_seconds:
        attempted += 1
        origin_grid_id = _sample_nearby_origin_grid(current_grid_id=current_grid_id, context=context, rng=rng)
        destination_grid_id = _sample_destination_grid(origin_grid_id, context.grid_ids, context.residents, rng)
        pickup_node = int(context.grid_to_node[origin_grid_id])
        dropoff_node = int(context.grid_to_node[destination_grid_id])

        travel_to_pickup_result = shortest_path_grid_times(context.road_graph, current_node, pickup_node, context.edge_to_grid)
        pickup_travel_time = float(travel_to_pickup_result.travel_time)
        if not np.isfinite(pickup_travel_time):
            break
        if current_time + pickup_travel_time > labor_time_limit_seconds:
            break

        ride_path_result = shortest_path_grid_times(context.road_graph, pickup_node, dropoff_node, context.edge_to_grid)
        ride_travel_time = float(ride_path_result.travel_time)
        if not np.isfinite(ride_travel_time):
            break
        break_seconds = float(rng.uniform(break_min_minutes * 60.0, break_max_minutes * 60.0))
        if current_time + pickup_travel_time + ride_travel_time + break_seconds > labor_time_limit_seconds:
            break

        wait_times.append(pickup_travel_time)
        served += 1
        current_time += pickup_travel_time + ride_travel_time + break_seconds
        current_node = dropoff_node
        current_grid_id = destination_grid_id

        for grid_id, time_value in travel_to_pickup_result.edge_grid_times.items():
            per_grid_time[grid_id] = per_grid_time.get(grid_id, 0.0) + time_value
        for grid_id, time_value in ride_path_result.edge_grid_times.items():
            per_grid_time[grid_id] = per_grid_time.get(grid_id, 0.0) + time_value

    return wait_times, per_grid_time, served, attempted


def _vehicle_states(vehicle_ids: Sequence[int], road_graph: RoadGraph, grid_gdf: pd.DataFrame) -> Dict[int, RideVehicleState]:
    states: Dict[int, RideVehicleState] = {}
    start_node = next(iter(road_graph.graph.nodes))
    for vehicle_id in vehicle_ids:
        states[int(vehicle_id)] = RideVehicleState(
            vehicle_id=int(vehicle_id),
            available_time=0.0,
            current_node=int(start_node),
            total_travel_time=0.0,
            visited_grids=tuple(),
        )
    return states


def simulate_ridehailing_composition(
    vehicle_ids: Sequence[int],
    road_graph: RoadGraph | None = None,
    grid_gdf: pd.DataFrame | None = None,
    context: RideHailingContext | None = None,
    labor_time_limit_hours: float = 5.0,
    break_min_minutes: float = 0.0,
    break_max_minutes: float = 10.0,
    seed: int = 42,
    progress_callback: Optional[ProgressCallback] = None,
    progress_offset: int = 0,
    progress_total: int = 0,
    progress_label: str = "Ride-hailing",
) -> RideHailingSimulationResult:
    rng = np.random.default_rng(seed)
    if context is None:
        if road_graph is None or grid_gdf is None:
            raise ValueError("Either context or both road_graph and grid_gdf must be provided.")
        context = build_ridehailing_context(road_graph, grid_gdf)
    labor_time_limit_seconds = labor_time_limit_hours * 3600.0

    wait_times: List[float] = []
    served = 0
    attempted = 0
    per_grid_time: Dict[int, float] = {}

    if progress_callback:
        progress_callback(progress_offset, progress_total, f"{progress_label}: initializing {len(vehicle_ids)} vehicles")

    for vehicle_index, vehicle_id in enumerate(vehicle_ids, start=1):
        if progress_callback:
            progress_callback(
                progress_offset,
                progress_total,
                f"{progress_label}: vehicle {vehicle_index}/{len(vehicle_ids)} starting",
            )
        vehicle_wait_times, vehicle_per_grid_time, vehicle_served, vehicle_attempted = _simulate_vehicle_sequence(
            vehicle_id=int(vehicle_id),
            context=context,
            labor_time_limit_seconds=labor_time_limit_seconds,
            break_min_minutes=break_min_minutes,
            break_max_minutes=break_max_minutes,
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        )
        wait_times.extend(vehicle_wait_times)
        served += vehicle_served
        attempted += vehicle_attempted
        for grid_id, time_value in vehicle_per_grid_time.items():
            per_grid_time[grid_id] = per_grid_time.get(grid_id, 0.0) + time_value

        if progress_callback:
            progress_callback(
                progress_offset,
                progress_total,
                f"{progress_label}: vehicle {vehicle_index}/{len(vehicle_ids)} completed, served {vehicle_served} requests",
            )

    if per_grid_time:
        per_grid_df = pd.DataFrame({"grid_id": list(per_grid_time.keys()), "mean_running_time": list(per_grid_time.values())})
    else:
        per_grid_df = pd.DataFrame(columns=["grid_id", "mean_running_time"])

    coverage_rate = float(per_grid_df["grid_id"].nunique() / max(len(grid_gdf), 1))
    if "residents" in grid_gdf.columns and not per_grid_df.empty:
        visited = per_grid_df.merge(grid_gdf[["grid_id", "residents"]], on="grid_id", how="left").fillna({"residents": 0})
        population_coverage_rate = float(visited["residents"].sum() / max(grid_gdf["residents"].sum(), 1.0))
    else:
        population_coverage_rate = coverage_rate

    utility_values = per_grid_df["mean_running_time"].to_numpy(dtype=float) if not per_grid_df.empty else np.array([0.0])
    summary = summarize_metrics(utility_values)
    wait_summary = summarize_metrics(wait_times)
    service_rate = float(served / max(attempted, 1))

    return RideHailingSimulationResult(
        composition=tuple(vehicle_ids),
        summary=summary,
        coverage_rate=coverage_rate,
        population_coverage_rate=population_coverage_rate,
        wait_time_summary=wait_summary,
        service_rate=service_rate,
        per_grid_travel_time=per_grid_df,
    )
