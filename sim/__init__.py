"""Fleet composition simulation package."""

from .simulation import ParetoProtocol, evaluate_composition_result, generate_compositions_for_total, pareto_frontier
from .simulation import SingleCompositionStats, run_budget_range_pareto, run_single_composition_simulation
from .simulation import RideHailingPoolMatrix, build_ridehailing_pool_matrix, save_ridehailing_pool_matrix, load_ridehailing_pool_matrix
from .postal_sim import PostalSimulationResult, evaluate_postal_composition, enumerate_combinations
from .postal_sim import VehicleGridMatrix, build_vehicle_grid_matrix, get_vehicle_grid_matrix
from .ridehailing_sim import RideHailingSimulationResult, simulate_ridehailing_composition
from .utility import MetricSummary, default_sensing_utility, summarize_metrics

__all__ = [
    "evaluate_postal_composition",
    "enumerate_combinations",
    "VehicleGridMatrix",
    "build_vehicle_grid_matrix",
    "get_vehicle_grid_matrix",
    "RideHailingPoolMatrix",
    "build_ridehailing_pool_matrix",
    "save_ridehailing_pool_matrix",
    "load_ridehailing_pool_matrix",
    "ParetoProtocol",
    "evaluate_composition_result",
    "generate_compositions_for_total",
    "SingleCompositionStats",
    "pareto_frontier",
    "run_budget_range_pareto",
    "run_single_composition_simulation",
    "PostalSimulationResult",
    "RideHailingSimulationResult",
    "simulate_ridehailing_composition",
    "MetricSummary",
    "default_sensing_utility",
    "summarize_metrics",
]
