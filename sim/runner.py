from __future__ import annotations

import argparse

from .simulation import run_budget_range_pareto, run_single_composition_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fleet composition simulation demo")
    parser.add_argument("--roads", default="data/lausanne_roads_encoded.gpkg")
    parser.add_argument("--grid", default="data/grid_100m.gpkg")
    parser.add_argument("--population", default="data/swiss_population.csv")
    parser.add_argument("--n-postal", type=int, default=2)
    parser.add_argument("--n-ridehailing", type=int, default=2)
    parser.add_argument("--m-runs", type=int, default=100)
    parser.add_argument("--total-min", type=int, default=3)
    parser.add_argument("--total-max", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    single_stats = run_single_composition_simulation(
        n_postal=args.n_postal,
        n_ridehailing=args.n_ridehailing,
        m_runs=args.m_runs,
        roads_path=args.roads,
        grid_path=args.grid,
        population_path=args.population,
        seed=args.seed,
    )
    print("Single composition stats:")
    print(
        f"composition=({single_stats.n_postal}, {single_stats.n_ridehailing}), "
        f"expected_utility={single_stats.expected_utility:.4f}, cv={single_stats.cv:.4f}"
    )
    print("Mean travel-time matrix (head):")
    print(single_stats.mean_travel_time_matrix.head())

    all_results, frontier, _ = run_budget_range_pareto(
        total_vehicle_min=args.total_min,
        total_vehicle_max=args.total_max,
        m_runs=args.m_runs,
        roads_path=args.roads,
        grid_path=args.grid,
        population_path=args.population,
        seed=args.seed,
    )
    print("All composition results (head):")
    print(all_results.head())
    print("Pareto frontier (head):")
    print(frontier.head())


if __name__ == "__main__":
    main()
