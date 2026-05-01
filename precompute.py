"""Precompute and save Pareto simulation results.

Focuses on near-balanced compositions (|n_postal - n_ridehailing| <= BALANCE_DELTA)
across a wider fleet size range to produce more frontier diversity.

Run this locally (not on Streamlit Cloud) before pushing:

    python precompute.py

Outputs written to data/:
    data/precomputed_results.pkl   — all_results DataFrame + frontier DataFrame
                                     + stats_map Dict[(n_postal, n_ridehailing), SingleCompositionStats]

The dashboard will load this file directly instead of running any simulation.
Re-run any time you want to refresh the results with different parameters.
"""
from __future__ import annotations

from sim.simulation import run_budget_range_pareto, save_precomputed_results
from pathlib import Path

TOTAL_MIN = 15
TOTAL_MAX = 20
M_RUNS = 200
SEED = 43
# Start from the equal split (n_postal == n_ridehailing) and allow up to
# BALANCE_DELTA steps shifting vehicles from one type to the other.
BALANCE_DELTA = 18
OUT_PATH = Path("data/precomputed_results.pkl")


def main() -> None:
    print(f"Running Pareto simulation for fleet sizes {TOTAL_MIN}–{TOTAL_MAX}, balance_delta={BALANCE_DELTA}, {M_RUNS} MC runs each…")

    done_steps: list[int] = [0]

    def _progress(done: int, total: int, message: str) -> None:
        if done != done_steps[0]:
            done_steps[0] = done
            pct = int(100 * done / total) if total > 0 else 100
            print(f"  [{pct:3d}%] {message}")

    all_results, frontier, stats_map = run_budget_range_pareto(
        total_vehicle_min=TOTAL_MIN,
        total_vehicle_max=TOTAL_MAX,
        m_runs=M_RUNS,
        seed=SEED,
        balance_delta=BALANCE_DELTA,
        progress_callback=_progress,
        max_workers=None,  # use all cores locally
    )

    save_precomputed_results(
        all_results=all_results,
        frontier=frontier,
        stats_map=stats_map,
        path=OUT_PATH,
        metadata={"total_min": TOTAL_MIN, "total_max": TOTAL_MAX, "m_runs": M_RUNS, "seed": SEED, "balance_delta": BALANCE_DELTA},
    )

    print(f"\nSaved → {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"Compositions: {len(all_results)}, Frontier points: {len(frontier)}")


if __name__ == "__main__":
    main()
