#!/usr/bin/env python3
"""VCSP Column Generation - Main Entry Point."""

import argparse
import time
from problems.vcsp.instance import VCSPInstance
from core.column_generation import VCSPSolver


def main():
    parser = argparse.ArgumentParser(description='VCSP Column Generation')
    parser.add_argument('--trips', type=int, default=30, help='Number of trips')
    parser.add_argument('--relief', type=int, default=2, help='Relief points per trip')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--selection', type=str, default='no_selection',
                        choices=['no_selection', 'milp'],
                        help='Column selection strategy')
    parser.add_argument('--max-iter', type=int, default=200, help='Max CG iterations')
    parser.add_argument('--bus-cost', type=float, default=50000, help='Bus fixed cost')
    parser.add_argument('--driver-cost', type=float, default=50000, help='Driver fixed cost')
    parser.add_argument('--cost-per-min', type=float, default=1.0, help='Cost per minute')

    args = parser.parse_args()

    # Create instance
    print("Creating VCSP instance...")
    instance = VCSPInstance(
        num_trips=args.trips,
        num_relief_points_per_trip=args.relief,
        seed=args.seed,
        bus_fixed_cost=args.bus_cost,
        driver_fixed_cost=args.driver_cost,
        cost_per_minute=args.cost_per_min,
    )

    print(f"Instance: {instance.num_trips} trips, {instance.num_d_trips} d-trips")
    print(f"Departure times: {instance.get_num_departure_times()}")

    # Configure
    config = {
        'n_min_cols': 100,
        'n_max_cols': 5000,
        'n_max_blks': 10,
        'epsilon': 0.1,
        'additional_pct': 0.5,
    }

    # Create solver
    solver = VCSPSolver(instance, config)

    # Solve
    start = time.time()
    columns, col_sol, bus_sol, obj = solver.solve(
        selection_strategy=args.selection,
        max_iterations=args.max_iter,
    )
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print("RESULT")
    print(f"{'=' * 60}")
    print(f"Selection strategy: {args.selection}")
    print(f"Trips: {args.trips}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Iterations: {solver.stats['iterations']}")
    print(f"Final columns in RMP: {len(columns)}")
    print(f"Objective: {obj:.4f}")
    print(f"Buses: {bus_sol:.2f}")

    # Count positive theta variables
    n_positive = sum(1 for v in col_sol if v > 1e-6) if col_sol else 0
    print(f"Positive duties: {n_positive}")


if __name__ == '__main__':
    main()
