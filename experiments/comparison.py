#!/usr/bin/env python3
"""VCSP Experiments: Compare NO-S vs MILP-S."""

import time
import numpy as np
from core.column_generation import VCSPSolver
from problems.vcsp.instance import VCSPInstance


def run_instance(num_trips, seed, selection, config=None):
    """Run a single CG instance with a given selection strategy."""
    instance = VCSPInstance(num_trips=num_trips, seed=seed)
    solver = VCSPSolver(instance, config)

    start = time.time()
    columns, col_sol, bus_sol, obj = solver.solve(
        selection_strategy=selection,
        max_iterations=300,
    )
    elapsed = time.time() - start

    return {
        'num_trips': num_trips,
        'seed': seed,
        'selection': selection,
        'total_time': elapsed,
        'rmp_time': solver.stats['rmp_solve_time'],
        'pp_time': solver.stats['pp_solve_time'],
        'sel_time': solver.stats['selection_time'],
        'iterations': solver.stats['iterations'],
        'final_columns': len(columns),
        'objective': obj,
        'buses': bus_sol,
        'columns_generated': sum(solver.stats['columns_generated']),
        'columns_selected': sum(solver.stats['columns_selected']),
    }


def run_comparison_experiment(trip_sizes=(20, 30, 50), num_instances=3):
    """Run comparison experiments across multiple instance sizes."""
    config = {
        'n_min_cols': 100,
        'n_max_cols': 5000,
        'n_max_blks': 10,
        'epsilon': 0.1,
        'additional_pct': 0.5,
    }

    results = []
    for num_trips in trip_sizes:
        for i in range(num_instances):
            seed = 100 * num_trips + i
            print(f"\n--- Instance: {num_trips} trips, seed={seed} ---")

            # NO-S
            print("\n[NO-S]")
            res_no = run_instance(num_trips, seed, 'no_selection', config)
            results.append(res_no)

            # MILP-S
            print("\n[MILP-S]")
            res_milp = run_instance(num_trips, seed, 'milp', config)
            results.append(res_milp)

            # Print comparison
            print(f"\n  Comparison (NO-S vs MILP-S):")
            print(f"    Time: {res_no['total_time']:.1f}s vs {res_milp['total_time']:.1f}s "
                  f"({(res_no['total_time'] - res_milp['total_time']) / res_no['total_time'] * 100:.0f}% reduction)")
            print(f"    Iterations: {res_no['iterations']} vs {res_milp['iterations']}")
            print(f"    Final columns: {res_no['final_columns']} vs {res_milp['final_columns']}")
            print(f"    Total cols generated: {res_no['columns_generated']} vs {res_milp['columns_generated']}")

    return results


def print_summary_table(results):
    """Print results formatted as a table matching the paper's format."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)

    # Group by trip size and selection
    grouped = {}
    for r in results:
        key = (r['num_trips'], r['selection'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    header = f"{'Trips':>6} {'Strategy':>12} {'Time':>8} {'RMP':>8} {'PP':>8} {'Iters':>6} {'Cols':>8} {'Obj':>10} {'Buses':>6}"
    print(header)
    print("-" * 100)

    for (num_trips, sel) in sorted(grouped.keys()):
        runs = grouped[(num_trips, sel)]
        avg_time = np.mean([r['total_time'] for r in runs])
        avg_rmp = np.mean([r['rmp_time'] for r in runs])
        avg_pp = np.mean([r['pp_time'] for r in runs])
        avg_iters = np.mean([r['iterations'] for r in runs])
        avg_cols = np.mean([r['final_columns'] for r in runs])
        avg_obj = np.mean([r['objective'] for r in runs])
        avg_buses = np.mean([r['buses'] for r in runs])

        line = f"{num_trips:>6} {sel:>12} {avg_time:>8.1f} {avg_rmp:>8.1f} {avg_pp:>8.1f} {avg_iters:>6.0f} {avg_cols:>8.0f} {avg_obj:>10.2f} {avg_buses:>6.1f}"
        print(line)

    # Also compute average reduction
    no_data = {}
    milp_data = {}
    for r in results:
        if r['selection'] == 'no_selection':
            no_data[r['num_trips']] = no_data.get(r['num_trips'], []) + [r]
        else:
            milp_data[r['num_trips']] = milp_data.get(r['num_trips'], []) + [r]

    print("\n" + "-" * 100)
    print("AVERAGE TIME REDUCTION")
    for n in sorted(no_data.keys()):
        no_avg = np.mean([r['total_time'] for r in no_data[n]])
        milp_avg = np.mean([r['total_time'] for r in milp_data.get(n, [])])
        if no_avg > 0:
            reduction = (no_avg - milp_avg) / no_avg * 100
        else:
            reduction = 0
        print(f"  {n} trips: {reduction:.1f}% time reduction")


if __name__ == '__main__':
    trip_sizes = (20, 30)
    num_instances = 2

    print("=" * 60)
    print("VCSP EXPERIMENTS: NO-S vs MILP-S")
    print(f"Trip sizes: {trip_sizes}, Instances per size: {num_instances}")
    print("=" * 60)

    results = run_comparison_experiment(trip_sizes, num_instances)
    print_summary_table(results)
