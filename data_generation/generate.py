#!/usr/bin/env python3
"""Generate training data for GNN column selection.

Solves multiple random VCSP instances using MILP column selection,
recording bipartite graph data (features + labels) at each CG iteration.

Usage:
    python -m data_generation.generate                     # default: 20 inst, 50 trips
    python -m data_generation.generate --trips 100 --instances 50
    python -m data_generation.generate --trips 200 --instances 30 --output data/vcsp_200

Paper reference: EBSCO Section 4.3 — 100 instances of 400 trips for VCSP training.
"""

import os
import sys
import time
import argparse
import numpy as np

# Ensure project root is in path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from problems.vcsp.instance import VCSPInstance
from data_generation.data_collector import DataCollector


def generate_single_instance(instance_id, num_trips, seed, config, output_base_dir):
    """Generate data from a single VCSP instance.

    Returns:
        num_data_points: int, number of iterations recorded
        elapsed: float, wall-clock time
        success: bool
    """
    output_dir = os.path.join(output_base_dir, f'instance_{instance_id:04d}')

    # Skip if already done
    metadata_path = os.path.join(output_dir, 'metadata.npz')
    if os.path.exists(metadata_path):
        try:
            meta = np.load(metadata_path)
            n = int(meta['num_data_points'])
            print(f"  [SKIP] Instance {instance_id} already exists ({n} data points)")
            return n, 0.0, True
        except Exception:
            pass  # Corrupted, regenerate

    print(f"\n{'#' * 60}")
    print(f"# Instance {instance_id}: {num_trips} trips, seed={seed}")
    print(f"{'#' * 60}")

    try:
        instance = VCSPInstance(num_trips=num_trips, seed=seed)

        collector = DataCollector(instance, config)
        collector.collect(max_iterations=config.get('max_iterations', 300))

        n_points = collector.save(output_dir)
        elapsed = collector.stats['total_time']

        return n_points, elapsed, True

    except Exception as e:
        print(f"  [FAIL] Instance {instance_id}: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0.0, False


def main():
    parser = argparse.ArgumentParser(
        description='Generate VCSP training data for GNN column selection'
    )
    parser.add_argument('--trips', type=int, default=50,
                        help='Number of trips per instance (default: 50)')
    parser.add_argument('--instances', type=int, default=20,
                        help='Number of instances to generate (default: 20)')
    parser.add_argument('--start-id', type=int, default=0,
                        help='Starting instance ID (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: data_generation/training_data/vcsp_{trips})')
    parser.add_argument('--max-iterations', type=int, default=300,
                        help='Max CG iterations per instance (default: 300)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='MILP penalty epsilon (default: 0.1)')
    parser.add_argument('--additional-pct', type=float, default=0.5,
                        help='Additional column percentage (default: 0.5)')
    parser.add_argument('--skip-first', type=int, default=0,
                        help='Skip first N iterations of each instance (default: 0)')
    parser.add_argument('--min-generated', type=int, default=5,
                        help='Min new columns to record an iteration (default: 5)')
    parser.add_argument('--cost-inflation', type=float, default=3.0,
                        help='Cost inflation factor for initial columns (default: 3.0)')
    parser.add_argument('--artificial-cost', type=float, default=1e7,
                        help='Cost for artificial f/g trip columns (default: 1e7)')
    parser.add_argument('--seed-base', type=int, default=42,
                        help='Base seed for instance generation (default: 42)')

    args = parser.parse_args()

    # Output directory
    if args.output is None:
        output_base_dir = os.path.join(
            _project_root, 'data_generation', 'training_data', f'vcsp_{args.trips}'
        )
    else:
        output_base_dir = args.output

    os.makedirs(output_base_dir, exist_ok=True)

    # Config for DataCollector
    config = {
        'epsilon': args.epsilon,
        'additional_pct': args.additional_pct,
        'max_iterations': args.max_iterations,
        'min_generated_for_record': args.min_generated,
        'skip_first_n_iterations': args.skip_first,
        'cost_inflation_factor': args.cost_inflation,
        'artificial_cost': args.artificial_cost,
    }

    print("=" * 60)
    print("VCSP TRAINING DATA GENERATION")
    print("=" * 60)
    print(f"Trips per instance: {args.trips}")
    print(f"Number of instances: {args.instances}")
    print(f"Output directory: {output_base_dir}")
    print(f"Config: {config}")
    print(f"Estimated constraints per instance: ~{args.trips * 2 + 2 * args.trips} eq + bus count")
    print("=" * 60)

    total_data_points = 0
    total_time = 0.0
    success_count = 0

    start_all = time.time()

    for i in range(args.start_id, args.start_id + args.instances):
        seed = args.seed_base * 1000 + i * 137 + args.trips
        n_points, elapsed, ok = generate_single_instance(
            i, args.trips, seed, config, output_base_dir
        )
        if ok:
            total_data_points += n_points
            total_time += elapsed
            success_count += 1

    total_wall = time.time() - start_all

    # Print overall summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Instances: {success_count}/{args.instances} succeeded")
    print(f"Total data points: {total_data_points}")
    print(f"Total compute time: {total_time:.1f}s (wall: {total_wall:.1f}s)")
    print(f"Avg per instance: {total_time / max(1, success_count):.1f}s")
    if total_data_points > 0:
        print(f"Avg data points per instance: {total_data_points / max(1, success_count):.1f}")
    print(f"Output: {output_base_dir}")

    # Estimate: paper reports ~7,000 data points from 100 instances of 400 trips
    if total_data_points > 0:
        target = 7000
        print(f"\nPaper reference: ~7,000 data points from 100 instances of 400 trips")
        print(f"Current: {total_data_points} data points from {success_count} instances of {args.trips} trips")
        if total_data_points < target:
            est_instances = int(success_count * target / total_data_points)
            print(f"Estimated instances needed for ~{target} points: ~{est_instances}")


if __name__ == '__main__':
    main()
