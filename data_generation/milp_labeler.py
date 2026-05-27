"""MILP labeler for column selection.

Solves the exact MILP from EBSCO paper Section 3.1.1 to produce ground-truth
labels for column classification:

    Min  sum(c_p * theta_p) + sum(epsilon * y_p)
    s.t.
      All RMP constraints (with B variable)
      theta_p <= y_p,  for all p in generated columns
      theta_p >= 0, B >= 0, y_p in {0, 1}

Returns:
  - Selected columns (y_p = 1)
  - Label vector for all new columns (0 or 1)
  - Additionally, 50% of remaining negative RC columns for convergence
"""

import numpy as np
from ortools.linear_solver import pywraplp


class MilpLabeler:
    """Solve the MILP column selection problem and return labels."""

    def __init__(self, instance, epsilon=0.1, additional_pct=0.5):
        self.instance = instance
        self.epsilon = epsilon
        self.additional_pct = additional_pct

    def label(self, rmp_columns, new_columns):
        """Solve the MILP and return (selected_columns, labels, labels_dict).

        Args:
            rmp_columns: list of VCSPColumn currently in RMP
            new_columns: list of newly generated VCSPColumn

        Returns:
            selected_columns: list of columns to add to RMP (y_p=1 + 50% extra)
            labels: np.array of shape (len(new_columns),) with 0/1 values
            theta_new: np.array of theta values for new columns from MILP solution
        """
        num_existing = len(rmp_columns)
        num_new = len(new_columns)
        num_total = num_existing + num_new

        inst = self.instance
        num_d_trips = inst.num_d_trips
        num_trips = inst.num_trips
        num_departure = len(inst.departure_times)
        num_eq = num_d_trips + 2 * num_trips
        num_ub = num_departure

        # Create MILP solver
        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            # No MILP solver available: use all columns, label top 10% as positive
            print("  [MilpLabeler] WARNING: No MILP solver (CBC/SCIP). Using heuristic labels.")
            return self._fallback_labels(rmp_columns, new_columns)

        # Variables
        theta_vars = [solver.NumVar(0.0, solver.infinity(), f'theta_{i}') for i in range(num_total)]
        B_var = solver.NumVar(0.0, solver.infinity(), 'B')
        y_vars = [solver.IntVar(0, 1, f'y_{i}') for i in range(num_new)]

        # Objective: sum(c_p * theta_p) + sum(epsilon * y_p)
        objective = solver.Objective()
        for i, col in enumerate(rmp_columns + new_columns):
            objective.SetCoefficient(theta_vars[i], col.cost)
        objective.SetCoefficient(B_var, inst.bus_fixed_cost)
        for i in range(num_new):
            objective.SetCoefficient(y_vars[i], self.epsilon)
        objective.SetMinimization()

        # Build coefficient vectors for all columns
        col_vectors = []
        for col in rmp_columns + new_columns:
            vec = np.zeros(num_eq + num_ub)
            for d_id in col.d_trips:
                vec[d_id] = 1.0
            for w_id in col.f_trips:
                vec[num_d_trips + w_id] = 1.0
            for w_id in col.g_trips:
                vec[num_d_trips + num_trips + w_id] = 1.0
            for h_idx, h in enumerate(inst.departure_times):
                if col.q_times.get(h, False):
                    vec[num_eq + h_idx] = 1.0
            col_vectors.append(vec)

        # 1. Equality constraints (d-trip covering, bus arrival, bus departure)
        for c_idx in range(num_eq):
            ct = solver.Constraint(1.0, 1.0)
            for i in range(num_total):
                if col_vectors[i][c_idx] != 0:
                    ct.SetCoefficient(theta_vars[i], col_vectors[i][c_idx])

        # 2. Bus count constraints: sum(q) - B <= 0
        for h_idx in range(num_ub):
            c_idx = num_eq + h_idx
            ct = solver.Constraint(-solver.infinity(), 0.0)
            for i in range(num_total):
                if col_vectors[i][c_idx] != 0:
                    ct.SetCoefficient(theta_vars[i], col_vectors[i][c_idx])
            ct.SetCoefficient(B_var, -1.0)

        # 3. Linking: theta_p <= y_p for new columns
        for i in range(num_new):
            idx = num_existing + i
            ct = solver.Constraint(-solver.infinity(), 0.0)
            ct.SetCoefficient(theta_vars[idx], 1.0)
            ct.SetCoefficient(y_vars[i], -1.0)

        # Solve
        status = solver.Solve()

        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            print(f"  [MilpLabeler] Solve status {status}, using heuristic labels.")
            return self._fallback_labels(rmp_columns, new_columns)

        # Extract labels (y_p values)
        labels = np.zeros(num_new, dtype=np.int32)
        selected_indices = []
        for i in range(num_new):
            if y_vars[i].solution_value() > 0.5:
                labels[i] = 1
                selected_indices.append(i)

        # Extract theta values for new columns (for analysis, not for features)
        theta_new = np.array([theta_vars[num_existing + i].solution_value() for i in range(num_new)])

        selected_cols = [new_columns[i] for i in selected_indices]

        # Add 50% of remaining negative reduced cost columns (paper strategy for convergence)
        unselected = [(i, new_columns[i]) for i in range(num_new) if i not in selected_indices]
        unselected.sort(key=lambda x: x[1].reduced_cost)
        n_additional = int(len(unselected) * self.additional_pct)
        additional = [col for _, col in unselected[:n_additional]]

        result = selected_cols + additional

        if len(result) < 1 and len(new_columns) > 0:
            result = [new_columns[0]]

        return result, labels, theta_new

    def _fallback_labels(self, rmp_columns, new_columns):
        """Heuristic: label top 10% by reduced cost as positive."""
        num_new = len(new_columns)
        if num_new == 0:
            return [], np.array([], dtype=np.int32), np.array([], dtype=np.float64)

        sorted_idx = sorted(range(num_new), key=lambda i: new_columns[i].reduced_cost)
        n_select = max(1, int(num_new * 0.1))
        labels = np.zeros(num_new, dtype=np.int32)
        labels[sorted_idx[:n_select]] = 1

        selected = [new_columns[i] for i in sorted_idx[:n_select]]
        # Also add 50% of remaining
        unselected = [new_columns[i] for i in sorted_idx[n_select:]]
        n_add = int(len(unselected) * self.additional_pct)
        additional = unselected[:n_add]

        theta_new = np.zeros(num_new)
        return selected + additional, labels, theta_new
