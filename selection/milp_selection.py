import numpy as np
from ortools.linear_solver import pywraplp


class MILPSelectionCG:
    """MILP-S: Column selection by solving a MILP.

    Per iteration, solves:
    Min  sum(c_p * theta_p) + c*B + sum(epsilon * y_p)
    s.t.
      All current RMP constraints (with B variable)
      theta_p <= y_p, for all p in generated columns
      theta_p >= 0, B >= 0, y_p in {0, 1}

    Then adds: selected columns (y_p=1) + 50% of remaining negative reduced cost columns.
    """

    def __init__(self, rmp, pp, config=None):
        self.rmp = rmp
        self.pp = pp
        self.config = config or {}
        self.epsilon = self.config.get('epsilon', 0.1)
        self.additional_pct = self.config.get('additional_pct', 0.5)
        self.min_select = self.config.get('min_select', 1)

    def select(self, columns):
        """Select columns using MILP."""
        if not columns:
            return []

        num_existing = len(self.rmp.columns)
        num_new = len(columns)
        num_total = num_existing + num_new

        inst = self.rmp.instance
        num_d_trips = inst.num_d_trips
        num_trips = inst.num_trips
        num_departure = len(inst.departure_times)
        num_eq = num_d_trips + 2 * num_trips
        num_ub = num_departure

        solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("  MILP-S: No MILP solver, falling back to NO-S")
            return columns

        # Variables: theta_p for all columns, B, y_p for new columns only
        theta_vars = [solver.NumVar(0.0, solver.infinity(), f'theta_{i}') for i in range(num_total)]
        B_var = solver.NumVar(0.0, solver.infinity(), 'B')
        y_vars = [solver.IntVar(0, 1, f'y_{i}') for i in range(num_new)]

        # Objective
        objective = solver.Objective()
        for i, col in enumerate(self.rmp.columns + columns):
            objective.SetCoefficient(theta_vars[i], col.cost)
        objective.SetCoefficient(B_var, inst.bus_fixed_cost)
        for i in range(num_new):
            objective.SetCoefficient(y_vars[i], self.epsilon)
        objective.SetMinimization()

        # Build coefficient vectors
        col_vectors = []
        for col in self.rmp.columns + columns:
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

        # 1. Equality constraints (d-trip, bus arrival, bus departure)
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
            print(f"  MILP-S: Solve status {status}, falling back to all columns")
            return columns

        # Get selected columns (y_p = 1)
        selected_indices = []
        for i in range(num_new):
            if y_vars[i].solution_value() > 0.5:
                selected_indices.append(i)

        selected_cols = [columns[i] for i in selected_indices]

        # Add 50% of remaining negative reduced cost columns
        unselected = [(i, columns[i]) for i in range(num_new) if i not in selected_indices]
        unselected.sort(key=lambda x: x[1].reduced_cost)
        n_additional = int(len(unselected) * self.additional_pct)
        additional = [col for _, col in unselected[:n_additional]]

        result = selected_cols + additional

        # Ensure at least min_select columns
        if len(result) < self.min_select:
            result = columns[:self.min_select]

        print(f"  MILP-S: {len(selected_cols)} selected + {len(additional)} additional = {len(result)} total")

        return result
