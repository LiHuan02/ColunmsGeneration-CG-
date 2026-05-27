import numpy as np
from ortools.linear_solver import pywraplp
from core.rmp import RestrictedMasterProblem


class VCSPRMP(RestrictedMasterProblem):
    """VCSP Restricted Master Problem.

    LP relaxation of:
    Min  c*B + sum(c_p * theta_p)
    s.t.
      sum(e_vp * theta_p) == 1, ∀ v ∈ V          (d-trip covering)
      sum(f_wp * theta_p) == 1, ∀ w ∈ W          (bus arrival)
      sum(g_wp * theta_p) == 1, ∀ w ∈ W          (bus departure)
      sum(q_hp * theta_p) - B <= 0, ∀ h ∈ H      (bus count)
      theta_p >= 0, B >= 0
    """

    def __init__(self, instance, initial_columns=None):
        super().__init__(initial_columns)
        self.instance = instance
        self.bus_solution = None
        self.column_solution = []

        # Constraint counts
        self.num_d_trips = instance.num_d_trips
        self.num_trips = instance.num_trips
        self.num_departure_times = len(instance.departure_times)

        # Constraint indexing
        # [0, num_d_trips): d-trip covering (eq)
        # [num_d_trips, num_d_trips + num_trips): bus arrival (eq)
        # [num_d_trips + num_trips, num_d_trips + 2*num_trips): bus departure (eq)
        # [num_d_trips + 2*num_trips, num_d_trips + 2*num_trips + num_departure_times): bus count (ub)
        self.idx_d_trip = 0
        self.idx_arrival = self.num_d_trips
        self.idx_departure = self.num_d_trips + self.num_trips
        self.idx_bus_count = self.num_d_trips + 2 * self.num_trips

    def _get_num_eq(self):
        return self.num_d_trips + 2 * self.num_trips

    def _get_num_ub(self):
        return self.num_departure_times

    def _get_total_constraints(self):
        return self._get_num_eq() + self._get_num_ub()

    def solve(self):
        """Solve the RMP using ortools LP solver (GLOP)."""
        num_cols = len(self.columns)
        if num_cols == 0:
            raise ValueError("No columns in RMP")

        num_eq = self._get_num_eq()
        num_ub = self._get_num_ub()
        num_total_constraints = num_eq + num_ub

        # Create solver
        solver = pywraplp.Solver.CreateSolver('GLOP')
        if not solver:
            raise RuntimeError("GLOP solver not available")

        # Variables: [theta_0, ..., theta_{n-1}, B]
        theta_vars = [solver.NumVar(0.0, solver.infinity(), f'theta_{i}') for i in range(num_cols)]
        B_var = solver.NumVar(0.0, solver.infinity(), 'B')

        # Objective: min sum(c_p * theta_p) + c * B
        objective = solver.Objective()
        for i, col in enumerate(self.columns):
            objective.SetCoefficient(theta_vars[i], col.cost)
        objective.SetCoefficient(B_var, self.instance.bus_fixed_cost)
        objective.SetMinimization()

        # --- Constraints ---

        # Build constraint coefficient arrays
        # For each column, compute its constraint column
        col_vectors = []
        for col in self.columns:
            vec = np.zeros(num_total_constraints)
            # D-trip covering
            for d_id in col.d_trips:
                vec[self.idx_d_trip + d_id] = 1.0
            # Bus arrival
            for w_id in col.f_trips:
                vec[self.idx_arrival + w_id] = 1.0
            # Bus departure
            for w_id in col.g_trips:
                vec[self.idx_departure + w_id] = 1.0
            # Bus count
            for h_idx, h in enumerate(self.instance.departure_times):
                if col.q_times.get(h, False):
                    vec[self.idx_bus_count + h_idx] = 1.0
            col_vectors.append(vec)

        # 1. Equality constraints: d-trip covering, bus arrival, bus departure
        eq_constraints = []
        for c_idx in range(num_eq):
            ct = solver.Constraint(1.0, 1.0)  # == 1
            for i, col_vec in enumerate(col_vectors):
                if col_vec[c_idx] != 0:
                    ct.SetCoefficient(theta_vars[i], col_vec[c_idx])
            # B does not appear in equality constraints
            eq_constraints.append(ct)

        # 2. Bus count constraints: sum(q_hp * theta_p) - B <= 0
        bus_count_constraints = []
        for h_idx in range(num_ub):
            c_idx = num_eq + h_idx
            ct = solver.Constraint(-solver.infinity(), 0.0)  # <= 0
            for i, col_vec in enumerate(col_vectors):
                if col_vec[c_idx] != 0:
                    ct.SetCoefficient(theta_vars[i], col_vec[c_idx])
            ct.SetCoefficient(B_var, -1.0)  # -B
            bus_count_constraints.append(ct)

        # Solve
        status = solver.Solve()
        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            print(f"RMP solve status: {status}")
            print(f"  Columns: {num_cols}")
            print(f"  Constraints: {num_eq} eq + {num_ub} ub")
            raise RuntimeError(f"RMP solution failed with status {status}")

        # Extract primal solution
        self.objective_value = objective.Value()
        self.column_solution = [theta_vars[i].solution_value() for i in range(num_cols)]
        self.bus_solution = B_var.solution_value()

        # Extract dual values
        # Equality constraints: dual can be positive or negative
        duals = []
        for ct in eq_constraints:
            duals.append(ct.dual_value())
        # Bus count constraints: dual should be <= 0 (inequality of the form <=)
        for ct in bus_count_constraints:
            duals.append(ct.dual_value())

        # OR-Tools dual for <= constraints is <= 0 (negate to get standard form dual >= 0)
        self.dual_values = {
            'alpha': np.array(duals[self.idx_d_trip:self.idx_d_trip + self.num_d_trips]),
            'beta': np.array(duals[self.idx_arrival:self.idx_arrival + self.num_trips]),
            'gamma': np.array(duals[self.idx_departure:self.idx_departure + self.num_trips]),
            'delta': -np.array(duals[self.idx_bus_count:self.idx_bus_count + self.num_departure_times]),
        }

        return self.column_solution, self.dual_values, self.objective_value

    def get_bus_solution(self):
        return self.bus_solution

    def print_solution_summary(self):
        """Print a summary of the current solution."""
        print(f"RMP Objective: {self.objective_value:.2f}")
        print(f"Number of buses: {self.bus_solution:.2f}")
        num_positive = sum(1 for v in self.column_solution if v > 1e-6)
        print(f"Positive theta variables: {num_positive} / {len(self.column_solution)}")
        print(f"Dual values: |alpha|={np.abs(self.dual_values['alpha']).sum():.2f}, "
              f"|beta|={np.abs(self.dual_values['beta']).sum():.2f}, "
              f"|gamma|={np.abs(self.dual_values['gamma']).sum():.2f}, "
              f"|delta|={np.abs(self.dual_values['delta']).sum():.2f}")
