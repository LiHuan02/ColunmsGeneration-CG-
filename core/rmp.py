from abc import ABC, abstractmethod


class RestrictedMasterProblem(ABC):
    """Abstract base class for the Restricted Master Problem."""

    def __init__(self, initial_columns=None):
        self.columns = [] if initial_columns is None else initial_columns
        self.current_solution = {}
        self.dual_values = {}
        self.objective_value = None

    def add_columns(self, new_columns):
        """Add new columns to the RMP."""
        self.columns.extend(new_columns)

    @abstractmethod
    def solve(self):
        """Solve the RMP: return primal solution, dual values, and objective value."""
        pass

    def get_current_columns(self):
        """Get all columns in the RMP."""
        return self.columns

    def get_dual_values(self):
        """Get dual values from the last solve."""
        return self.dual_values

    def get_objective_value(self):
        """Get objective value from the last solve."""
        return self.objective_value


class GenericRMP(RestrictedMasterProblem):
    """Generic RMP that can handle mixed constraints (eq + ineq)."""

    def __init__(self, num_eq_constraints, num_ub_constraints, initial_columns=None):
        super().__init__(initial_columns)
        self.num_eq = num_eq_constraints
        self.num_ub = num_ub_constraints

    @abstractmethod
    def solve(self):
        pass
