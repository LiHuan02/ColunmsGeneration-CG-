from abc import ABC, abstractmethod


class PricingProblem(ABC):
    """Abstract base class for the Pricing Problem (subproblem)."""

    def __init__(self, duty_type):
        self.duty_type = duty_type
        self.new_columns = []

    @abstractmethod
    def set_dual_values(self, dual_values):
        """Set dual values to compute reduced costs on arcs."""
        pass

    @abstractmethod
    def solve(self, heuristic=True):
        """Solve the pricing problem.
        Returns: list of new columns with negative reduced cost.
        """
        pass

    def get_new_columns(self):
        """Get columns generated in the last solve."""
        return self.new_columns
