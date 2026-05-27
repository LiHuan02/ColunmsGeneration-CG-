"""Column and constraint feature extraction for VCSP bipartite graph.

Features from EBSCO paper Section 4.2:

Column features (12):
  1. cost
  2. reduced_cost
  3. total number of constraints the column contributes to
  4-7. number of constraints per constraint group (4 groups)
  8. duty_length (work time)
  9. duty_type (0 for 'I')
  10. columnIsNew (1 if newly generated, 0 if in basis)
  11. column_value (theta_p if in basis, 0 otherwise)
  12. column incompatibility degree

Constraint features (2):
  1. dual_value
  2. node degree (number of columns contributing to this constraint)
"""

import numpy as np


class FeatureExtractor:
    """Extract node features for columns and constraints in the bipartite graph."""

    def __init__(self, instance):
        self.instance = instance
        self.num_eq = instance.num_d_trips + 2 * instance.num_trips
        self.num_ub = len(instance.departure_times)
        self.num_total_constraints = self.num_eq + self.num_ub

    # ------------------------------------------------------------------
    # Constraint group counts for a single column
    # ------------------------------------------------------------------
    def _constraint_group_counts(self, col):
        """Return (n_d_trip, n_arrival, n_departure, n_bus_count) for a column."""
        return (
            len(col.d_trips),
            len(col.f_trips),
            len(col.g_trips),
            len(col.q_times),
        )

    def _total_constraints(self, col):
        return len(col.d_trips) + len(col.f_trips) + len(col.g_trips) + len(col.q_times)

    # ------------------------------------------------------------------
    # Column incompatibility degree (simplified from Elhallaoui et al. 2010)
    # Measures how much a column overlaps with the current basis:
    #   incomp = 1 - (shared_constraints / total_nonzero_constraints)
    # where shared_constraints counts constraints where BOTH this column
    # AND at least one basic column have a non-zero entry.
    # ------------------------------------------------------------------
    def _incompatibility_degree(self, col, basis_constraint_support):
        """Compute column incompatibility degree.

        Args:
            col: the column to evaluate
            basis_constraint_support: set of constraint indices that have
                non-zero support from at least one basic column
        """
        nonzero_cols = self._total_constraints(col)
        if nonzero_cols == 0:
            return 0.0

        # Build the set of constraints this column contributes to
        col_constraints = set()
        for d_id in col.d_trips:
            col_constraints.add(d_id)  # d-trip group
        for w_id in col.f_trips:
            col_constraints.add(self.instance.num_d_trips + w_id)  # arrival group
        for w_id in col.g_trips:
            col_constraints.add(self.instance.num_d_trips + self.instance.num_trips + w_id)  # departure group
        for h_idx, h in enumerate(self.instance.departure_times):
            if col.q_times.get(h, False):
                col_constraints.add(self.num_eq + h_idx)  # bus count group

        # Count shared constraints
        shared = len(col_constraints & basis_constraint_support)
        return 1.0 - (shared / nonzero_cols)

    # ------------------------------------------------------------------
    # Extract features for a single column
    # ------------------------------------------------------------------
    def extract_column_features(self, col, is_new, theta_value, basis_constraint_support):
        """Extract 12 column features.

        Args:
            col: VCSPColumn instance
            is_new: True if newly generated, False if in basic set
            theta_value: current theta_p value (0 if not in basis)
            basis_constraint_support: set of constraint indices with basis support
        """
        cg_counts = self._constraint_group_counts(col)
        return np.array([
            col.cost,                                   # 1. cost
            col.reduced_cost,                           # 2. reduced cost
            self._total_constraints(col),               # 3. total constraints
            cg_counts[0],                               # 4. d-trip constraints
            cg_counts[1],                               # 5. arrival constraints
            cg_counts[2],                               # 6. departure constraints
            cg_counts[3],                               # 7. bus count constraints
            col.duty_length,                            # 8. duty length / work time
            0.0 if col.duty_type == 'I' else 1.0,       # 9. duty type
            1.0 if is_new else 0.0,                     # 10. columnIsNew
            theta_value,                                # 11. column value in solution
            self._incompatibility_degree(col, basis_constraint_support),  # 12. incompatibility
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Extract features for all constraints
    # ------------------------------------------------------------------
    def extract_constraint_features(self, duals, constraint_degrees):
        """Extract 2 constraint features for each constraint.

        Args:
            duals: dict with 'alpha', 'beta', 'gamma', 'delta' as arrays
            constraint_degrees: numpy array of shape (num_total_constraints,)
                counting how many columns (basic + new) contribute to each constraint
        """
        num_d = self.instance.num_d_trips
        num_w = self.instance.num_trips
        num_h = len(self.instance.departure_times)

        features = np.zeros((self.num_total_constraints, 2), dtype=np.float32)

        # D-trip constraints: idx [0, num_d_trips)
        features[0:num_d, 0] = duals['alpha']
        features[0:num_d, 1] = constraint_degrees[0:num_d]

        # Bus arrival constraints: idx [num_d, num_d + num_w)
        features[num_d:num_d + num_w, 0] = duals['beta']
        features[num_d:num_d + num_w, 1] = constraint_degrees[num_d:num_d + num_w]

        # Bus departure constraints: idx [num_d + num_w, num_d + 2*num_w)
        features[num_d + num_w:num_d + 2 * num_w, 0] = duals['gamma']
        features[num_d + num_w:num_d + 2 * num_w, 1] = constraint_degrees[num_d + num_w:num_d + 2 * num_w]

        # Bus count constraints: idx [num_d + 2*num_w, total)
        features[num_d + 2 * num_w:, 0] = -duals['delta']  # store as non-negative (standard form dual)
        features[num_d + 2 * num_w:, 1] = constraint_degrees[num_d + 2 * num_w:]

        return features

    # ------------------------------------------------------------------
    # Build per-column constraint coefficient vector
    # ------------------------------------------------------------------
    def column_coefficient_vector(self, col):
        """Return dense constraint coefficient vector for a column.

        Order: [d-trip (0..nd), arrival (nd..nd+nw), departure (nd+nw..nd+2nw),
                bus_count (nd+2nw..total)]
        """
        vec = np.zeros(self.num_total_constraints, dtype=np.float32)
        for d_id in col.d_trips:
            vec[d_id] = 1.0
        for w_id in col.f_trips:
            vec[self.instance.num_d_trips + w_id] = 1.0
        for w_id in col.g_trips:
            vec[self.instance.num_d_trips + self.instance.num_trips + w_id] = 1.0
        for h_idx, h in enumerate(self.instance.departure_times):
            if col.q_times.get(h, False):
                vec[self.num_eq + h_idx] = 1.0
        return vec

    # ------------------------------------------------------------------
    # Build basis constraint support (union of non-zero constraint indices
    # across all basic columns)
    # ------------------------------------------------------------------
    def build_basis_constraint_support(self, columns, basic_indices):
        """Build the set of constraint indices supported by basic columns.

        Args:
            columns: list of all VCSPColumn in the RMP
            basic_indices: list of indices of basic columns (theta_p > 0)
        """
        support = set()
        for idx in basic_indices:
            col = columns[idx]
            coeff_vec = self.column_coefficient_vector(col)
            support.update(np.nonzero(coeff_vec)[0].tolist())
        return support

    # ------------------------------------------------------------------
    # Compute constraint degrees (how many columns contribute to each constraint)
    # ------------------------------------------------------------------
    def compute_constraint_degrees(self, all_columns):
        """Compute degree of each constraint node = number of columns
        (basic + new) that have a non-zero coefficient for that constraint.
        """
        degrees = np.zeros(self.num_total_constraints, dtype=np.float32)
        for col in all_columns:
            coeff_vec = self.column_coefficient_vector(col)
            degrees += (coeff_vec > 0).astype(np.float32)
        return np.maximum(degrees, 1.0)  # avoid zero degrees
