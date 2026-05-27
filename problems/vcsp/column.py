import numpy as np


class VCSPColumn:
    """Represents a driver duty (path in driver network G_u).

    Binary parameters (from Haase et al. 2001):
    - e_vp: 1 if path contains d-trip v
    - f_wp: 1 if path has driving movement ending at start location of trip w
    - g_wp: 1 if path has driving movement starting from end location of trip w
    - q_hp: 1 if path has driving/bus attendance task covering departure time h
    """

    def __init__(self, duty_type, cost, node_sequence, arc_sequence,
                 d_trips=None, f_trips=None, g_trips=None, q_times=None,
                 reduced_cost=None):
        self.duty_type = duty_type  # 'I' or 'II'
        self.cost = cost  # actual operational cost
        self.reduced_cost = reduced_cost if reduced_cost is not None else cost
        self.node_sequence = list(node_sequence)
        self.arc_sequence = list(arc_sequence)  # list of (from, to, arc_type)

        # Binary parameter maps
        self.d_trips = d_trips if d_trips is not None else {}  # d_trip_id -> bool
        self.f_trips = f_trips if f_trips is not None else {}  # trip_id -> bool (bus arrival)
        self.g_trips = g_trips if g_trips is not None else {}  # trip_id -> bool (bus departure)
        self.q_times = q_times if q_times is not None else {}  # departure_time -> bool

    def covers_d_trip(self, d_trip_id):
        return self.d_trips.get(d_trip_id, False)

    def get_e_vp_array(self, num_d_trips):
        """Return e_vp as a dense numpy array."""
        return np.array([1 if self.d_trips.get(v, False) else 0 for v in range(num_d_trips)], dtype=float)

    def get_f_wp_array(self, num_trips):
        """Return f_wp as a dense numpy array."""
        return np.array([1 if self.f_trips.get(w, False) else 0 for w in range(num_trips)], dtype=float)

    def get_g_wp_array(self, num_trips):
        """Return g_wp as a dense numpy array."""
        return np.array([1 if self.g_trips.get(w, False) else 0 for w in range(num_trips)], dtype=float)

    def get_q_hp_array(self, departure_times):
        """Return q_hp as a dense numpy array (ordered by departure_times)."""
        return np.array([1 if self.q_times.get(h, False) else 0 for h in departure_times], dtype=float)

    def get_full_column(self, instance):
        """Return full constraint column for the master problem.

        Order: [e_vp (all d-trips), f_wp (all trips), g_wp (all trips), q_hp (all departure times), B coefficient]
        """
        e = self.get_e_vp_array(instance.num_d_trips)
        f = self.get_f_wp_array(instance.num_trips)
        g = self.get_g_wp_array(instance.num_trips)
        q = self.get_q_hp_array(instance.departure_times)
        # B coefficient for this column is 0 (B only appears in its own column)
        return np.concatenate([e, f, g, q, [0]])

    def __repr__(self):
        return f"VCSPColumn(type={self.duty_type}, cost={self.cost:.2f}, arcs={len(self.arc_sequence)})"
