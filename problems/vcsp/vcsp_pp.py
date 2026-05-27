import numpy as np
from core.pp import PricingProblem


class Label:
    """Label for the dynamic programming labeling algorithm.

    Represents a partial path from source to a node.
    Tracks both actual cost and reduced cost separately.
    """

    __slots__ = ['node_id', 'cost', 'reduced_cost', 'duty_length', 'predecessor', 'arc_used',
                 'd_trips', 'f_trips', 'g_trips', 'q_times']

    def __init__(self, node_id, cost=0.0, reduced_cost=0.0, duty_length=0.0,
                 predecessor=None, arc_used=None,
                 d_trips=None, f_trips=None, g_trips=None, q_times=None):
        self.node_id = node_id
        self.cost = cost  # actual operational cost
        self.reduced_cost = reduced_cost  # cost - dual contributions
        self.duty_length = duty_length
        self.predecessor = predecessor
        self.arc_used = arc_used
        self.d_trips = {} if d_trips is None else dict(d_trips)
        self.f_trips = {} if f_trips is None else dict(f_trips)
        self.g_trips = {} if g_trips is None else dict(g_trips)
        self.q_times = {} if q_times is None else dict(q_times)

    def dominates(self, other):
        """Check if this label dominates another at the same node.
        L1 dominates L2 if both cost and duty_length are <= L2's.
        """
        if self.node_id != other.node_id:
            return False
        return (self.reduced_cost <= other.reduced_cost + 1e-10 and
                self.duty_length <= other.duty_length + 1e-10)

    def to_column(self, duty_type):
        """Convert the full label path to a VCSPColumn.
        Uses ACTUAL cost (not reduced cost) for the column.
        """
        from problems.vcsp.column import VCSPColumn
        nodes = []
        arcs = []
        label = self
        while label is not None and label.arc_used is not None:
            arcs.append(label.arc_used)
            nodes.append(label.node_id)
            label = label.predecessor
        nodes.append(label.node_id if label else 0)
        nodes.reverse()
        arcs.reverse()

        return VCSPColumn(
            duty_type=duty_type,
            cost=self.cost,
            node_sequence=nodes,
            arc_sequence=arcs,
            d_trips=self.d_trips,
            f_trips=self.f_trips,
            g_trips=self.g_trips,
            q_times=self.q_times,
            reduced_cost=self.reduced_cost,
        )

    def __repr__(self):
        return f"Label(node={self.node_id}, cost={self.cost:.2f}, rc={self.reduced_cost:.2f}, len={self.duty_length:.0f})"


class VCSPPricingProblem(PricingProblem):
    """VCSP Pricing Problem: Resource-Constrained Shortest Path.

    Finds minimum reduced cost duty paths in the driver network G_u.
    Uses forward labeling algorithm (dynamic programming).
    """

    MAX_DUTY_LENGTH = 300

    def __init__(self, duty_type, driver_network, instance):
        super().__init__(duty_type)
        self.network = driver_network
        self.instance = instance
        self.dual_values = None

    def set_dual_values(self, dual_values):
        self.dual_values = dual_values

    def _get_arc_actual_cost(self, arc):
        """Return the actual operational cost of an arc."""
        return arc['cost']

    def _get_arc_reduced_cost(self, arc, from_nid, to_nid):
        """Compute reduced cost = actual cost - dual contributions."""
        if self.dual_values is None:
            return arc['cost']

        dv = self.dual_values
        rc = arc['cost']
        arc_type = arc['type']

        if arc_type == self.network.ARC_D_TRIP:
            d_id = arc['d_trip_id']
            rc -= dv['alpha'][d_id]
            from_time = self.network.nodes[from_nid]['time']
            to_time = self.network.nodes[to_nid]['time']
            for h_idx, h in enumerate(self.instance.departure_times):
                if from_time <= h < to_time:
                    rc -= dv['delta'][h_idx]

        elif arc_type == self.network.ARC_INTER_TRIP_DRIVING:
            trip_id = arc['trip_id']
            rc -= dv['beta'][trip_id]
            from_node = self.network.nodes[from_nid]
            if from_node['type'] == 'end_of_trip':
                rc -= dv['gamma'][from_node['trip_id']]
            from_time = self.network.nodes[from_nid]['time']
            to_time = self.network.nodes[to_nid]['time']
            for h_idx, h in enumerate(self.instance.departure_times):
                if from_time <= h < to_time:
                    rc -= dv['delta'][h_idx]

        elif arc_type == self.network.ARC_START_OF_DUTY:
            if arc['is_driving']:
                to_node = self.network.nodes[to_nid]
                if to_node['type'] == 'start_of_trip':
                    rc -= dv['beta'][to_node['trip_id']]
                from_time = to_node['time'] - arc['time']
                to_time = to_node['time']
                for h_idx, h in enumerate(self.instance.departure_times):
                    if from_time <= h < to_time:
                        rc -= dv['delta'][h_idx]

        elif arc_type == self.network.ARC_END_OF_DUTY:
            if arc['is_driving']:
                from_node = self.network.nodes[from_nid]
                if from_node['type'] == 'end_of_trip':
                    rc -= dv['gamma'][from_node['trip_id']]
                from_time = self.network.nodes[from_nid]['time']
                to_time = from_time + arc['time']
                for h_idx, h in enumerate(self.instance.departure_times):
                    if from_time <= h < to_time:
                        rc -= dv['delta'][h_idx]

        return rc

    def _extend_label(self, label, from_nid, to_nid, arc):
        """Extend a label along an arc.
        Tracks both actual cost and reduced cost.
        """
        new_cost = label.cost + self._get_arc_actual_cost(arc)
        new_rc = label.reduced_cost + self._get_arc_reduced_cost(arc, from_nid, to_nid)
        new_duty_length = label.duty_length + arc['time']

        if new_duty_length > self.MAX_DUTY_LENGTH + 1e-10:
            return None

        new_d_trips = dict(label.d_trips)
        new_f_trips = dict(label.f_trips)
        new_g_trips = dict(label.g_trips)
        new_q_times = dict(label.q_times)

        arc_type = arc['type']

        if arc_type == self.network.ARC_D_TRIP:
            new_d_trips[arc['d_trip_id']] = True
            from_time = self.network.nodes[from_nid]['time']
            to_time = self.network.nodes[to_nid]['time']
            for h in self.instance.departure_times:
                if from_time <= h < to_time:
                    new_q_times[h] = True

        elif arc_type == self.network.ARC_INTER_TRIP_DRIVING:
            new_f_trips[arc['trip_id']] = True
            from_node = self.network.nodes[from_nid]
            if from_node['type'] == 'end_of_trip':
                new_g_trips[from_node['trip_id']] = True
            from_time = self.network.nodes[from_nid]['time']
            to_time = self.network.nodes[to_nid]['time']
            for h in self.instance.departure_times:
                if from_time <= h < to_time:
                    new_q_times[h] = True

        elif arc_type == self.network.ARC_START_OF_DUTY:
            if arc['is_driving']:
                to_node = self.network.nodes[to_nid]
                if to_node['type'] == 'start_of_trip':
                    new_f_trips[to_node['trip_id']] = True
                from_time = to_node['time'] - arc['time']
                to_time = to_node['time']
                for h in self.instance.departure_times:
                    if from_time <= h < to_time:
                        new_q_times[h] = True

        elif arc_type == self.network.ARC_END_OF_DUTY:
            if arc['is_driving']:
                from_node = self.network.nodes[from_nid]
                if from_node['type'] == 'end_of_trip':
                    new_g_trips[from_node['trip_id']] = True
                from_time = self.network.nodes[from_nid]['time']
                to_time = from_time + arc['time']
                for h in self.instance.departure_times:
                    if from_time <= h < to_time:
                        new_q_times[h] = True

        return Label(
            node_id=to_nid,
            cost=new_cost,
            reduced_cost=new_rc,
            duty_length=new_duty_length,
            predecessor=label,
            arc_used=(from_nid, to_nid),
            d_trips=new_d_trips,
            f_trips=new_f_trips,
            g_trips=new_g_trips,
            q_times=new_q_times,
        )

    def _dominance_filter(self, labels):
        """Apply dominance: keep only non-dominated labels (by reduced_cost and duty_length)."""
        filtered = []
        for i, l1 in enumerate(labels):
            dominated = False
            for j, l2 in enumerate(labels):
                if i == j:
                    continue
                if l2.dominates(l1):
                    dominated = True
                    break
            if not dominated:
                filtered.append(l1)
        return filtered

    def solve(self, heuristic=True):
        """Solve the pricing problem using the labeling algorithm.

        Returns: list of VCSPColumn with negative reduced cost.
        """
        source = self.network.NODE_SOURCE
        sink = self.network.NODE_SINK

        labels_at_node = {nid: [] for nid in self.network.get_node_ids()}
        initial = Label(node_id=source, cost=0.0, reduced_cost=0.0, duty_length=0.0)
        labels_at_node[source].append(initial)

        node_times = [(nid, info['time']) for nid, info in self.network.nodes.items()]
        node_times.sort(key=lambda x: x[1] if x[1] != float('inf') else 1e9)

        for nid, _ in node_times:
            current = labels_at_node[nid]
            if not current:
                continue
            current = self._dominance_filter(current)
            labels_at_node[nid] = current

            for (f, t, rest), arc_info in self.network.get_outgoing_arcs(nid):
                for label in current:
                    new_label = self._extend_label(label, f, t, arc_info)
                    if new_label is not None:
                        labels_at_node[t].append(new_label)

        sink_labels = labels_at_node.get(sink, [])
        sink_labels = self._dominance_filter(sink_labels)

        new_columns = []
        for label in sink_labels:
            if label.reduced_cost < -1e-6:
                col = label.to_column(self.duty_type)
                new_columns.append(col)

        new_columns.sort(key=lambda c: c.cost)  # sort by actual cost

        self.new_columns = new_columns
        return new_columns

    def print_stats(self):
        print(f"PP (Type {self.duty_type}): {len(self.new_columns)} columns generated")
        if self.new_columns:
            print(f"  Best actual cost: {self.new_columns[0].cost:.4f}")

    def has_negative_reduced_cost(self):
        return any(c.cost < -1e-6 for c in self.new_columns)
