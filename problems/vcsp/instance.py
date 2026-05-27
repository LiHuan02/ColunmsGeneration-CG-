import numpy as np


class VCSPInstance:
    """Random VCSP instance generator (simplified version).

    Based on Haase, Desaulniers, Desrosiers (2001) Section 4.1.
    Simplified: single bus line system with configurable trips.
    """

    # Peak hour probability distribution (Table 1 in the paper)
    # Hours 0-23, peak: 7-9 (morning), 16-18 (afternoon)
    PEAK_HOUR_DIST = {
        # off-peak hours (0-6, 10-15, 19-23): lower probability
        # peak hours (7-9, 16-18): higher probability
    }

    def __init__(
        self,
        num_trips=50,
        num_relief_points_per_trip=2,
        seed=42,
        bus_fixed_cost=50000,
        driver_fixed_cost=50000,
        cost_per_minute=1.0,
        sign_on_time=10,
        sign_off_time=10,
        day_start=360,  # 6:00 AM
        day_end=1320,   # 10:00 PM
    ):
        self.num_trips = num_trips
        self.num_d_trips = num_trips * num_relief_points_per_trip
        self.num_relief_points_per_trip = num_relief_points_per_trip
        self.seed = seed
        self.bus_fixed_cost = bus_fixed_cost
        self.driver_fixed_cost = driver_fixed_cost
        self.cost_per_minute = cost_per_minute
        self.sign_on_time = sign_on_time
        self.sign_off_time = sign_off_time
        self.day_start = day_start
        self.day_end = day_end

        self.rng = np.random.RandomState(seed)

        # Generate relief point coordinates
        self.depot_location = (0, 0)
        self._generate_coordinates()

        # Generate trips
        self.trips = []
        self._generate_trips()

        # Compute departure times for bus count constraints
        self._compute_departure_times()

    def _generate_coordinates(self):
        """Generate random relief point coordinates on a grid."""
        self.relief_points = {}
        # Each trip has a start, end, and optional intermediate relief points
        for t in range(self.num_trips):
            start = (
                self.rng.randint(-50, 50),
                self.rng.randint(-50, 50),
            )
            end = (
                self.rng.randint(-50, 50),
                self.rng.randint(-50, 50),
            )
            # Intermediate relief point (linear interpolation)
            mid = (
                (start[0] + end[0]) // 2,
                (start[1] + end[1]) // 2,
            )
            self.relief_points[f"t{t}_start"] = start
            self.relief_points[f"t{t}_mid"] = mid
            self.relief_points[f"t{t}_end"] = end

    def _get_coords(self, loc_name):
        """Get coordinates for a location name (handles 'depot')."""
        if loc_name == 'depot':
            return self.depot_location
        return self.relief_points[loc_name]

    def _euclidean_distance(self, loc1, loc2):
        """Compute truncated Euclidean distance between two locations."""
        dx = loc1[0] - loc2[0]
        dy = loc1[1] - loc2[1]
        return int(np.round(np.sqrt(dx ** 2 + dy ** 2)))

    def _generate_trips(self):
        """Generate trips with random start times and durations."""
        for t in range(self.num_trips):
            # Start time uniformly distributed within day window
            start_time = self.rng.randint(self.day_start, self.day_end)

            # Duration: 30-90 minutes
            duration = self.rng.randint(30, 91)

            # Segment times (d-trip durations)
            seg1_dur = int(duration * 0.4)
            seg2_dur = duration - seg1_dur

            start_loc = f"t{t}_start"
            mid_loc = f"t{t}_mid"
            end_loc = f"t{t}_end"

            # Cost of driving the trip
            driving_cost = duration * self.cost_per_minute

            self.trips.append({
                'id': t,
                'start_time': start_time,
                'end_time': start_time + duration,
                'duration': duration,
                'start_location': start_loc,
                'mid_location': mid_loc,
                'end_location': end_loc,
                'd_trips': [
                    {'id': t * self.num_relief_points_per_trip, 'trip_id': t,
                     'start_node': f"t{t}_start_of_trip", 'end_node': f"t{t}_relief",
                     'start_location': start_loc, 'end_location': mid_loc,
                     'start_time': start_time, 'end_time': start_time + seg1_dur,
                     'duration': seg1_dur},
                    {'id': t * self.num_relief_points_per_trip + 1, 'trip_id': t,
                     'start_node': f"t{t}_relief", 'end_node': f"t{t}_end_of_trip",
                     'start_location': mid_loc, 'end_location': end_loc,
                     'start_time': start_time + seg1_dur, 'end_time': start_time + duration,
                     'duration': seg2_dur},
                ],
                'cost': driving_cost,
            })

    def _compute_departure_times(self):
        """Compute H: set of times at which a bus must leave the depot."""
        self.departure_times = set()
        for trip in self.trips:
            start_coords = self._get_coords(trip['start_location'])
            deadhead_time = self._euclidean_distance(self.depot_location, start_coords)
            departure_time = trip['start_time'] - deadhead_time - self.sign_on_time
            self.departure_times.add(departure_time)
        self.departure_times = sorted(self.departure_times)

    def driving_time(self, from_loc, to_loc):
        """Driving time between two locations (Euclidean distance)."""
        return self._euclidean_distance(
            self._get_coords(from_loc),
            self._get_coords(to_loc),
        )

    def walking_time(self, from_loc, to_loc):
        """Walking time = driving + 10 minutes."""
        return self.driving_time(from_loc, to_loc) + 10

    def get_num_trips(self):
        return self.num_trips

    def get_num_d_trips(self):
        return self.num_d_trips

    def get_num_departure_times(self):
        return len(self.departure_times)
