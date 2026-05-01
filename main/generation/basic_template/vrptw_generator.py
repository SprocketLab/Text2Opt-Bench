import numpy as np
from main.generation.base_problem import BaseProblem


class VRPTWGenerator(BaseProblem):
    """
    Vehicle Routing Problem with Time Windows (VRPTW) - MILP.

    A fleet of vehicles delivers goods from a central depot to customers.
    Each customer has a demand, a time window, and a service time.
    Vehicles have limited capacity and must return to the depot.

    Uses the 2-index MTZ formulation:
      Node 0 = depot, nodes 1..N = customers.

    Decision Variables:
    - Arc_i_j:  Binary     — arc from node i to node j is used
    - Load_i:   Continuous — cumulative demand served when leaving customer i (MTZ)
    - Time_i:   Continuous — service start time at node i

    Constraints:
    1. Visit Out:  Σ_{j≠i} x[i,j] = 1           for each customer i
    2. Visit In:   Σ_{i≠j} x[i,j] = 1           for each customer j
    3. Vehicle Limit: Σ_j x[0,j] ≤ K
    4. MTZ Subtour:   u[j] - u[i] + Q·x[i,j] ≤ Q - d[j]   for customer pairs i≠j
    5. Time Linking:  t[i] - t[j] + M·x[i,j] ≤ M - service[i] - travel[i,j]
                      for arcs ending at customers

    Objective: Minimize Σ distance[i,j]·x[i,j] + vehicle_cost · Σ_j x[0,j]
    """

    SYSTEM_PROMPT = """You are a Logistics Operations Manager writing a memo about a delivery routing challenge.

**TASK:** Write a natural, business-like problem description for a vehicle routing and scheduling problem.

**TONE:** Professional operations memo. The reader should understand the delivery challenge intuitively.

**REQUIRED PLACEHOLDERS (include ALL exactly once):**
{DEPOT_LOCATION}, {DEPOT_CLOSING_TIME}, {CUSTOMER_LOCATIONS}, {CUSTOMER_DEMANDS}, {TIME_WINDOWS}, {SERVICE_TIMES}, {VEHICLE_CAPACITY}, {NUM_VEHICLES}, {VEHICLE_FIXED_COST}

**CRITICAL - DATA HANDLING:**
- Use ONLY the placeholders above for data. Do NOT write out sample values.
- Each placeholder should appear EXACTLY ONCE.
- Place all data placeholders in a dedicated "ANNEX" or "DATA" section at the end.

**WHAT TO INCLUDE:**
- A delivery company operates a fleet of identical vehicles from a central depot
- All vehicles depart the depot at time 0 and must return to the depot before the planning horizon ends (the depot's latest time, provided in the data)
- Each vehicle makes a single trip: leaves the depot at time 0, visits a sequence of customers, and returns before the depot closing time
- Each customer has a delivery time window (earliest and latest acceptable delivery time). Delivery must START within this window
- If a vehicle arrives before the customer's earliest time, it must wait until the window opens
- Each customer requires a certain service/unloading time at the location. After serving, the vehicle departs at arrival_time + service_time (or window_open + service_time if it waited)
- Each vehicle has limited carrying capacity (total demand on its route must not exceed capacity)
- Using each vehicle incurs a fixed deployment cost on top of the travel distance cost
- Travel time between any two locations equals the straight-line (Euclidean) distance between their coordinates, with an implicit speed of 1 (distance units = time units)
- Goal: Find vehicle routes that visit every customer exactly once, respect time windows and capacity, and minimize total travel distance plus vehicle deployment costs

**WHAT TO AVOID:**
- Mathematical formulas or technical jargon (no "MTZ", "Big-M", "subtour elimination")
- Do NOT say "compute Euclidean distance" or give the sqrt formula
- Do NOT list sample data — only use placeholders
- Do NOT use OR terminology like "objective function" or "decision variables"

**SUGGESTED STRUCTURE:**
1. Subject/Context: Overview of the delivery operation
2. Challenge: The routing and scheduling problem
3. Requirements: Capacity, time windows, service times
4. Goal: Minimize costs
5. Annex - Operations Data: All placeholders"""

    def __init__(
        self,
        n_customers: int = 8,
        n_vehicles: int = 3,
        coord_range: tuple = (5, 95),
        demand_range: tuple = (5, 25),
        service_time_range: tuple = (5, 15),
        vehicle_capacity: int | None = None,
        vehicle_fixed_cost: float = 100.0,
        time_window_width: tuple = (50, 100),
        prob_type: str = "LP",
        goal: str = "MINIMIZE",
    ):
        self.n_customers = n_customers
        self.n_vehicles = n_vehicles
        self.coord_range = coord_range
        self.demand_range = demand_range
        self.service_time_range = service_time_range
        self._vehicle_capacity = vehicle_capacity
        self.vehicle_fixed_cost = vehicle_fixed_cost
        self.time_window_width = time_window_width

        # Node 0 = depot, nodes 1..N = customers
        n_nodes = n_customers + 1
        arcs_per_node = n_nodes - 1  # no self-loops

        # Variable counts
        self.num_arcs = n_nodes * arcs_per_node   # binary
        self.num_loads = n_customers               # continuous (customers only)
        self.num_times = n_nodes                   # continuous (all nodes)
        n_vars = self.num_arcs + self.num_loads + self.num_times

        # Constraint counts
        self.n_visit_out = n_customers
        self.n_visit_in = n_customers
        self.n_vehicle_limit = 1
        self.n_mtz = n_customers * (n_customers - 1)
        self.n_time_link = n_customers * n_customers  # arcs ending at customers
        n_constrs = (self.n_visit_out + self.n_visit_in +
                     self.n_vehicle_limit + self.n_mtz + self.n_time_link)

        super().__init__(
            n_vars, n_constrs,
            prob_type=prob_type, goal=goal,
            problem_type="vehicle routing",
        )

        # Instance data (populated by _generate_instance_data)
        self.depot_coord = None
        self.customer_coords = None
        self.customer_demands = None
        self.time_windows = None       # [(a, b), ...] for depot + customers
        self.service_times = None
        self.capacity = None
        self.distance_matrix = None    # (n_nodes × n_nodes)
        self.T_max = None
        self.big_M = None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, time_limit: float = 300.0):
        self._generate_instance_data()
        self._compute_distance_matrix()
        self._build_matrices()
        self.solve(time_limit=time_limit)
        return self

    def _generate_instance_data(self):
        # Depot at centre of grid
        self.depot_coord = (50, 50)

        # Random customer coordinates (integers)
        self.customer_coords = [
            (
                int(np.random.randint(self.coord_range[0], self.coord_range[1] + 1)),
                int(np.random.randint(self.coord_range[0], self.coord_range[1] + 1)),
            )
            for _ in range(self.n_customers)
        ]

        # Customer demands (integers)
        self.customer_demands = [
            int(np.random.randint(self.demand_range[0], self.demand_range[1] + 1))
            for _ in range(self.n_customers)
        ]

        # Vehicle capacity
        if self._vehicle_capacity is not None:
            self.capacity = self._vehicle_capacity
        else:
            total_demand = sum(self.customer_demands)
            avg_per_vehicle = total_demand / self.n_vehicles
            self.capacity = int(max(
                max(self.customer_demands) * 2,
                avg_per_vehicle * 1.5,
            ))

        # Service times (integers)
        self.service_times = [
            int(np.random.randint(self.service_time_range[0],
                                  self.service_time_range[1] + 1))
            for _ in range(self.n_customers)
        ]

        # Distances from depot (for time-window generation)
        depot_dists = []
        dx, dy = self.depot_coord
        for cx, cy in self.customer_coords:
            depot_dists.append(np.sqrt((cx - dx) ** 2 + (cy - dy) ** 2))

        # Customer time windows  (index 0 = depot, set below)
        # Key feasibility guarantee: each customer must be reachable from depot
        # within its time window, and the vehicle must be able to return.
        customer_tw = []
        for i in range(self.n_customers):
            dist = depot_dists[i]
            # Earliest: vehicle can arrive no sooner than dist from depot
            # Add a small random offset but ensure a ≤ dist + buffer
            offset = int(np.random.randint(0, 31))
            a = max(0, int(dist) - 5 + offset)
            # Window width: generous enough for feasibility
            width = int(np.random.randint(self.time_window_width[0],
                                          self.time_window_width[1] + 1))
            b = a + width
            # Ensure the window opens no later than when the vehicle could arrive
            # (otherwise infeasible for direct trips)
            if a > int(dist) + 30:
                a = max(0, int(dist))
                b = a + width
            customer_tw.append((a, b))

        # T_max: depot closing — enough time to return from any customer
        max_b = max(b for _, b in customer_tw)
        max_return = max(depot_dists)
        max_service = max(self.service_times)
        self.T_max = int(max_b + max_return + max_service + 100)

        # time_windows[0] = depot, time_windows[1..n] = customers
        self.time_windows = [(0, self.T_max)] + customer_tw

    def _compute_distance_matrix(self):
        all_coords = [self.depot_coord] + self.customer_coords
        n_nodes = len(all_coords)
        self.distance_matrix = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    xi, yi = all_coords[i]
                    xj, yj = all_coords[j]
                    self.distance_matrix[i, j] = np.sqrt(
                        (xi - xj) ** 2 + (yi - yj) ** 2
                    )

        # Big-M for time-linking constraints
        max_service = max(self.service_times)
        max_dist = float(np.max(self.distance_matrix))
        self.big_M = self.T_max + max_service + max_dist + 1

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _get_arc_idx(self, i, j):
        """Flat index for Arc_{i}_{j}  (i ≠ j).
        Each node has (n_nodes-1) = n_customers outgoing arcs.
        """
        n = self.n_customers  # arcs per node = n_nodes - 1
        pos = j if j < i else j - 1
        return i * n + pos

    def _get_load_idx(self, cust):
        """Flat index for Load_{cust}  (cust ∈ 1..n_customers)."""
        return self.num_arcs + (cust - 1)

    def _get_time_idx(self, node):
        """Flat index for Time_{node}  (node ∈ 0..n_customers)."""
        return self.num_arcs + self.num_loads + node

    # ------------------------------------------------------------------
    # Build A, b, c
    # ------------------------------------------------------------------

    def _build_matrices(self):
        n_cust = self.n_customers
        n_nodes = n_cust + 1
        Q = self.capacity
        M = self.big_M

        # Service time indexed by node (depot = 0)
        svc = [0] + list(self.service_times)

        # ---- Variable names ----
        self.var_names = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    self.var_names.append(f"Arc_{i}_{j}")
        for i in range(1, n_nodes):
            self.var_names.append(f"Load_{i}")
        for i in range(n_nodes):
            self.var_names.append(f"Time_{i}")

        # ---- Variable types ----
        self.var_types = (
            ["Binary"] * self.num_arcs +
            ["Continuous"] * self.num_loads +
            ["Continuous"] * self.num_times
        )

        # ---- Variable bounds ----
        self.var_bounds = []
        # Arcs
        for _ in range(self.num_arcs):
            self.var_bounds.append((0.0, 1.0))
        # Loads: [demand_i, Q]
        for i in range(1, n_nodes):
            self.var_bounds.append((float(self.customer_demands[i - 1]),
                                   float(Q)))
        # Times: [a_i, b_i]
        for i in range(n_nodes):
            a, b = self.time_windows[i]
            self.var_bounds.append((float(a), float(b)))
        # Fix depot departure time to 0
        self.var_bounds[self.num_arcs + self.num_loads] = (0.0, 0.0)

        # ---- Objective ----
        self.c = np.zeros(self.n)
        # Arc travel distance
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    idx = self._get_arc_idx(i, j)
                    self.c[idx] = self.distance_matrix[i, j]
        # Vehicle fixed cost on arcs leaving depot
        for j in range(1, n_nodes):
            idx = self._get_arc_idx(0, j)
            self.c[idx] += self.vehicle_fixed_cost

        # ---- Constraints ----
        self.A = np.zeros((self.m, self.n))
        self.b_vec = np.zeros(self.m)
        self.senses = []
        self.constr_names = []
        row = 0

        # C1: Visit Out  — Σ_{j≠i} x[i,j] = 1  for each customer i
        for i in range(1, n_nodes):
            for j in range(n_nodes):
                if j != i:
                    self.A[row, self._get_arc_idx(i, j)] = 1.0
            self.b_vec[row] = 1.0
            self.senses.append("=")
            self.constr_names.append(f"VisitOut_C{i}")
            row += 1

        # C2: Visit In  — Σ_{i≠j} x[i,j] = 1  for each customer j
        for j in range(1, n_nodes):
            for i in range(n_nodes):
                if i != j:
                    self.A[row, self._get_arc_idx(i, j)] = 1.0
            self.b_vec[row] = 1.0
            self.senses.append("=")
            self.constr_names.append(f"VisitIn_C{j}")
            row += 1

        # C3: Vehicle Limit  — Σ_j x[0,j] ≤ K
        for j in range(1, n_nodes):
            self.A[row, self._get_arc_idx(0, j)] = 1.0
        self.b_vec[row] = float(self.n_vehicles)
        self.senses.append("<=")
        self.constr_names.append("VehicleLimit")
        row += 1

        # C4: MTZ Subtour  — u[i] - u[j] + Q·x[i,j] ≤ Q - d[j]
        #     for i,j ∈ customers, i ≠ j
        for i in range(1, n_nodes):
            for j in range(1, n_nodes):
                if i != j:
                    self.A[row, self._get_load_idx(i)] = 1.0
                    self.A[row, self._get_load_idx(j)] = -1.0
                    self.A[row, self._get_arc_idx(i, j)] = float(Q)
                    self.b_vec[row] = float(Q - self.customer_demands[j - 1])
                    self.senses.append("<=")
                    self.constr_names.append(f"MTZ_{i}_{j}")
                    row += 1

        # C5: Time Linking
        #   t[i] + service[i] + travel[i,j] ≤ t[j] + M·(1 - x[i,j])
        #   ⟹  t[i] - t[j] + M·x[i,j]  ≤  M - service[i] - travel[i,j]
        #   for j ∈ customers, i ∈ all nodes \ {j}
        for j in range(1, n_nodes):
            for i in range(n_nodes):
                if i != j:
                    self.A[row, self._get_time_idx(i)] = 1.0
                    self.A[row, self._get_time_idx(j)] = -1.0
                    self.A[row, self._get_arc_idx(i, j)] = M
                    self.b_vec[row] = M - svc[i] - self.distance_matrix[i, j]
                    self.senses.append("<=")
                    self.constr_names.append(f"TimeLink_{i}_{j}")
                    row += 1

        assert row == self.m, f"Expected {self.m} constraints, built {row}"

        # BaseProblem expects self.b (not self.b_vec)
        self.b = self.b_vec

    # ------------------------------------------------------------------
    # Instance data for JSON
    # ------------------------------------------------------------------

    def _get_instance_data(self):
        return {
            "n_customers": self.n_customers,
            "n_vehicles": self.n_vehicles,
            "vehicle_capacity": self.capacity,
            "vehicle_fixed_cost": self.vehicle_fixed_cost,
            "depot_coordinates": list(self.depot_coord),
            "customer_coordinates": [list(c) for c in self.customer_coords],
            "customer_demands": list(self.customer_demands),
            "time_windows": [list(tw) for tw in self.time_windows],
            "service_times": list(self.service_times),
        }

    # ------------------------------------------------------------------
    # Gold solution code
    # ------------------------------------------------------------------

    def generate_gurobi_code_reference(self, data: dict) -> str:
        inst = data.get("instance_data", {})
        meta = data.get("meta", {})

        n_cust = inst.get("n_customers", 0)
        n_veh = inst.get("n_vehicles", 0)
        Q = inst.get("vehicle_capacity", 0)
        veh_cost = inst.get("vehicle_fixed_cost", 0)
        depot = inst.get("depot_coordinates", [50, 50])
        cust_coords = inst.get("customer_coordinates", [])
        demands = inst.get("customer_demands", [])
        tws = inst.get("time_windows", [])
        svc_times = inst.get("service_times", [])

        sense = "GRB.MINIMIZE" if meta.get("goal") == "MINIMIZE" else "GRB.MAXIMIZE"

        lines = []
        lines.append("```python")
        lines.append("import math")
        lines.append("from gurobipy import *")
        lines.append("")
        lines.append("def solve_problem():")
        lines.append('    m = Model("VRPTW")')
        lines.append("    m.setParam('TimeLimit', 300)")
        lines.append("")
        lines.append("    # --- Data ---")
        lines.append(f"    n_customers = {n_cust}")
        lines.append(f"    n_vehicles = {n_veh}")
        lines.append(f"    Q = {Q}  # vehicle capacity")
        lines.append(f"    vehicle_fixed_cost = {veh_cost}")
        lines.append(f"    depot = {depot}")
        lines.append(f"    customers = {cust_coords}")
        lines.append(f"    demands = {demands}")
        lines.append(f"    time_windows = {tws}")
        lines.append(f"    service_times = {svc_times}")
        lines.append("")
        lines.append("    # Node 0 = depot, nodes 1..N = customers")
        lines.append("    N = n_customers")
        lines.append("    nodes = list(range(N + 1))")
        lines.append("    cust_nodes = list(range(1, N + 1))")
        lines.append("")
        lines.append("    # Coordinates: index 0 = depot, 1..N = customers")
        lines.append("    coords = [depot] + customers")
        lines.append("")
        lines.append("    # COMPUTE distance matrix from coordinates")
        lines.append("    dist = {}")
        lines.append("    for i in nodes:")
        lines.append("        for j in nodes:")
        lines.append("            if i != j:")
        lines.append("                dist[(i,j)] = math.sqrt(")
        lines.append("                    (coords[i][0]-coords[j][0])**2 +")
        lines.append("                    (coords[i][1]-coords[j][1])**2)")
        lines.append("")
        lines.append("    # Service time by node (depot = 0)")
        lines.append("    svc = [0] + service_times")
        lines.append("")
        lines.append("    # Big-M for time linking")
        lines.append("    T_max = time_windows[0][1]")
        lines.append("    max_svc = max(service_times)")
        lines.append("    max_dist = max(dist.values())")
        lines.append("    M = T_max + max_svc + max_dist + 1")
        lines.append("")
        lines.append("    # --- Variables ---")
        lines.append("    # Arc variables (binary)")
        lines.append("    x = {}")
        lines.append("    for i in nodes:")
        lines.append("        for j in nodes:")
        lines.append("            if i != j:")
        lines.append('                x[(i,j)] = m.addVar(vtype=GRB.BINARY, name=f"Arc_{i}_{j}")')
        lines.append("")
        lines.append("    # Load variables (MTZ subtour elimination)")
        lines.append("    u = {}")
        lines.append("    for i in cust_nodes:")
        lines.append('        u[i] = m.addVar(lb=demands[i-1], ub=Q, vtype=GRB.CONTINUOUS, name=f"Load_{i}")')
        lines.append("")
        lines.append("    # Time variables")
        lines.append("    t = {}")
        lines.append("    for i in nodes:")
        lines.append("        a, b = time_windows[i]")
        lines.append('        t[i] = m.addVar(lb=a, ub=b, vtype=GRB.CONTINUOUS, name=f"Time_{i}")')
        lines.append("    # Fix depot departure to time 0")
        lines.append("    t[0].ub = 0.0")
        lines.append("")
        lines.append("    # --- Objective ---")
        lines.append("    # Minimize travel distance + vehicle fixed costs")
        lines.append("    obj = quicksum(dist[(i,j)] * x[(i,j)] for i in nodes for j in nodes if i != j)")
        lines.append("    obj += quicksum(vehicle_fixed_cost * x[(0,j)] for j in cust_nodes)")
        lines.append(f"    m.setObjective(obj, {sense})")
        lines.append("")
        lines.append("    # --- Constraints ---")
        lines.append("    # 1. Each customer has exactly one outgoing arc")
        lines.append("    for i in cust_nodes:")
        lines.append("        m.addConstr(")
        lines.append("            quicksum(x[(i,j)] for j in nodes if j != i) == 1,")
        lines.append('            name=f"VisitOut_C{i}")')
        lines.append("")
        lines.append("    # 2. Each customer has exactly one incoming arc")
        lines.append("    for j in cust_nodes:")
        lines.append("        m.addConstr(")
        lines.append("            quicksum(x[(i,j)] for i in nodes if i != j) == 1,")
        lines.append('            name=f"VisitIn_C{j}")')
        lines.append("")
        lines.append("    # 3. At most K vehicles leave the depot")
        lines.append("    m.addConstr(")
        lines.append("        quicksum(x[(0,j)] for j in cust_nodes) <= n_vehicles,")
        lines.append('        name="VehicleLimit")')
        lines.append("")
        lines.append("    # 4. MTZ subtour elimination")
        lines.append("    for i in cust_nodes:")
        lines.append("        for j in cust_nodes:")
        lines.append("            if i != j:")
        lines.append("                m.addConstr(")
        lines.append("                    u[i] - u[j] + Q * x[(i,j)] <= Q - demands[j-1],")
        lines.append('                    name=f"MTZ_{i}_{j}")')
        lines.append("")
        lines.append("    # 5. Time linking: if arc (i,j) is used, arrival at j >= departure from i")
        lines.append("    for j in cust_nodes:")
        lines.append("        for i in nodes:")
        lines.append("            if i != j:")
        lines.append("                m.addConstr(")
        lines.append("                    t[i] - t[j] + M * x[(i,j)] <= M - svc[i] - dist[(i,j)],")
        lines.append('                    name=f"TimeLink_{i}_{j}")')
        lines.append("")
        lines.append("    m.optimize()")
        lines.append("    return m")
        lines.append("```")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM prompt generation
    # ------------------------------------------------------------------

    @classmethod
    def build_prompt_messages(cls, problem_data: dict) -> list[dict]:
        import json

        clean_data = cls._sanitize_for_prompt(problem_data)
        system_prompt = cls.get_system_prompt()

        inst = clean_data.get("instance_data", {})
        meta = clean_data.get("meta", {})

        n_cust = inst.get("n_customers", 0)
        n_veh = inst.get("n_vehicles", 0)
        Q = inst.get("vehicle_capacity", 0)
        veh_cost = inst.get("vehicle_fixed_cost", 0)

        # Abbreviated previews
        coords_preview = inst.get("customer_coordinates", [])[:3]
        coords_str = ", ".join(f"C{i+1}: ({x},{y})" for i, (x, y) in enumerate(coords_preview))
        if n_cust > 3:
            coords_str += f", ... ({n_cust} total)"

        user_content = f"""Write a professional operations memo for a vehicle routing and scheduling problem.

**Scenario:**
A delivery company operates {n_veh} vehicles from a central depot.
They must deliver to {n_cust} customers, each with specific time windows.

**Problem Scale:**
- 1 depot + {n_cust} customer locations (with map coordinates)
- Up to {n_veh} delivery vehicles, each with capacity {Q}
- Fixed deployment cost per vehicle: ${veh_cost}
- Travel time = straight-line distance between coordinates

**Sample coordinates (DO NOT copy into memo):**
{coords_str}

**Key Rules:**
- Every customer must be visited exactly once
- Each vehicle leaves the depot, visits a sequence of customers, and returns
- Vehicle load must not exceed capacity at any point
- Must arrive within each customer's time window (can wait if early)
- Each customer requires unloading/service time
- Goal: Minimize travel distance + vehicle deployment costs

**Your Task:**
Write a business memo describing this delivery challenge.
IMPORTANT: Use ONLY these placeholders for data:
{{DEPOT_LOCATION}}, {{DEPOT_CLOSING_TIME}}, {{CUSTOMER_LOCATIONS}}, {{CUSTOMER_DEMANDS}}, {{TIME_WINDOWS}}, {{SERVICE_TIMES}}, {{VEHICLE_CAPACITY}}, {{NUM_VEHICLES}}, {{VEHICLE_FIXED_COST}}

Each placeholder should appear exactly ONCE in an Annex section.
Do NOT duplicate data by writing sample values AND using placeholders.
"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    @classmethod
    def finish_description(cls, llm_description: str, problem_data: dict) -> str | None:
        if not llm_description:
            return llm_description

        required = [
            "DEPOT_LOCATION", "DEPOT_CLOSING_TIME", "CUSTOMER_LOCATIONS",
            "CUSTOMER_DEMANDS", "TIME_WINDOWS", "SERVICE_TIMES",
            "VEHICLE_CAPACITY", "NUM_VEHICLES", "VEHICLE_FIXED_COST",
        ]

        has_placeholders = any(f"{{{p}}}" in llm_description for p in required)
        if not has_placeholders:
            return llm_description

        inst = problem_data.get("instance_data", {})
        if not inst:
            return llm_description

        missing = [p for p in required if f"{{{p}}}" not in llm_description]
        if missing:
            print(f"❌ Missing placeholders: {missing}")
            return None

        fmt = {}

        # Depot
        depot = inst.get("depot_coordinates", [50, 50])
        fmt["DEPOT_LOCATION"] = f"({depot[0]}, {depot[1]})"

        # Depot closing time (= planning horizon)
        tws_all = inst.get("time_windows", [])
        depot_close = tws_all[0][1] if tws_all else 999
        fmt["DEPOT_CLOSING_TIME"] = str(depot_close)

        # Customer locations
        cust_coords = inst.get("customer_coordinates", [])
        coord_lines = ["Customer Locations (coordinates):"]
        for i, (x, y) in enumerate(cust_coords):
            coord_lines.append(f"  - C{i+1}: ({x}, {y})")
        fmt["CUSTOMER_LOCATIONS"] = "\n".join(coord_lines)

        # Demands
        demands = inst.get("customer_demands", [])
        demand_lines = ["Customer Demands:"]
        for i, d in enumerate(demands):
            demand_lines.append(f"  - C{i+1}: {d} units")
        fmt["CUSTOMER_DEMANDS"] = "\n".join(demand_lines)

        # Time windows (skip depot at index 0)
        tws = inst.get("time_windows", [])
        tw_lines = ["Customer Time Windows [earliest, latest]:"]
        for i, tw in enumerate(tws):
            if i == 0:
                continue  # skip depot
            tw_lines.append(f"  - C{i}: [{tw[0]}, {tw[1]}]")
        fmt["TIME_WINDOWS"] = "\n".join(tw_lines)

        # Service times
        stimes = inst.get("service_times", [])
        svc_lines = ["Service Times (minutes at customer):"]
        for i, s in enumerate(stimes):
            svc_lines.append(f"  - C{i+1}: {s}")
        fmt["SERVICE_TIMES"] = "\n".join(svc_lines)

        # Scalars
        fmt["VEHICLE_CAPACITY"] = str(inst.get("vehicle_capacity", ""))
        fmt["NUM_VEHICLES"] = str(inst.get("n_vehicles", ""))
        fmt["VEHICLE_FIXED_COST"] = str(inst.get("vehicle_fixed_cost", ""))

        result = llm_description
        for ph, val in fmt.items():
            result = result.replace(f"{{{ph}}}", val)

        remaining = [p for p in required if f"{{{p}}}" in result]
        if remaining:
            print(f"❌ Placeholders not filled: {remaining}")
            return None

        return result
