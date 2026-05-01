import numpy as np
import json
from main.generation.base_problem import BaseProblem


class DisasterResponseGenerator(BaseProblem):
    """
    Multi-Period Disaster Response Logistics & Allocation (MILP).
    
    Decision Variables (for each day t):
    - x[i,j,k,t]: Amount of supply k sent from depot i to unit j.
    - s[j,k,t]: Shortage/Slack of supply k at unit j.
    - v[i,j,t]: Number of vehicles assigned to route i->j. (Integer)
    - y[i,j,t]: Route i->j is active/secured. (Binary)

    Constraints:
    1. Depot Capacity: Sum of outflows <= Supply at depot.
    2. Demand Satisfaction: Inflow + Shortage >= Demand.
    3. Vehicle Capacity: Total volume on route <= v * capacity_per_vehicle.
    4. Route Activation: v <= M * y (Risk/Fixed cost trigger).
    5. Fleet Size: Sum of v <= Total Vehicles.

    Objective:
    Minimize Cost = TransportCost + PenaltyCost + VehicleCost + RiskCost
    """

    SYSTEM_PROMPT = """You are a Senior Disaster Response Logistics Officer writing a concise operational briefing for a supply distribution problem.

**TASK:** Write a concise and professional scenario describing a disaster response logistics challenge. Frame it as an operational situation.

**CRITICAL INSTRUCTION - TEMPLATE GENERATION:**
- You must generate a **text template** that uses specific placeholders for data.
- **DO NOT** attempt to describe the specific data values (numbers, matrices) in the narrative.
- **DO NOT** invent numbers. Use the placeholders.
- **DO NOT** write headers or titles for the data placeholders. The placeholders will be replaced by formatted tables/lists including headers.
- **DO NOT** embed large data matrices in the middle of sentences.
- Place all data placeholders in a dedicated "ANNEX" or "DATA" section at the end of the briefing.

**WRITING STYLE:**
- Be concise and direct. Avoid excessive "flavor text" or "fluff".
- Write as if briefing response coordinators on a real operation (e.g., "Forward Operating Base Alpha has limited medical supply stocks...")
- Give depots and units realistic names or designations (e.g., "Supply Depot Bravo", "3rd Relief Battalion")
- Briefly explain constraints (e.g., "Routes through the valley are exposed to hazardous conditions...")
- **CRITICAL:** Explicitly explain that 'activating' a route (sending any vehicles) incurs a **fixed Risk Cost**, representing security overhead (e.g., escort teams, route clearance), regardless of the number of vehicles.
- Focus on the mission objective.

**REQUIRED PLACEHOLDERS:**
Use these exactly as written:
{DEMANDS}
{DEPOT_SUPPLIES}
{TRANSPORT_COSTS}
{RISK_COSTS}
{PENALTY_COSTS}
{SUPPLY_VOLUMES}
{VEHICLE_COST}
{VEHICLE_CAPACITY}
{TOTAL_VEHICLES}

**SUGGESTED STRUCTURE:**
1. **Situation:** Brief operational context.
2. **Mission:** The goal of the logistics operation.
3. **Execution/Concept:** Concise summary of constraints and approach.
4. **Annex A - Logistics Data:**
   {DEMANDS}
   {DEPOT_SUPPLIES}
   {TRANSPORT_COSTS}
   {RISK_COSTS}
   {PENALTY_COSTS}
   {SUPPLY_VOLUMES}
   Fleet Parameters: Cost {VEHICLE_COST}, Capacity {VEHICLE_CAPACITY}, Total {TOTAL_VEHICLES}
"""

    def __init__(
        self,
        n_depots: int = 2,
        n_units: int = 3,
        n_supplies: int = 2,
        n_days: int = 1,
        total_vehicles: int = 20,
        vehicle_capacity: float = 100.0,
        prob_type: str = "LP",
        goal: str = "MINIMIZE",
    ):
        self.D = n_depots
        self.U = n_units
        self.S = n_supplies
        self.T = n_days
        self.total_vehicles = total_vehicles
        self.vehicle_capacity = vehicle_capacity

        # Variable Counts
        self.num_x = self.D * self.U * self.S * self.T
        self.num_s = self.U * self.S * self.T
        self.num_v = self.D * self.U * self.T
        self.num_y = self.D * self.U * self.T
        
        n_vars = self.num_x + self.num_s + self.num_v + self.num_y

        # Constraint Counts
        # 1. Depot Cap: D * S * T
        # 2. Demand: U * S * T
        # 3. Veh Cap: D * U * T
        # 4. Route Act: D * U * T
        # 5. Fleet Size: T
        n_constrs = (self.D * self.S * self.T) + \
                    (self.U * self.S * self.T) + \
                    (self.D * self.U * self.T) + \
                    (self.D * self.U * self.T) + \
                    self.T

        super().__init__(
            n_vars, 
            n_constrs, 
            prob_type=prob_type, 
            goal=goal, 
            problem_type="disaster response logistics"
        )
        
        # Initialize instance data placeholders
        self.demands = None        # [u, s, t]
        self.depot_supplies = None # [d, s, t]
        self.transport_costs = None # [d, u] (per unit volume)
        self.risk_costs = None     # [d, u] (fixed cost if route used)
        self.vehicle_cost = 10.0   # Cost per vehicle deployed
        self.penalty_costs = None  # [s] (per unit shortage)
        self.supply_volumes = None # [s] (volume per unit of supply)

    def generate(self, time_limit: float = 300.0):
        self._generate_instance_data()
        self._build_matrices()
        self.solve(time_limit=time_limit)
        return self

    def _generate_instance_data(self):
        # 1. Supply Volumes (random 0.5 to 2.0)
        self.supply_volumes = np.round(np.random.uniform(0.5, 2.0, size=self.S), 2)

        # 2. Demands (Unit, Supply, Day)
        # Random demand 10-50
        self.demands = np.round(np.random.uniform(10, 50, size=(self.U, self.S, self.T)), 2)

        # 3. Depot Supplies (Depot, Supply, Day)
        # Make total supply somewhat abundant globally, but maybe tight locally
        total_demand = np.sum(self.demands, axis=0) # [S, T]
        avg_demand_per_depot = total_demand / self.D
        # Depots have 80% to 150% of average needed share
        self.depot_supplies = np.zeros((self.D, self.S, self.T))
        for t in range(self.T):
            for s in range(self.S):
                for d in range(self.D):
                    base = avg_demand_per_depot[s, t]
                    self.depot_supplies[d, s, t] = np.round(np.random.uniform(0.8 * base, 1.5 * base), 2)

        # 4. Costs
        # Transport cost depends on distance (random proxy)
        self.transport_costs = np.round(np.random.uniform(1.0, 5.0, size=(self.D, self.U)), 2)
        
        # Risk cost (fixed cost for opening route)
        self.risk_costs = np.round(np.random.uniform(50.0, 200.0, size=(self.D, self.U)), 2)
        
        # Penalty costs (high to discourage shortage)
        self.penalty_costs = np.round(np.random.uniform(50.0, 100.0, size=self.S), 2)

    def _build_matrices(self):
        # We need to populate self.A, self.b, self.c, self.senses, self.var_types, self.var_names, self.constr_names
        
        # Offsets for variables in the flat vector
        # Order: x[d,u,s,t], s[u,s,t], v[d,u,t], y[d,u,t]
        
        idx_x = 0
        idx_s = idx_x + self.num_x
        idx_v = idx_s + self.num_s
        idx_y = idx_v + self.num_v
        
        # Helper to get flat indices
        def get_idx_x(d, u, s, t):
            return idx_x + t*(self.D*self.U*self.S) + d*(self.U*self.S) + u*(self.S) + s
            
        def get_idx_s(u, s, t):
            return idx_s + t*(self.U*self.S) + u*(self.S) + s
            
        def get_idx_v(d, u, t):
            return idx_v + t*(self.D*self.U) + d*(self.U) + u

        def get_idx_y(d, u, t):
            return idx_y + t*(self.D*self.U) + d*(self.U) + u

        # 1. Objective Function (c)
        self.c = np.zeros(self.n)
        
        # x: transport cost
        for t in range(self.T):
            for d in range(self.D):
                for u in range(self.U):
                    cost = self.transport_costs[d, u]
                    for s in range(self.S):
                        idx = get_idx_x(d, u, s, t)
                        self.c[idx] = cost # Cost per unit transported

        # s: penalty cost
        for t in range(self.T):
            for u in range(self.U):
                for s in range(self.S):
                    idx = get_idx_s(u, s, t)
                    self.c[idx] = self.penalty_costs[s]

        # v: vehicle cost
        for t in range(self.T):
            for d in range(self.D):
                for u in range(self.U):
                    idx = get_idx_v(d, u, t)
                    self.c[idx] = self.vehicle_cost

        # y: risk cost
        for t in range(self.T):
            for d in range(self.D):
                for u in range(self.U):
                    idx = get_idx_y(d, u, t)
                    self.c[idx] = self.risk_costs[d, u]

        # 2. Variable Types and Names
        self.var_types = []
        self.var_names = []
        
        # x
        for i in range(self.num_x):
            self.var_types.append("Continuous")
        # s
        for i in range(self.num_s):
            self.var_types.append("Continuous")
        # v
        for i in range(self.num_v):
            self.var_types.append("Integer")
        # y
        for i in range(self.num_y):
            self.var_types.append("Binary")

        # Set names for debugging/readability
        # Re-iterate to match order exactly
        # Note: BaseProblem constructor set generic names, we overwrite them
        curr_names = []
        for t in range(self.T):
            for d in range(self.D):
                for u in range(self.U):
                    for s in range(self.S):
                        curr_names.append(f"Flow_D{d}_U{u}_S{s}_Day{t}")
        for t in range(self.T):
            for u in range(self.U):
                for s in range(self.S):
                    curr_names.append(f"Slack_U{u}_S{s}_Day{t}")
        for t in range(self.T):
            for d in range(self.D):
                for u in range(self.U):
                    curr_names.append(f"Veh_D{d}_U{u}_Day{t}")
        for t in range(self.T):
            for d in range(self.D):
                for u in range(self.U):
                    curr_names.append(f"Route_D{d}_U{u}_Day{t}")
        self.var_names = curr_names

        # 3. Constraints
        self.A = np.zeros((self.m, self.n))
        self.b = np.zeros(self.m)
        self.senses = []
        self.constr_names = []

        row_idx = 0

        # C1: Depot Capacity
        # Sum_j x_{d,j,k,t} <= Supply_{d,k,t}
        for t in range(self.T):
            for d in range(self.D):
                for s in range(self.S):
                    for u in range(self.U):
                        col = get_idx_x(d, u, s, t)
                        self.A[row_idx, col] = 1.0
                    self.b[row_idx] = self.depot_supplies[d, s, t]
                    self.senses.append("<=")
                    self.constr_names.append(f"DepotCap_D{d}_S{s}_Day{t}")
                    row_idx += 1

        # C2: Demand Satisfaction
        # Sum_i x_{i,u,k,t} + s_{u,k,t} >= Demand_{u,k,t}
        for t in range(self.T):
            for u in range(self.U):
                for s in range(self.S):
                    for d in range(self.D):
                        col = get_idx_x(d, u, s, t)
                        self.A[row_idx, col] = 1.0
                    col_s = get_idx_s(u, s, t)
                    self.A[row_idx, col_s] = 1.0
                    
                    self.b[row_idx] = self.demands[u, s, t]
                    self.senses.append(">=") # Could be "="
                    self.constr_names.append(f"Demand_U{u}_S{s}_Day{t}")
                    row_idx += 1

        # C3: Vehicle Capacity
        # Sum_k (Vol_k * x_{d,u,k,t}) - Cap * v_{d,u,t} <= 0
        for t in range(self.T):
            for d in range(self.D):
                for u in range(self.U):
                    # LHS sum
                    for s in range(self.S):
                        col_x = get_idx_x(d, u, s, t)
                        self.A[row_idx, col_x] = self.supply_volumes[s]
                    # - Cap * v
                    col_v = get_idx_v(d, u, t)
                    self.A[row_idx, col_v] = -1.0 * self.vehicle_capacity
                    
                    self.b[row_idx] = 0.0
                    self.senses.append("<=")
                    self.constr_names.append(f"VehCap_D{d}_U{u}_Day{t}")
                    row_idx += 1

        # C4: Route Activation
        # v_{d,u,t} - M * y_{d,u,t} <= 0
        # M can be total_vehicles (since v <= total_vehicles)
        M = self.total_vehicles
        for t in range(self.T):
            for d in range(self.D):
                for u in range(self.U):
                    col_v = get_idx_v(d, u, t)
                    col_y = get_idx_y(d, u, t)
                    
                    self.A[row_idx, col_v] = 1.0
                    self.A[row_idx, col_y] = -1.0 * M
                    
                    self.b[row_idx] = 0.0
                    self.senses.append("<=")
                    self.constr_names.append(f"RouteAct_D{d}_U{u}_Day{t}")
                    row_idx += 1
        
        # C5: Fleet Size
        # Sum_{d,u} v_{d,u,t} <= TotalVehicles
        for t in range(self.T):
            for d in range(self.D):
                for u in range(self.U):
                    col_v = get_idx_v(d, u, t)
                    self.A[row_idx, col_v] = 1.0
            self.b[row_idx] = self.total_vehicles
            self.senses.append("<=")
            self.constr_names.append(f"FleetSize_Day{t}")
            row_idx += 1

    def _get_instance_data(self):
        return {
            "n_depots": self.D,
            "n_units": self.U,
            "n_supplies": self.S,
            "n_days": self.T,
            "total_vehicles": self.total_vehicles,
            "vehicle_capacity": self.vehicle_capacity,
            "demands": self.demands.tolist() if self.demands is not None else None,
            "depot_supplies": self.depot_supplies.tolist() if self.depot_supplies is not None else None,
            "transport_costs": self.transport_costs.tolist() if self.transport_costs is not None else None,
            "risk_costs": self.risk_costs.tolist() if self.risk_costs is not None else None,
            "vehicle_cost": self.vehicle_cost,
            "supply_volumes": self.supply_volumes.tolist() if self.supply_volumes is not None else None,
            "penalty_costs": self.penalty_costs.tolist() if self.penalty_costs is not None else None,
        }

    def generate_gurobi_code_reference(self, data):
        """
        Generate a compact, matrix-based Gurobi script for disaster response logistics problems.
        
        Overrides the base class method to produce more readable, matrix-structured code
        using loops and quicksum instead of individual variable declarations.
        Uses only instance_data to reconstruct the problem.
        """
        instance_data = data.get("instance_data", {})
        meta = data.get("meta", {})
        
        n_depots = instance_data.get("n_depots", 0)
        n_units = instance_data.get("n_units", 0)
        n_supplies = instance_data.get("n_supplies", 0)
        n_days = instance_data.get("n_days", 0)
        total_vehicles = instance_data.get("total_vehicles", 0)
        vehicle_capacity = instance_data.get("vehicle_capacity", 0.0)
        vehicle_cost = instance_data.get("vehicle_cost", 0.0)
        
        demands = instance_data.get("demands", [])
        depot_supplies = instance_data.get("depot_supplies", [])
        transport_costs = instance_data.get("transport_costs", [])
        risk_costs = instance_data.get("risk_costs", [])
        supply_volumes = instance_data.get("supply_volumes", [])
        penalty_costs = instance_data.get("penalty_costs", [])
        
        goal = meta.get("goal", "MINIMIZE")
        sense = "GRB.MINIMIZE" if goal == "MINIMIZE" else "GRB.MAXIMIZE"
        
        lines = []
        lines.append("```python")
        lines.append("from gurobipy import *")
        lines.append("")
        lines.append("def solve_problem():")
        lines.append(f"    m = Model(\"Disaster_Response_Logistics\")")
        lines.append("    m.setParam(\"TimeLimit\", 300)  # Set time limit to 300 seconds")
        lines.append("")
        lines.append("    # Data")
        lines.append(f"    depots = list(range({n_depots}))  # D0..D{n_depots-1}")
        lines.append(f"    units = list(range({n_units}))  # U0..U{n_units-1}")
        lines.append(f"    supplies = list(range({n_supplies}))  # S0..S{n_supplies-1}")
        lines.append(f"    days = list(range({n_days}))  # Day0..Day{n_days-1}")
        lines.append("")
        lines.append(f"    demands = {demands}")
        lines.append(f"    depot_supplies = {depot_supplies}")
        lines.append(f"    transport_costs = {transport_costs}")
        lines.append(f"    risk_costs = {risk_costs}")
        lines.append(f"    supply_volumes = {supply_volumes}")
        lines.append(f"    penalty_costs = {penalty_costs}")
        lines.append(f"    vehicle_cost = {vehicle_cost}")
        lines.append(f"    vehicle_capacity = {vehicle_capacity}")
        lines.append(f"    total_vehicles = {total_vehicles}")
        lines.append("")
        lines.append("    # Decision variables")
        lines.append("    # Flow_D{d}_U{u}_S{s}_Day{t}: Amount of supply s from depot d to unit u on day t")
        lines.append("    flows = {}")
        lines.append("    for t in days:")
        lines.append("        for d in depots:")
        lines.append("            for u in units:")
        lines.append("                for s in supplies:")
        lines.append(f'                    flows[(d, u, s, t)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"Flow_D{{d}}_U{{u}}_S{{s}}_Day{{t}}")')
        lines.append("")
        lines.append("    # Slack_U{u}_S{s}_Day{t}: Shortage of supply s at unit u on day t")
        lines.append("    slacks = {}")
        lines.append("    for t in days:")
        lines.append("        for u in units:")
        lines.append("            for s in supplies:")
        lines.append(f'                slacks[(u, s, t)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"Slack_U{{u}}_S{{s}}_Day{{t}}")')
        lines.append("")
        lines.append("    # Veh_D{d}_U{u}_Day{t}: Number of vehicles on route d->u on day t")
        lines.append("    vehicles = {}")
        lines.append("    for t in days:")
        lines.append("        for d in depots:")
        lines.append("            for u in units:")
        lines.append(f'                vehicles[(d, u, t)] = m.addVar(lb=0.0, vtype=GRB.INTEGER, name=f"Veh_D{{d}}_U{{u}}_Day{{t}}")')
        lines.append("")
        lines.append("    # Route_D{d}_U{u}_Day{t}: Route d->u is active on day t (binary)")
        lines.append("    routes = {}")
        lines.append("    for t in days:")
        lines.append("        for d in depots:")
        lines.append("            for u in units:")
        lines.append(f'                routes[(d, u, t)] = m.addVar(vtype=GRB.BINARY, name=f"Route_D{{d}}_U{{u}}_Day{{t}}")')
        lines.append("")
        lines.append("    # Objective: Minimize total cost (transport + penalty + vehicle + risk)")
        lines.append("    obj = quicksum(")
        lines.append("        transport_costs[d][u] * flows[(d, u, s, t)]")
        lines.append("        for t in days for d in depots for u in units for s in supplies")
        lines.append("    )")
        lines.append("    obj += quicksum(")
        lines.append("        penalty_costs[s] * slacks[(u, s, t)]")
        lines.append("        for t in days for u in units for s in supplies")
        lines.append("    )")
        lines.append("    obj += quicksum(")
        lines.append("        vehicle_cost * vehicles[(d, u, t)]")
        lines.append("        for t in days for d in depots for u in units")
        lines.append("    )")
        lines.append("    obj += quicksum(")
        lines.append("        risk_costs[d][u] * routes[(d, u, t)]")
        lines.append("        for t in days for d in depots for u in units")
        lines.append("    )")
        lines.append(f"    m.setObjective(obj, {sense})")
        lines.append("")
        lines.append("    # Depot capacity constraints: Total outflow <= supply")
        lines.append("    for t in days:")
        lines.append("        for d in depots:")
        lines.append("            for s in supplies:")
        lines.append("                m.addConstr(")
        lines.append("                    quicksum(flows[(d, u, s, t)] for u in units) <= depot_supplies[d][s][t],")
        lines.append(f'                    name=f"DepotCap_D{{d}}_S{{s}}_Day{{t}}"')
        lines.append("                )")
        lines.append("")
        lines.append("    # Demand constraints: Inflow + slack >= demand")
        lines.append("    for t in days:")
        lines.append("        for u in units:")
        lines.append("            for s in supplies:")
        lines.append("                m.addConstr(")
        lines.append("                    quicksum(flows[(d, u, s, t)] for d in depots) + slacks[(u, s, t)] >= demands[u][s][t],")
        lines.append(f'                    name=f"Demand_U{{u}}_S{{s}}_Day{{t}}"')
        lines.append("                )")
        lines.append("")
        lines.append("    # Vehicle capacity constraints: Total volume <= vehicles * capacity")
        lines.append("    for t in days:")
        lines.append("        for d in depots:")
        lines.append("            for u in units:")
        lines.append("                m.addConstr(")
        lines.append("                    quicksum(supply_volumes[s] * flows[(d, u, s, t)] for s in supplies) <= vehicle_capacity * vehicles[(d, u, t)],")
        lines.append(f'                    name=f"VehCap_D{{d}}_U{{u}}_Day{{t}}"')
        lines.append("                )")
        lines.append("")
        lines.append("    # Route activation constraints: vehicles <= total_vehicles * route")
        lines.append("    for t in days:")
        lines.append("        for d in depots:")
        lines.append("            for u in units:")
        lines.append("                m.addConstr(")
        lines.append("                    vehicles[(d, u, t)] <= total_vehicles * routes[(d, u, t)],")
        lines.append(f'                    name=f"RouteAct_D{{d}}_U{{u}}_Day{{t}}"')
        lines.append("                )")
        lines.append("")
        lines.append("    # Fleet size constraints: Total vehicles <= total_vehicles")
        lines.append("    for t in days:")
        lines.append("        m.addConstr(")
        lines.append("            quicksum(vehicles[(d, u, t)] for d in depots for u in units) <= total_vehicles,")
        lines.append(f'            name=f"FleetSize_Day{{t}}"')
        lines.append("        )")
        lines.append("")
        lines.append("    m.optimize()")
        lines.append("    return m")
        lines.append("```")
        
        return "\n".join(lines)

    @classmethod
    def build_prompt_messages(cls, problem_data: dict) -> list[dict]:
        """
        Build prompt messages with instance_data clearly separated and visible.
        The LLM should provide a compact description template while the matrix structure remains directly accessible.
        """
        import json
        
        clean_data = cls._sanitize_for_prompt(problem_data)
        system_prompt = cls.get_system_prompt()
        
        # Extract instance_data for clear presentation
        instance_data = clean_data.get("instance_data", {})
        meta = clean_data.get("meta", {})
        
        # Build a structured prompt that guides template creation
        n_depots = instance_data.get('n_depots', 0)
        n_units = instance_data.get('n_units', 0)
        n_supplies = instance_data.get('n_supplies', 0)
        n_days = instance_data.get('n_days', 0)
        
        # Note: We do NOT provide values to the LLM to prevent it from hallucinating or embedding them.
        # It must use the placeholders.
        
        user_content = f"""Problem: {meta.get('problem_type', 'Disaster Response Logistics')} ({meta.get('goal', 'MINIMIZE')})
Scale: {n_depots} depots, {n_units} units, {n_supplies} supplies, {n_days} days

**Variables:**
- Flow_D{{d}}_U{{u}}_S{{s}}_Day{{t}}: Continuous (≥0)
- Slack_U{{u}}_S{{s}}_Day{{t}}: Continuous (≥0)
- Veh_D{{d}}_U{{u}}_Day{{t}}: Integer (≥0)
- Route_D{{d}}_U{{u}}_Day{{t}}: Binary (1 if route is active, 0 otherwise)

**Constraints:**
1. Depot capacity (≤): Sum_u Flow_D{{d}}_U{{u}}_S{{s}}_Day{{t}} ≤ {{DEPOT_SUPPLIES}}[d][s][t]
2. Demand (≥): Sum_d Flow_D{{d}}_U{{u}}_S{{s}}_Day{{t}} + Slack_U{{u}}_S{{s}}_Day{{t}} ≥ {{DEMANDS}}[u][s][t]
3. Vehicle capacity (≤): Sum_s ({{SUPPLY_VOLUMES}}[s] × Flow_D{{d}}_U{{u}}_S{{s}}_Day{{t}}) ≤ {{VEHICLE_CAPACITY}} × Veh_D{{d}}_U{{u}}_Day{{t}}
4. Route activation (≤): Veh_D{{d}}_U{{u}}_Day{{t}} ≤ {{TOTAL_VEHICLES}} × Route_D{{d}}_U{{u}}_Day{{t}} (Vehicles can only use active routes)
5. Fleet size (≤): Sum_d,u Veh_D{{d}}_U{{u}}_Day{{t}} ≤ {{TOTAL_VEHICLES}}

**MISSION:** Minimize total operational cost (transport + shortage penalties + vehicle deployment + fixed route security risk costs).

Write a realistic disaster response briefing that captures this scenario. Focus on the operational context and goals.
IMPORTANT: Do NOT include any specific numbers or data values in your text. You MUST use the provided placeholders in an Annex section.
"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    @classmethod
    def finish_description(cls, llm_description: str, problem_data: dict) -> str | None:
        """
        Fill in template placeholders in the LLM description with actual instance data.
        
        Args:
            llm_description: The template description from LLM with placeholders.
            problem_data: The full problem data dictionary containing instance_data.
            
        Returns:
            The description with all placeholders filled in, or None if an error occurs.
        """
        if not llm_description:
            return llm_description
        
        # Required placeholders for disaster response logistics
        required_placeholders = ["DEMANDS", "DEPOT_SUPPLIES", "TRANSPORT_COSTS", "RISK_COSTS", 
                                 "PENALTY_COSTS", "SUPPLY_VOLUMES", "VEHICLE_COST", "VEHICLE_CAPACITY", "TOTAL_VEHICLES"]
        
        # Check if description contains template placeholders
        has_placeholders = any(f"{{{p}}}" in llm_description for p in required_placeholders)
        
        # Check for missing placeholders
        missing_placeholders = []
        for placeholder in required_placeholders:
            placeholder_str = f"{{{placeholder}}}"
            if placeholder_str not in llm_description:
                missing_placeholders.append(placeholder)
        
        if not has_placeholders:
            # No placeholders to fill, return as-is
            return llm_description
        
        instance_data = problem_data.get("instance_data", {})
        if not instance_data:
            # No instance data available, return template as-is
            return llm_description
        
        if missing_placeholders:
            print(f"❌ Error: Missing required placeholders in LLM description: {missing_placeholders}")
            return None
        
        # Check all required instance_data fields at once
        required_fields = ["demands", "depot_supplies", "transport_costs", "risk_costs", 
                          "penalty_costs", "supply_volumes", "vehicle_cost", "vehicle_capacity", "total_vehicles"]
        missing_fields = [field for field in required_fields if field not in instance_data]
        
        if missing_fields:
            print(f"❌ Error: Missing required fields in instance_data: {missing_fields}")
            return None
        
        # Helper for formatting tables
        def format_table(matrix, row_label="Row", col_label="Col", row_prefix="R", col_prefix="C"):
            if not matrix:
                return "[]"
            rows = len(matrix)
            cols = len(matrix[0]) if rows > 0 else 0
            
            lines = []
            header = f"{row_label}\\{col_label} | " + " | ".join(f"{col_prefix}{j:2d}" for j in range(cols))
            lines.append(header)
            lines.append("-" * len(header))
            for i in range(rows):
                row_str = f"{row_prefix}{i:2d}        | " + " | ".join(f"{val:6.2f}" for val in matrix[i])
                lines.append(row_str)
            return "\n".join(lines)

        # Format instance data for insertion
        formatted_data = {}
        
        # Format demands [U, S, T]
        demands = instance_data["demands"]
        # Assuming T=1 for simplified view, or iterate days
        n_days = len(demands[0][0]) if demands and demands[0] and demands[0][0] else 1
        # If demands is [U, S, T], actually json serializes it as lists.
        # Demands structure is [U][S][T] based on generation logic.
        # Check structure:
        # np.random.uniform(..., size=(U, S, T)).tolist() -> list of lists of lists.
        # dem[u][s][t]
        
        # For readability, we will flatten T if T=1
        demands_str = ""
        u_len = len(demands)
        s_len = len(demands[0]) if u_len > 0 else 0
        t_len = len(demands[0][0]) if s_len > 0 else 0
        
        if t_len == 1:
            # Create U x S table
            table_data = [[demands[u][s][0] for s in range(s_len)] for u in range(u_len)]
            demands_str = "Unit Demands (Units x Supplies):\n" + format_table(table_data, "Unit", "Supply", "U", "S")
        else:
            # Multiple days
            blocks = []
            for t in range(t_len):
                table_data = [[demands[u][s][t] for s in range(s_len)] for u in range(u_len)]
                blocks.append(f"Day {t} Demands:\n" + format_table(table_data, "Unit", "Supply", "U", "S"))
            demands_str = "\n\n".join(blocks)
            
        formatted_data["DEMANDS"] = demands_str
        
        # Format depot supplies [D, S, T]
        depot_supplies = instance_data["depot_supplies"]
        d_len = len(depot_supplies)
        s_len = len(depot_supplies[0]) if d_len > 0 else 0
        t_len = len(depot_supplies[0][0]) if s_len > 0 else 0
        
        depot_str = ""
        if t_len == 1:
            table_data = [[depot_supplies[d][s][0] for s in range(s_len)] for d in range(d_len)]
            depot_str = "Depot Supplies (Depots x Supplies):\n" + format_table(table_data, "Depot", "Supply", "D", "S")
        else:
            blocks = []
            for t in range(t_len):
                table_data = [[depot_supplies[d][s][t] for s in range(s_len)] for d in range(d_len)]
                blocks.append(f"Day {t} Depot Supplies:\n" + format_table(table_data, "Depot", "Supply", "D", "S"))
            depot_str = "\n\n".join(blocks)
            
        formatted_data["DEPOT_SUPPLIES"] = depot_str
        
        # Format transport cost matrix [D, U]
        transport_costs = instance_data["transport_costs"]
        formatted_data["TRANSPORT_COSTS"] = "Transport Costs (Depot x Unit):\n" + format_table(transport_costs, "Depot", "Unit", "D", "U")
        
        # Format risk cost matrix [D, U]
        risk_costs = instance_data["risk_costs"]
        formatted_data["RISK_COSTS"] = "Risk Costs (Depot x Unit):\n" + format_table(risk_costs, "Depot", "Unit", "D", "U")
        
        # Format penalty costs [S]
        penalty_costs = instance_data["penalty_costs"]
        penalty_str = ", ".join(f"Supply S{i}: {val}" for i, val in enumerate(penalty_costs))
        formatted_data["PENALTY_COSTS"] = f"Penalty Costs per Unit Shortage: [{penalty_str}]"
        
        # Format supply volumes [S]
        supply_volumes = instance_data["supply_volumes"]
        vol_str = ", ".join(f"Supply S{i}: {val}" for i, val in enumerate(supply_volumes))
        formatted_data["SUPPLY_VOLUMES"] = f"Volume per Supply Unit: [{vol_str}]"
        
        # Format scalar values
        formatted_data["VEHICLE_COST"] = str(instance_data["vehicle_cost"])
        formatted_data["VEHICLE_CAPACITY"] = str(instance_data["vehicle_capacity"])
        formatted_data["TOTAL_VEHICLES"] = str(instance_data["total_vehicles"])
        
        # Fill in placeholders using targeted string replacement
        result = llm_description
        for placeholder, value in formatted_data.items():
            result = result.replace(f"{{{placeholder}}}", value)
        
        # Verify all placeholders were filled
        remaining_placeholders = []
        for placeholder in required_placeholders:
            if f"{{{placeholder}}}" in result:
                remaining_placeholders.append(placeholder)
        
        if remaining_placeholders:
            print(f"❌ Error: Placeholders were not filled: {remaining_placeholders}")
            return None
        
        return result
