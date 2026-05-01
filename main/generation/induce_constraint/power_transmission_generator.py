import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple, Optional
from main.generation.base_problem import BaseProblem


class PowerTransmissionGenerator(BaseProblem):
    """
    Power Transmission Network Design Problem with Induced Resistance Constraints (MIQP).
    
    This problem features INDUCED CONSTRAINTS/COSTS where the power loss costs
    are NOT directly given but must be COMPUTED from:
    - Node coordinates (x, y) in km
    - Wire material resistivity (ρ in Ω·m)
    - Wire cross-sectional area (A in mm²)
    - Operating voltage (V in kV)
    - Power loss cost rate ($/MWh)
    
    PHYSICS (Ideal Conditions - Ohm's Law & Joule Heating):
    - Resistance: R = ρ × L / A  (where L = distance in meters, A in m²)
    - Current: I = P / V  (Power in MW, Voltage in kV → Current in kA)
    - Power Loss: P_loss = I² × R = (P/V)² × R = (R/V²) × P²
    
    COST MODEL (Quadratic):
    - Power transmission incurs operational costs due to resistive losses
    - Loss Cost = LossRate × P_loss = LossRate × (R/V²) × Flow²
    - This creates a QUADRATIC cost structure: doubling flow quadruples losses
    
    The LLM must compute:
    1. Distance from coordinates: L_ij = sqrt((x_i-x_j)² + (y_i-y_j)²) × 1000 (km to m)
    2. Resistance: R_ij = ρ × L_ij / (A × 10⁻⁶) (convert mm² to m²)
    3. Quadratic coefficient: k_ij = LossRate × R_ij / V²
    4. Operational Cost_ij = k_ij × Flow_ij²
    
    Decision Variables:
    - Build_i_j: Binary - whether to build transmission line from node i to j
    - Flow_i_j: Continuous - power flow from node i to node j (MW)
    
    Constraints:
    1. Flow Conservation: At each node, inflow + generation = outflow + demand
    2. Capacity: Flow_i_j <= MaxCapacity × Build_i_j
    3. Symmetry: Build_i_j = Build_j_i (undirected line, but directed flow)
    
    Objective (MINIMIZE):
    - Construction costs + Power loss costs
    - = Σ (build_cost_ij × Build_i_j) + Σ (k_ij × Flow_i_j²)
    """

    SYSTEM_PROMPT = """You are a Power Systems Engineer writing a technical memo about a transmission network design decision.

**TASK:** Create a natural, human-like problem description. Write it as if an engineering client is explaining their grid planning challenge, NOT as a technical specification.

**TONE:** Professional engineering memo. The reader should understand the problem intuitively without complex electrical formulas.

**REQUIRED PLACEHOLDERS (include ALL exactly once):**
{NODE_COORDINATES}, {NODE_TYPES}, {POWER_DEMANDS}, {GENERATOR_CAPACITIES}, {WIRE_RESISTIVITY}, {WIRE_CROSS_SECTION}, {OPERATING_VOLTAGE}, {LINE_CAPACITY}, {BUILD_COST_PER_KM}, {LOSS_COST_RATE}

**CRITICAL - DATA HANDLING:**
- Use ONLY the placeholders above for data. Do NOT write out sample values or example data.
- Each placeholder should appear EXACTLY ONCE in the description.
- The placeholders will be replaced with actual data later - do not duplicate by also writing sample values.
- WRONG: "The resistivity is 1.68e-8 ohm-m... {WIRE_RESISTIVITY}"
- CORRECT: "The wire material specification is provided in {WIRE_RESISTIVITY}"

**WHAT TO INCLUDE:**
- We have a set of nodes (generators and consumers) at specific locations
- We can build transmission lines between any pair of nodes — there are no predefined corridors
- A transmission line, once built, can carry power in either direction
- Each line has a rated capacity that limits how much power can flow through it in each direction
- Building lines costs money based on the distance between the nodes
- Transmitting power incurs resistive losses that increase with the square of the power flow ($I^2R$ losses)
- Wire resistance depends on length and material properties (copper wire)
- Every node must have its power needs met: generators produce power, consumers require power, and the network must deliver enough to each consumer (surplus is acceptable)
- Important: For grid planning, assume power flow is conserved at nodes (losses are a financial penalty, not a physical deduction from the flow)
- Goal: Minimize total cost (construction + cost of lost power)

**WHAT TO AVOID:**
- Do NOT use linear approximations for power loss
- Do NOT simply sum up flows for cost - use the physics-based quadratic loss
- Do NOT list out sample data values - only use placeholders
- Do NOT model physical power loss deduction in nodal balance (assume lossless flow conservation)
- Do NOT use technical OR terminology like "binary variable", "flow conservation constraint", "complete graph", "directed/undirected"

**STYLE EXAMPLE:**
Instead of: "Minimize sum of c_ij * x_ij"
Write: "We need to account for the fact that pushing more power through a line drastically increases losses due to heating."

**COST MODEL GUIDANCE:**
- Explicitly mention that resistive losses follow a square law (quadratic) relationship with power flow.
- Explain that doubling the power flow results in four times the resistive losses.
- Connect this to the operating voltage: higher voltage reduces current, which helps mitigate these quadratic losses.
- The cost is driven by the physical loss formula: Loss ~ Resistance * (Power/Voltage)^2.
- Clarify that this is a "planning model": losses are treated as a separate financial penalty.
- Power flow is conserved at nodes (lossless mass balance); we pay for losses as an external cost, they do not reduce the power delivered.

The description should read like a real engineering planning problem where the reader naturally understands that:
- Physics dictates that losses are quadratic, not linear
- We want a precise model that captures this reality
- High voltage is used specifically to reduce these $I^2R$ losses"""

    # Physical constants - always using copper wire
    COPPER_RESISTIVITY = 1.68e-8  # Ω·m

    def __init__(
        self,
        n_nodes: int = 6,
        n_generators: int = 2,
        coord_range: Tuple[float, float] = (0.0, 100.0),  # km
        demand_range: Tuple[float, float] = (10.0, 50.0),  # MW
        capacity_range: Tuple[float, float] = (50.0, 200.0),  # MW per generator
        wire_area: float = 500.0,  # mm² (cross-sectional area)
        operating_voltage: float = 220.0,  # kV
        line_capacity: float = 100.0,  # MW per line
        build_cost_per_km: float = 100000.0,  # $/km
        loss_cost_rate: float = 50.0,  # $/MWh equivalent loss cost
        prob_type: str = "MIQP",
        goal: str = "MINIMIZE",
    ):
        self.n_nodes = n_nodes
        self.n_generators = n_generators
        self.n_consumers = n_nodes - n_generators
        self.coord_range = coord_range
        self.demand_range = demand_range
        self.capacity_range = capacity_range
        self.wire_area = wire_area  # mm²
        self.operating_voltage = operating_voltage  # kV
        self.line_capacity = line_capacity  # MW
        self.build_cost_per_km = build_cost_per_km
        self.loss_cost_rate = loss_cost_rate

        # Always use copper resistivity
        self.resistivity = self.COPPER_RESISTIVITY

        # Initialize Q matrix for quadratic objective
        self.Q = None

        # Calculate number of potential edges (for undirected graph, but with directed flow)
        # We use all pairs (i,j) where i < j for Build variables
        # And all pairs (i,j) where i != j for Flow variables
        self.n_edges = n_nodes * (n_nodes - 1) // 2  # Undirected build decisions
        self.n_flows = n_nodes * (n_nodes - 1)  # Directed flow variables

        # Variable counts
        # Build_i_j: n_edges binary variables
        # Flow_i_j: n_flows continuous variables
        n_vars = self.n_edges + self.n_flows

        # Constraint counts
        # Flow conservation: n_nodes constraints
        # Capacity constraints: n_flows constraints (Flow <= Cap * Build)
        n_constrs = n_nodes + self.n_flows

        super().__init__(
            n_vars,
            n_constrs,
            prob_type=prob_type,
            goal=goal,
            problem_type="power transmission"
        )

        # Instance data placeholders
        self.node_coords = None  # [(x, y), ...]
        self.node_types = None  # ['generator', 'consumer', ...]
        self.power_demands = None  # [demand for each consumer]
        self.generator_capacities = None  # [capacity for each generator]
        self.distance_matrix = None  # Computed from coordinates
        self.resistance_matrix = None  # Computed from distances and wire properties

    def generate(self, time_limit: float = 300.0):
        """Generate the power transmission network design problem instance."""
        self._generate_instance_data()
        self._compute_resistance_matrix()
        self._build_matrices()
        self.solve(time_limit=time_limit)
        return self

    def _generate_instance_data(self):
        """Generate random node locations and power parameters.
        
        Uses integer coordinates to avoid floating point precision issues.
        """
        # Generate node coordinates as integers
        self.node_coords = [
            (
                int(np.random.uniform(*self.coord_range)),
                int(np.random.uniform(*self.coord_range))
            )
            for _ in range(self.n_nodes)
        ]

        # Assign node types (first n_generators are generators, rest are consumers)
        self.node_types = (
            ['generator'] * self.n_generators + 
            ['consumer'] * self.n_consumers
        )

        # Generate power demands for consumers (integers)
        self.power_demands = np.zeros(self.n_nodes)
        for i in range(self.n_generators, self.n_nodes):
            self.power_demands[i] = int(np.random.uniform(*self.demand_range))

        total_demand = np.sum(self.power_demands)

        # Generate generator capacities (ensure total capacity > total demand)
        min_total_capacity = total_demand * 1.3  # 30% reserve
        avg_capacity = min_total_capacity / self.n_generators
        
        self.generator_capacities = np.zeros(self.n_nodes)
        for i in range(self.n_generators):
            self.generator_capacities[i] = int(np.random.uniform(
                max(self.capacity_range[0], avg_capacity * 0.8),
                max(self.capacity_range[1], avg_capacity * 1.5)
            ))

    def _compute_resistance_matrix(self):
        """Compute distances and resistances between all node pairs.
        
        Physics:
        - Distance: L_ij = sqrt((x_i-x_j)² + (y_i-y_j)²) km
        - Resistance: R_ij = ρ × (L_ij × 1000) / (A × 10⁻⁶) Ω
                    = ρ × L_ij × 10⁹ / A  (simplified)
        
        No rounding to ensure exact match with gold solution code.
        """
        self.distance_matrix = np.zeros((self.n_nodes, self.n_nodes))
        self.resistance_matrix = np.zeros((self.n_nodes, self.n_nodes))
        
        # Convert wire area from mm² to m²
        wire_area_m2 = self.wire_area * 1e-6
        
        for i in range(self.n_nodes):
            xi, yi = self.node_coords[i]
            for j in range(self.n_nodes):
                if i != j:
                    xj, yj = self.node_coords[j]
                    # Distance in km
                    dist_km = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                    self.distance_matrix[i, j] = dist_km
                    
                    # Distance in meters for resistance calculation
                    dist_m = dist_km * 1000
                    
                    # Resistance R = ρ × L / A (Ohm's Law for resistance)
                    resistance = self.resistivity * dist_m / wire_area_m2
                    self.resistance_matrix[i, j] = resistance

    def _build_matrices(self):
        """Build the constraint matrix A, RHS b, and objective c."""
        # Create mapping for edges (undirected, i < j)
        self.edge_list = []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                self.edge_list.append((i, j))
        
        # Create mapping for flows (directed, i != j)
        self.flow_list = []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    self.flow_list.append((i, j))

        # Index functions
        def get_idx_build(i, j):
            """Get index for Build variable (ensure i < j)."""
            if i > j:
                i, j = j, i
            return self.edge_list.index((i, j))

        def get_idx_flow(i, j):
            """Get index for Flow variable."""
            return self.n_edges + self.flow_list.index((i, j))

        # Build variable names
        self.var_names = []
        for i, j in self.edge_list:
            self.var_names.append(f"Build_{i}_{j}")
        for i, j in self.flow_list:
            self.var_names.append(f"Flow_{i}_{j}")

        # Build variable types
        self.var_types = []
        for _ in range(self.n_edges):
            self.var_types.append("Binary")
        for _ in range(self.n_flows):
            self.var_types.append("Continuous")

        # Build variable bounds
        self.var_bounds = []
        for _ in range(self.n_edges):
            self.var_bounds.append((0.0, 1.0))  # Binary
        for _ in range(self.n_flows):
            self.var_bounds.append((0.0, None))  # Non-negative continuous

        # Build objective coefficients
        self.c = np.zeros(self.n)
        self.Q = np.zeros((self.n, self.n))
        
        # Construction costs (per edge, based on distance)
        for idx, (i, j) in enumerate(self.edge_list):
            dist_km = self.distance_matrix[i, j]
            self.c[idx] = dist_km * self.build_cost_per_km
        
        # Power loss costs (Quadratic: loss_factor * resistance * flow^2)
        # Loss = I^2 * R = (P/V)^2 * R = P^2 * (R/V^2)
        # Cost = LossRate * Loss = LossRate * (R/V^2) * P^2
        # So coeff = LossRate * Resistance / Voltage^2
        loss_factor = self.loss_cost_rate / (self.operating_voltage ** 2)
        
        for idx, (i, j) in enumerate(self.flow_list):
            resistance = self.resistance_matrix[i, j]
            coeff = resistance * loss_factor
            
            # Gurobi standard form minimizes 0.5 * x'Qx + c'x
            # We want to minimize k * x^2
            # So 0.5 * Q_ii = k  =>  Q_ii = 2 * k
            var_idx = self.n_edges + idx
            self.Q[var_idx, var_idx] = 2.0 * coeff

        # Build constraints
        self.A = np.zeros((self.m, self.n))
        self.b = np.zeros(self.m)
        self.senses = []
        self.constr_names = []
        
        row_idx = 0

        # Constraint 1: Flow conservation at each node
        # For generators: outflow - inflow <= capacity
        # For consumers: inflow - outflow >= demand
        # Combined: sum_j(Flow_j_i) - sum_j(Flow_i_j) + generation_i = demand_i
        for node in range(self.n_nodes):
            # Inflows to this node
            for other in range(self.n_nodes):
                if other != node:
                    col = get_idx_flow(other, node)
                    self.A[row_idx, col] = 1.0  # Inflow
            
            # Outflows from this node
            for other in range(self.n_nodes):
                if other != node:
                    col = get_idx_flow(node, other)
                    self.A[row_idx, col] = -1.0  # Outflow
            
            if self.node_types[node] == 'generator':
                # Net outflow <= capacity (can generate up to capacity)
                # -inflow + outflow <= capacity → inflow - outflow >= -capacity
                self.b[row_idx] = -self.generator_capacities[node]
                self.senses.append(">=")
                self.constr_names.append(f"Gen_Cap_{node}")
            else:
                # Net inflow >= demand (must receive at least demand)
                self.b[row_idx] = self.power_demands[node]
                self.senses.append(">=")
                self.constr_names.append(f"Demand_{node}")
            
            row_idx += 1

        # Constraint 2: Capacity constraints
        # Flow_i_j <= line_capacity × Build_i_j (or Build_j_i if j < i)
        for idx, (i, j) in enumerate(self.flow_list):
            col_flow = self.n_edges + idx
            col_build = get_idx_build(i, j)
            
            self.A[row_idx, col_flow] = 1.0
            self.A[row_idx, col_build] = -self.line_capacity
            self.b[row_idx] = 0.0
            self.senses.append("<=")
            self.constr_names.append(f"Cap_{i}_{j}")
            row_idx += 1

    def _get_instance_data(self):
        """Return instance-specific data for JSON serialization."""
        return {
            "n_nodes": self.n_nodes,
            "n_generators": self.n_generators,
            "n_consumers": self.n_consumers,
            "node_coordinates": self.node_coords,
            "node_types": self.node_types,
            "power_demands": self.power_demands.tolist() if isinstance(self.power_demands, np.ndarray) else self.power_demands,
            "generator_capacities": self.generator_capacities.tolist() if isinstance(self.generator_capacities, np.ndarray) else self.generator_capacities,
            "wire_resistivity": self.resistivity,  # Ω·m (copper)
            "wire_cross_section": self.wire_area,  # mm²
            "operating_voltage": self.operating_voltage,  # kV
            "line_capacity": self.line_capacity,  # MW
            "build_cost_per_km": self.build_cost_per_km,  # $/km
            "loss_cost_rate": self.loss_cost_rate,  # $/MWh equivalent
            # Note: distance_matrix and resistance_matrix are intentionally NOT included
            # The LLM must compute them from coordinates and wire properties
        }

    def generate_gurobi_code_reference(self, data: dict) -> str:
        """
        Generate a Gurobi script that computes resistances from physical parameters.
        
        This demonstrates the 'induced' nature of the problem - the resistance matrix
        is computed within the code using Ohm's law, not passed directly.
        """
        instance_data = data.get("instance_data", {})
        meta = data.get("meta", {})

        n_nodes = instance_data.get("n_nodes", 0)
        n_generators = instance_data.get("n_generators", 0)
        node_coords = instance_data.get("node_coordinates", [])
        node_types = instance_data.get("node_types", [])
        power_demands = instance_data.get("power_demands", [])
        generator_capacities = instance_data.get("generator_capacities", [])
        resistivity = instance_data.get("wire_resistivity", self.COPPER_RESISTIVITY)
        wire_area = instance_data.get("wire_cross_section", 500.0)
        operating_voltage = instance_data.get("operating_voltage", 220.0)
        line_capacity = instance_data.get("line_capacity", 100.0)
        build_cost_per_km = instance_data.get("build_cost_per_km", 100000.0)
        loss_cost_rate = instance_data.get("loss_cost_rate", 50.0)

        goal = meta.get("goal", "MINIMIZE")
        sense = "GRB.MINIMIZE" if goal == "MINIMIZE" else "GRB.MAXIMIZE"

        lines = []
        lines.append("```python")
        lines.append("import math")
        lines.append("from gurobipy import *")
        lines.append("")
        lines.append("def solve_problem():")
        lines.append(f"    m = Model(\"Power_Transmission_Network\")")
        lines.append("    m.setParam(\"TimeLimit\", 300)")
        lines.append("")
        lines.append("    # Problem Data")
        lines.append(f"    n_nodes = {n_nodes}")
        lines.append(f"    nodes = list(range(n_nodes))")
        lines.append(f"    n_generators = {n_generators}")
        lines.append("")
        lines.append(f"    # Node coordinates (x, y) in km")
        lines.append(f"    node_coords = {node_coords}")
        lines.append("")
        lines.append(f"    # Node types")
        lines.append(f"    node_types = {node_types}")
        lines.append("")
        lines.append(f"    # Power demands and generator capacities (MW)")
        lines.append(f"    power_demands = {power_demands}")
        lines.append(f"    generator_capacities = {generator_capacities}")
        lines.append("")
        lines.append("    # Wire physical properties (INDUCED PARAMETERS)")
        lines.append(f"    resistivity = {resistivity}  # Ohm-meter")
        lines.append(f"    wire_area_mm2 = {wire_area}  # mm^2")
        lines.append(f"    wire_area_m2 = wire_area_mm2 * 1e-6  # Convert to m^2")
        lines.append("")
        lines.append("    # Operating parameters")
        lines.append(f"    operating_voltage = {operating_voltage}  # kV")
        lines.append(f"    line_capacity = {line_capacity}  # MW")
        lines.append(f"    build_cost_per_km = {build_cost_per_km}  # $/km")
        lines.append(f"    loss_cost_rate = {loss_cost_rate}  # $/MWh equivalent")
        lines.append("")
        lines.append("    # COMPUTE distance and resistance matrices (INDUCED COSTS)")
        lines.append("    # Using Ohm's Law: R = resistivity * length / area")
        lines.append("    def compute_distance_km(coord1, coord2):")
        lines.append("        return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)")
        lines.append("")
        lines.append("    def compute_resistance(dist_km):")
        lines.append("        dist_m = dist_km * 1000  # Convert km to m")
        lines.append("        return resistivity * dist_m / wire_area_m2")
        lines.append("")
        lines.append("    distances = {}  # km")
        lines.append("    resistances = {}  # Ohm")
        lines.append("    for i in nodes:")
        lines.append("        for j in nodes:")
        lines.append("            if i != j:")
        lines.append("                dist = compute_distance_km(node_coords[i], node_coords[j])")
        lines.append("                distances[(i, j)] = dist")
        lines.append("                resistances[(i, j)] = compute_resistance(dist)")
        lines.append("")
        lines.append("    # Create edge list (undirected, i < j)")
        lines.append("    edges = [(i, j) for i in nodes for j in nodes if i < j]")
        lines.append("")
        lines.append("    # Decision Variables")
        lines.append("    # Build_i_j: Binary - whether to build line between nodes i and j")
        lines.append("    build = {}")
        lines.append("    for (i, j) in edges:")
        lines.append("        build[(i, j)] = m.addVar(vtype=GRB.BINARY, name=f\"Build_{i}_{j}\")")
        lines.append("")
        lines.append("    # Flow_i_j: Continuous - power flow from node i to j (MW)")
        lines.append("    flow = {}")
        lines.append("    for i in nodes:")
        lines.append("        for j in nodes:")
        lines.append("            if i != j:")
        lines.append("                flow[(i, j)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f\"Flow_{i}_{j}\")")
        lines.append("")
        lines.append("    # Helper function to get build variable (handles i > j case)")
        lines.append("    def get_build(i, j):")
        lines.append("        if i < j:")
        lines.append("            return build[(i, j)]")
        lines.append("        else:")
        lines.append("            return build[(j, i)]")
        lines.append("")
        lines.append("    # Objective: Minimize construction cost + operational transmission cost")
        lines.append("    # Cost factor: converts resistance to cost coefficient")
        lines.append("    # Cost = LossRate * (R / V^2) * P^2")
        lines.append("    # factor = LossRate / V^2")
        lines.append("    loss_factor = loss_cost_rate / (operating_voltage ** 2)")
        lines.append("")
        lines.append("    # Construction costs (based on distance)")
        lines.append("    obj = quicksum(distances[(i, j)] * build_cost_per_km * build[(i, j)] for (i, j) in edges)")
        lines.append("")
        lines.append("    # Operational costs: Quadratic (I^2*R losses)")
        lines.append("    # Note: Gurobi supports quadratic objectives directly")
        lines.append("    obj += quicksum(")
        lines.append("        resistances[(i, j)] * loss_factor * flow[(i, j)] * flow[(i, j)]")
        lines.append("        for i in nodes for j in nodes if i != j")
        lines.append("    )")
        lines.append(f"    m.setObjective(obj, {sense})")
        lines.append("")
        lines.append("    # Flow conservation constraints")
        lines.append("    for node in nodes:")
        lines.append("        inflow = quicksum(flow[(other, node)] for other in nodes if other != node)")
        lines.append("        outflow = quicksum(flow[(node, other)] for other in nodes if other != node)")
        lines.append("        ")
        lines.append("        if node_types[node] == 'generator':")
        lines.append("            # Generator: net outflow (generation) <= capacity")
        lines.append("            # Rewritten: inflow - outflow >= -capacity")
        lines.append("            m.addConstr(")
        lines.append("                inflow - outflow >= -generator_capacities[node],")
        lines.append("                name=f\"Gen_Cap_{node}\"")
        lines.append("            )")
        lines.append("        else:")
        lines.append("            # Consumer: net inflow >= demand")
        lines.append("            m.addConstr(")
        lines.append("                inflow - outflow >= power_demands[node],")
        lines.append("                name=f\"Demand_{node}\"")
        lines.append("            )")
        lines.append("")
        lines.append("    # Capacity constraints: Flow <= line_capacity * Build")
        lines.append("    for i in nodes:")
        lines.append("        for j in nodes:")
        lines.append("            if i != j:")
        lines.append("                m.addConstr(")
        lines.append("                    flow[(i, j)] <= line_capacity * get_build(i, j),")
        lines.append("                    name=f\"Cap_{i}_{j}\"")
        lines.append("                )")
        lines.append("")
        lines.append("    m.optimize()")
        lines.append("    return m")
        lines.append("```")

        return "\n".join(lines)

    @classmethod
    def build_prompt_messages(cls, problem_data: dict) -> list[dict]:
        """Build prompt messages emphasizing the need to compute resistances."""
        import json

        clean_data = cls._sanitize_for_prompt(problem_data)
        system_prompt = cls.get_system_prompt()

        instance_data = clean_data.get("instance_data", {})
        meta = clean_data.get("meta", {})

        n_nodes = instance_data.get("n_nodes", 0)
        n_generators = instance_data.get("n_generators", 0)
        node_coords = instance_data.get("node_coordinates", [])
        node_types = instance_data.get("node_types", [])
        power_demands = instance_data.get("power_demands", [])
        generator_capacities = instance_data.get("generator_capacities", [])
        resistivity = instance_data.get("wire_resistivity", cls.COPPER_RESISTIVITY)
        wire_area = instance_data.get("wire_cross_section", 500.0)
        operating_voltage = instance_data.get("operating_voltage", 220.0)
        line_capacity = instance_data.get("line_capacity", 100.0)
        build_cost_per_km = instance_data.get("build_cost_per_km", 100000.0)
        loss_cost_rate = instance_data.get("loss_cost_rate", 50.0)

        # Format abbreviated data preview
        max_preview = 3
        node_preview = node_coords[:max_preview]
        
        node_coords_abbrev = ", ".join(
            f"N{i}: ({x}, {y})" for i, (x, y) in enumerate(node_preview)
        )
        if n_nodes > max_preview:
            node_coords_abbrev += f", ... ({n_nodes} total)"

        def abbrev_list(data, max_items=3):
            if len(data) <= max_items:
                return str(data)
            preview = data[:max_items]
            return f"{preview}... ({len(data)} total)"

        user_content = f"""Write an engineering memo for a power transmission network design problem.

**Scenario:**
A power utility needs to design a transmission network connecting power generators to consumers.

**Problem Scale:**
- {n_nodes} nodes total ({n_generators} generators, {n_nodes - n_generators} consumers)
- Locations given as map coordinates (km)
- Using copper wire (resistivity affects power loss)

**Sample Data Preview (abbreviated - DO NOT copy these values into your memo):**
- Node coords: {node_coords_abbrev}
- Node types: {abbrev_list(node_types)}
- Power demands (MW): {abbrev_list(power_demands)}
- Generator capacities (MW): {abbrev_list(generator_capacities)}
- Wire area: {wire_area} mm², Resistivity: {resistivity:.2e} Ω·m
- Operating voltage: {operating_voltage} kV, Line capacity: {line_capacity} MW
- Build cost: ${build_cost_per_km:,.0f}/km, Loss cost rate: ${loss_cost_rate}/MWh

**Business Goal:** Minimize total cost (construction + power losses)

**Key Engineering Rules:**
- Power can only flow through built transmission lines
- Each consumer must receive enough power to meet their demand
- Generators have maximum output limits
- Power loss creates a cost that increases with the SQUARE of the power flow ($I^2R$)
- Operational cost = CostRate * Resistance * (Power / Voltage)^2
- Higher voltage reduces this cost significantly ($1/V^2$ factor)
- Wire resistance depends on length and cross-section (using copper wire)
- For planning purposes, assume power flow is conserved (losses are a separate financial cost, not a physical deduction).

**Your Task:**
Write a natural engineering memo that describes this problem.

IMPORTANT: Use ONLY these placeholders for data (do NOT write out sample values):
{{NODE_COORDINATES}}, {{NODE_TYPES}}, {{POWER_DEMANDS}}, {{GENERATOR_CAPACITIES}}, {{WIRE_RESISTIVITY}}, {{WIRE_CROSS_SECTION}}, {{OPERATING_VOLTAGE}}, {{LINE_CAPACITY}}, {{BUILD_COST_PER_KM}}, {{LOSS_COST_RATE}}

Each placeholder should appear exactly ONCE. The placeholders will be replaced with actual data later.
Do NOT duplicate data by writing sample values AND using placeholders - just use the placeholders.

Write it as a real engineering problem, not a physics specification. The reader should understand that:
- Doubling the power flow quadruples the cost of losses (quadratic relationship)
- We need to minimize the combined construction and quadratic operational costs
- Don't simply sum flows; use the proper physics model for costs
- Use language like "resistive losses", "heating costs", and "square law"
"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    @classmethod
    def finish_description(cls, llm_description: str, problem_data: dict) -> str | None:
        """Fill in template placeholders with actual instance data."""
        if not llm_description:
            return llm_description

        required_placeholders = [
            "NODE_COORDINATES", "NODE_TYPES", "POWER_DEMANDS", "GENERATOR_CAPACITIES",
            "WIRE_RESISTIVITY", "WIRE_CROSS_SECTION", "OPERATING_VOLTAGE", 
            "LINE_CAPACITY", "BUILD_COST_PER_KM", "LOSS_COST_RATE"
        ]

        # Check for placeholders
        has_placeholders = any(f"{{{p}}}" in llm_description for p in required_placeholders)
        if not has_placeholders:
            return llm_description

        instance_data = problem_data.get("instance_data", {})
        if not instance_data:
            return llm_description

        # Check for missing placeholders
        missing_placeholders = [p for p in required_placeholders if f"{{{p}}}" not in llm_description]
        if missing_placeholders:
            print(f"❌ Error: Missing required placeholders: {missing_placeholders}")
            return None

        # Format data for insertion
        formatted_data = {}

        # Format node coordinates
        node_coords = instance_data.get("node_coordinates", [])
        node_types = instance_data.get("node_types", [])
        lines = ["Node Coordinates (km):"]
        for i, (x, y) in enumerate(node_coords):
            node_type = node_types[i] if i < len(node_types) else "unknown"
            lines.append(f"  - N{i} ({node_type}): ({x}, {y})")
        formatted_data["NODE_COORDINATES"] = "\n".join(lines)

        # Format node types
        formatted_data["NODE_TYPES"] = str(node_types)

        # Format power data
        power_demands = instance_data.get("power_demands", [])
        demands_str = []
        for i, d in enumerate(power_demands):
            if d > 0:
                demands_str.append(f"N{i}: {d} MW")
        formatted_data["POWER_DEMANDS"] = "Consumer demands: " + ", ".join(demands_str) if demands_str else str(power_demands)

        gen_caps = instance_data.get("generator_capacities", [])
        caps_str = []
        for i, c in enumerate(gen_caps):
            if c > 0:
                caps_str.append(f"N{i}: {c} MW")
        formatted_data["GENERATOR_CAPACITIES"] = "Generator capacities: " + ", ".join(caps_str) if caps_str else str(gen_caps)

        # Format wire properties (always copper)
        resistivity = instance_data.get("wire_resistivity", cls.COPPER_RESISTIVITY)
        formatted_data["WIRE_RESISTIVITY"] = f"{resistivity:.2e} Ω·m (copper)"
        formatted_data["WIRE_CROSS_SECTION"] = f"{instance_data.get('wire_cross_section', 500.0)} mm²"

        # Format operating parameters
        formatted_data["OPERATING_VOLTAGE"] = f"{instance_data.get('operating_voltage', 220.0)} kV"
        formatted_data["LINE_CAPACITY"] = f"{instance_data.get('line_capacity', 100.0)} MW"
        formatted_data["BUILD_COST_PER_KM"] = f"${instance_data.get('build_cost_per_km', 100000.0):,.0f}/km"
        formatted_data["LOSS_COST_RATE"] = f"${instance_data.get('loss_cost_rate', 50.0)}/MWh"

        # Fill in placeholders
        result = llm_description
        for placeholder, value in formatted_data.items():
            result = result.replace(f"{{{placeholder}}}", value)

        # Verify all placeholders were filled
        remaining = [p for p in required_placeholders if f"{{{p}}}" in result]
        if remaining:
            print(f"❌ Error: Placeholders not filled: {remaining}")
            return None

        return result
