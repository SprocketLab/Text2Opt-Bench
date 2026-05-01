import numpy as np
from typing import List, Tuple, Optional
from main.generation.base_problem import BaseProblem


class FacilityLocationGenerator(BaseProblem):
    """
    Facility Location Problem with Distance-Based Costs (MILP).
    
    This problem features INDUCED CONSTRAINTS/COSTS where the transport costs
    are NOT directly given but must be COMPUTED from:
    - Facility coordinates (x, y)
    - Customer coordinates (x, y)
    - Transport cost rate ($/km/unit)
    
    The LLM must compute: distance_ij = sqrt((fx_i - cx_j)^2 + (fy_i - cy_j)^2)
    And then: transport_cost_ij = distance_ij * rate
    
    Decision Variables:
    - Open_F{i}: Binary - whether facility i is opened
    - Ship_F{i}_C{j}: Continuous - units shipped from facility i to customer j
    
    Constraints:
    1. Demand Satisfaction: sum_i Ship_F{i}_C{j} >= Demand_j (for each customer)
    2. Capacity Constraint: sum_j Ship_F{i}_C{j} <= Capacity_i * Open_F{i} (for each facility)
    
    Objective (MINIMIZE):
    Fixed opening costs + Transport costs
    = sum_i (FixedCost_i * Open_F{i}) + sum_ij (distance_ij * rate * Ship_F{i}_C{j})
    """

    SYSTEM_PROMPT = """You are a Supply Chain Consultant writing a business memo about a facility location decision.

**TASK:** Create a natural, human-like problem description. Write it as if a business client is explaining their logistics challenge, NOT as a technical specification.

**TONE:** Conversational business memo. The reader should understand the problem intuitively without mathematical formulas.

**REQUIRED PLACEHOLDERS (include ALL exactly once):**
{FACILITY_COORDINATES}, {CUSTOMER_COORDINATES}, {FACILITY_CAPACITIES}, {CUSTOMER_DEMANDS}, {FIXED_COSTS}, {TRANSPORT_RATE}

**CRITICAL - DATA HANDLING:**
- Use ONLY the placeholders above for data. Do NOT write out sample values or example data.
- Each placeholder should appear EXACTLY ONCE in the description.
- The placeholders will be replaced with actual data later - do not duplicate by also writing sample values.
- WRONG: "Here are the coordinates: F0: (49, 64), F1: (89, 54)... {FACILITY_COORDINATES}"
- CORRECT: "The facility locations are provided in {FACILITY_COORDINATES}"

**WHAT TO INCLUDE:**
- We have candidate facility locations (given as map coordinates)
- We have customer locations (given as map coordinates)  
- Opening a facility has a fixed cost
- Shipping goods costs money based on distance traveled and volume shipped
- Each facility has limited capacity
- Each customer has demand that must be satisfied
- Goal: Minimize total cost (opening + shipping)

**DISTANCE CLARIFICATION (MUST INCLUDE):**
- You MUST state clearly that distances between facilities and customers are Euclidean (straight-line) distances computed from the map coordinates.
- Example phrasing: "Distances between facilities and customers are calculated as straight-line (Euclidean) distances from the map coordinates."
- This is important so the reader knows exactly how to measure distance — do NOT leave this ambiguous.

**WHAT TO AVOID:**
- Do NOT write mathematical formulas or give the sqrt formula
- Do NOT use technical terms like "induced constraints" or "derived costs"
- Do NOT list out sample data values - only use placeholders

The description should read like a real business problem that clearly states Euclidean distance is the distance metric."""

    def __init__(
        self,
        n_facilities: int = 3,
        n_customers: int = 4,
        coord_range: Tuple[float, float] = (0.0, 100.0),
        capacity_range: Tuple[float, float] = (50.0, 200.0),
        demand_range: Tuple[float, float] = (10.0, 50.0),
        fixed_cost_range: Tuple[float, float] = (100.0, 500.0),
        transport_rate: float = 0.5,  # $/km/unit
        prob_type: str = "LP",
        goal: str = "MINIMIZE",
    ):
        self.n_facilities = n_facilities
        self.n_customers = n_customers
        self.coord_range = coord_range
        self.capacity_range = capacity_range
        self.demand_range = demand_range
        self.fixed_cost_range = fixed_cost_range
        self.transport_rate = transport_rate

        # Variable counts
        # Open_F{i}: n_facilities binary variables
        # Ship_F{i}_C{j}: n_facilities * n_customers continuous variables
        self.num_open = n_facilities
        self.num_ship = n_facilities * n_customers
        n_vars = self.num_open + self.num_ship

        # Constraint counts
        # Demand satisfaction: n_customers
        # Capacity: n_facilities
        n_constrs = n_customers + n_facilities

        super().__init__(
            n_vars,
            n_constrs,
            prob_type=prob_type,
            goal=goal,
            problem_type="facility location"
        )

        # Instance data placeholders
        self.facility_coords = None  # [(x, y), ...]
        self.customer_coords = None  # [(x, y), ...]
        self.facility_capacities = None  # [cap_0, cap_1, ...]
        self.customer_demands = None  # [demand_0, demand_1, ...]
        self.fixed_costs = None  # [fixed_0, fixed_1, ...]
        self.distance_matrix = None  # Computed from coordinates

    def generate(self, time_limit: float = 300.0):
        """Generate the facility location problem instance."""
        self._generate_instance_data()
        self._compute_distance_matrix()
        self._build_matrices()
        self.solve(time_limit=time_limit)
        return self

    def _generate_instance_data(self):
        """Generate random facility and customer locations and parameters.
        
        Note: We use integer coordinates to avoid floating point precision issues
        between the internal solver and the gold solution code.
        """
        # Generate facility coordinates as integers to avoid rounding issues
        self.facility_coords = [
            (
                int(np.random.uniform(*self.coord_range)),
                int(np.random.uniform(*self.coord_range))
            )
            for _ in range(self.n_facilities)
        ]

        # Generate customer coordinates as integers
        self.customer_coords = [
            (
                int(np.random.uniform(*self.coord_range)),
                int(np.random.uniform(*self.coord_range))
            )
            for _ in range(self.n_customers)
        ]

        # Generate capacities (ensure total capacity > total demand for feasibility)
        # Use integers to avoid precision issues
        self.customer_demands = np.random.randint(
            int(self.demand_range[0]), int(self.demand_range[1]) + 1, size=self.n_customers
        ).astype(float)
        total_demand = np.sum(self.customer_demands)

        # Ensure sufficient capacity (at least 1.3x total demand distributed across facilities)
        min_total_capacity = total_demand * 1.3
        avg_capacity = min_total_capacity / self.n_facilities
        self.facility_capacities = np.random.randint(
            int(max(self.capacity_range[0], avg_capacity * 0.8)),
            int(max(self.capacity_range[1], avg_capacity * 1.5)) + 1,
            size=self.n_facilities
        ).astype(float)

        # Generate fixed opening costs as integers
        self.fixed_costs = np.random.randint(
            int(self.fixed_cost_range[0]), int(self.fixed_cost_range[1]) + 1, size=self.n_facilities
        ).astype(float)

    def _compute_distance_matrix(self):
        """Compute Euclidean distances between all facility-customer pairs.
        
        No rounding is applied to ensure exact match with gold solution code.
        """
        self.distance_matrix = np.zeros((self.n_facilities, self.n_customers))
        for i in range(self.n_facilities):
            fx, fy = self.facility_coords[i]
            for j in range(self.n_customers):
                cx, cy = self.customer_coords[j]
                dist = np.sqrt((fx - cx) ** 2 + (fy - cy) ** 2)
                self.distance_matrix[i, j] = dist  # No rounding

    def _build_matrices(self):
        """Build the constraint matrix A, RHS b, and objective c."""
        # Variable ordering:
        # [Open_F0, Open_F1, ..., Open_F{n-1}, Ship_F0_C0, Ship_F0_C1, ..., Ship_F{n-1}_C{m-1}]

        idx_open = 0
        idx_ship = self.num_open

        def get_idx_open(i):
            return idx_open + i

        def get_idx_ship(i, j):
            return idx_ship + i * self.n_customers + j

        # Build variable names
        self.var_names = []
        for i in range(self.n_facilities):
            self.var_names.append(f"Open_F{i}")
        for i in range(self.n_facilities):
            for j in range(self.n_customers):
                self.var_names.append(f"Ship_F{i}_C{j}")

        # Build variable types
        self.var_types = []
        for _ in range(self.num_open):
            self.var_types.append("Binary")
        for _ in range(self.num_ship):
            self.var_types.append("Continuous")

        # Build variable bounds
        self.var_bounds = []
        for _ in range(self.num_open):
            self.var_bounds.append((0.0, 1.0))  # Binary
        for _ in range(self.num_ship):
            self.var_bounds.append((0.0, None))  # Non-negative continuous

        # Build objective coefficients
        self.c = np.zeros(self.n)
        # Fixed costs for opening facilities
        for i in range(self.n_facilities):
            self.c[get_idx_open(i)] = self.fixed_costs[i]
        # Transport costs (distance * rate) - no rounding for exact match
        for i in range(self.n_facilities):
            for j in range(self.n_customers):
                idx = get_idx_ship(i, j)
                self.c[idx] = self.distance_matrix[i, j] * self.transport_rate

        # Build constraints
        self.A = np.zeros((self.m, self.n))
        self.b = np.zeros(self.m)
        self.senses = []
        self.constr_names = []

        row_idx = 0

        # Constraint 1: Demand satisfaction
        # sum_i Ship_F{i}_C{j} >= Demand_j
        for j in range(self.n_customers):
            for i in range(self.n_facilities):
                col = get_idx_ship(i, j)
                self.A[row_idx, col] = 1.0
            self.b[row_idx] = self.customer_demands[j]
            self.senses.append(">=")
            self.constr_names.append(f"Demand_C{j}")
            row_idx += 1

        # Constraint 2: Capacity constraint
        # sum_j Ship_F{i}_C{j} <= Capacity_i * Open_F{i}
        # Rewritten: sum_j Ship_F{i}_C{j} - Capacity_i * Open_F{i} <= 0
        for i in range(self.n_facilities):
            for j in range(self.n_customers):
                col = get_idx_ship(i, j)
                self.A[row_idx, col] = 1.0
            col_open = get_idx_open(i)
            self.A[row_idx, col_open] = -self.facility_capacities[i]
            self.b[row_idx] = 0.0
            self.senses.append("<=")
            self.constr_names.append(f"Capacity_F{i}")
            row_idx += 1

    def _get_instance_data(self):
        """Return instance-specific data for JSON serialization."""
        return {
            "n_facilities": self.n_facilities,
            "n_customers": self.n_customers,
            "facility_coordinates": self.facility_coords,
            "customer_coordinates": self.customer_coords,
            "facility_capacities": self.facility_capacities.tolist() if isinstance(self.facility_capacities, np.ndarray) else self.facility_capacities,
            "customer_demands": self.customer_demands.tolist() if isinstance(self.customer_demands, np.ndarray) else self.customer_demands,
            "fixed_costs": self.fixed_costs.tolist() if isinstance(self.fixed_costs, np.ndarray) else self.fixed_costs,
            "transport_rate": self.transport_rate,
            # Note: distance_matrix is intentionally NOT included
            # The LLM must compute it from coordinates
        }

    def generate_gurobi_code_reference(self, data: dict) -> str:
        """
        Generate a Gurobi script that computes distances from coordinates.
        
        This demonstrates the 'induced' nature of the problem - the distance matrix
        is computed within the code, not passed directly.
        """
        instance_data = data.get("instance_data", {})
        meta = data.get("meta", {})

        n_facilities = instance_data.get("n_facilities", 0)
        n_customers = instance_data.get("n_customers", 0)
        facility_coords = instance_data.get("facility_coordinates", [])
        customer_coords = instance_data.get("customer_coordinates", [])
        facility_capacities = instance_data.get("facility_capacities", [])
        customer_demands = instance_data.get("customer_demands", [])
        fixed_costs = instance_data.get("fixed_costs", [])
        transport_rate = instance_data.get("transport_rate", 0.5)

        goal = meta.get("goal", "MINIMIZE")
        sense = "GRB.MINIMIZE" if goal == "MINIMIZE" else "GRB.MAXIMIZE"

        lines = []
        lines.append("```python")
        lines.append("import math")
        lines.append("from gurobipy import *")
        lines.append("")
        lines.append("def solve_problem():")
        lines.append(f"    m = Model(\"Facility_Location\")")
        lines.append("    m.setParam(\"TimeLimit\", 300)")
        lines.append("")
        lines.append("    # Problem Data")
        lines.append(f"    facilities = list(range({n_facilities}))  # F0..F{n_facilities-1}")
        lines.append(f"    customers = list(range({n_customers}))  # C0..C{n_customers-1}")
        lines.append("")
        lines.append(f"    # Facility coordinates (x, y)")
        lines.append(f"    facility_coords = {facility_coords}")
        lines.append("")
        lines.append(f"    # Customer coordinates (x, y)")
        lines.append(f"    customer_coords = {customer_coords}")
        lines.append("")
        lines.append(f"    facility_capacities = {facility_capacities}")
        lines.append(f"    customer_demands = {customer_demands}")
        lines.append(f"    fixed_costs = {fixed_costs}")
        lines.append(f"    transport_rate = {transport_rate}  # $/km/unit")
        lines.append("")
        lines.append("    # COMPUTE distance matrix from coordinates (INDUCED COSTS)")
        lines.append("    def euclidean_distance(coord1, coord2):")
        lines.append("        return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)")
        lines.append("")
        lines.append("    distances = {}")
        lines.append("    for i in facilities:")
        lines.append("        for j in customers:")
        lines.append("            distances[(i, j)] = euclidean_distance(facility_coords[i], customer_coords[j])")
        lines.append("")
        lines.append("    # Decision Variables")
        lines.append("    # Open_F{i}: Binary - whether facility i is opened")
        lines.append("    open_facility = {}")
        lines.append("    for i in facilities:")
        lines.append("        open_facility[i] = m.addVar(vtype=GRB.BINARY, name=f\"Open_F{i}\")")
        lines.append("")
        lines.append("    # Ship_F{i}_C{j}: Continuous - units shipped from facility i to customer j")
        lines.append("    ship = {}")
        lines.append("    for i in facilities:")
        lines.append("        for j in customers:")
        lines.append("            ship[(i, j)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f\"Ship_F{i}_C{j}\")")
        lines.append("")
        lines.append("    # Objective: Minimize fixed costs + transport costs")
        lines.append("    # Transport cost = distance * rate * units shipped")
        lines.append("    obj = quicksum(fixed_costs[i] * open_facility[i] for i in facilities)")
        lines.append("    obj += quicksum(")
        lines.append("        distances[(i, j)] * transport_rate * ship[(i, j)]")
        lines.append("        for i in facilities for j in customers")
        lines.append("    )")
        lines.append(f"    m.setObjective(obj, {sense})")
        lines.append("")
        lines.append("    # Demand satisfaction constraints")
        lines.append("    for j in customers:")
        lines.append("        m.addConstr(")
        lines.append("            quicksum(ship[(i, j)] for i in facilities) >= customer_demands[j],")
        lines.append("            name=f\"Demand_C{j}\"")
        lines.append("        )")
        lines.append("")
        lines.append("    # Capacity constraints (can only ship if facility is open)")
        lines.append("    for i in facilities:")
        lines.append("        m.addConstr(")
        lines.append("            quicksum(ship[(i, j)] for j in customers) <= facility_capacities[i] * open_facility[i],")
        lines.append("            name=f\"Capacity_F{i}\"")
        lines.append("        )")
        lines.append("")
        lines.append("    m.optimize()")
        lines.append("    return m")
        lines.append("```")

        return "\n".join(lines)

    @classmethod
    def build_prompt_messages(cls, problem_data: dict) -> list[dict]:
        """Build prompt messages emphasizing the need to compute distances."""
        import json

        clean_data = cls._sanitize_for_prompt(problem_data)
        system_prompt = cls.get_system_prompt()

        instance_data = clean_data.get("instance_data", {})
        meta = clean_data.get("meta", {})

        n_facilities = instance_data.get("n_facilities", 0)
        n_customers = instance_data.get("n_customers", 0)
        facility_coords = instance_data.get("facility_coordinates", [])
        customer_coords = instance_data.get("customer_coordinates", [])
        facility_capacities = instance_data.get("facility_capacities", [])
        customer_demands = instance_data.get("customer_demands", [])
        fixed_costs = instance_data.get("fixed_costs", [])
        transport_rate = instance_data.get("transport_rate", 0.5)

        # Format ABBREVIATED coordinates (first 2 items only)
        max_preview = 2
        facility_preview = facility_coords[:max_preview]
        customer_preview = customer_coords[:max_preview]
        
        facility_coords_abbrev = ", ".join(
            f"F{i}: ({x}, {y})" for i, (x, y) in enumerate(facility_preview)
        )
        if n_facilities > max_preview:
            facility_coords_abbrev += f", ... ({n_facilities} total)"
            
        customer_coords_abbrev = ", ".join(
            f"C{j}: ({x}, {y})" for j, (x, y) in enumerate(customer_preview)
        )
        if n_customers > max_preview:
            customer_coords_abbrev += f", ... ({n_customers} total)"
        
        # Abbreviated lists for other data
        def abbrev_list(data, max_items=3):
            if len(data) <= max_items:
                return str(data)
            preview = data[:max_items]
            return f"{preview}... ({len(data)} total)"

        user_content = f"""Write a business memo for a facility location decision problem.

**Scenario:**
A company needs to decide which warehouses to open and how to ship products to customers.

**Problem Scale:**
- {n_facilities} candidate facility locations (with map coordinates)
- {n_customers} customer locations (with map coordinates)
- Transport rate: ${transport_rate} per km per unit shipped

**Sample Data Preview (abbreviated - DO NOT copy these values into your memo):**
- Facility coords: {facility_coords_abbrev}
- Customer coords: {customer_coords_abbrev}
- Capacities: {abbrev_list(facility_capacities)}
- Demands: {abbrev_list(customer_demands)}
- Fixed costs: {abbrev_list(fixed_costs)}

**Business Goal:** Minimize total cost (facility opening costs + shipping costs)

**Key Business Rules:**
- A facility can only ship if it's opened
- Each customer's demand must be fully satisfied
- Shipping cost depends on distance traveled and volume shipped
- Distances are Euclidean (straight-line) distances from map coordinates

**Your Task:**
Write a natural business memo that describes this problem. 

IMPORTANT: Use ONLY these placeholders for data (do NOT write out sample values):
{{FACILITY_COORDINATES}}, {{CUSTOMER_COORDINATES}}, {{FACILITY_CAPACITIES}}, {{CUSTOMER_DEMANDS}}, {{FIXED_COSTS}}, {{TRANSPORT_RATE}}

Each placeholder should appear exactly ONCE. The placeholders will be replaced with actual data later.
Do NOT duplicate data by writing sample values AND using placeholders - just use the placeholders.

Write it as a real business problem, not a math specification. You MUST clearly state that distances are Euclidean (straight-line) from the map coordinates — do NOT leave the distance metric ambiguous.
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
            "FACILITY_COORDINATES", "CUSTOMER_COORDINATES", "FACILITY_CAPACITIES",
            "CUSTOMER_DEMANDS", "FIXED_COSTS", "TRANSPORT_RATE"
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

        # Format facility coordinates
        facility_coords = instance_data.get("facility_coordinates", [])
        lines = ["Facility Coordinates:"]
        for i, (x, y) in enumerate(facility_coords):
            lines.append(f"  - F{i}: ({x}, {y})")
        formatted_data["FACILITY_COORDINATES"] = "\n".join(lines)

        # Format customer coordinates
        customer_coords = instance_data.get("customer_coordinates", [])
        lines = ["Customer Coordinates:"]
        for j, (x, y) in enumerate(customer_coords):
            lines.append(f"  - C{j}: ({x}, {y})")
        formatted_data["CUSTOMER_COORDINATES"] = "\n".join(lines)

        # Format other data
        formatted_data["FACILITY_CAPACITIES"] = str(instance_data.get("facility_capacities", []))
        formatted_data["CUSTOMER_DEMANDS"] = str(instance_data.get("customer_demands", []))
        formatted_data["FIXED_COSTS"] = str(instance_data.get("fixed_costs", []))
        formatted_data["TRANSPORT_RATE"] = str(instance_data.get("transport_rate", 0.5))

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
