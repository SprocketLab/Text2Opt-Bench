import numpy as np
from typing import List, Tuple
from main.generation.base_problem import BaseProblem


# ── Modifier catalogue ─────────────────────────────────────────────
# Each modifier is a dict with:
#   "key":         unique identifier
#   "add_constrs": callable(gen) → number of rows added
#   "describe":    callable(gen) → natural-language business-rule string
#
# Constraints are appended to A, b, senses, constr_names by _apply_modifiers().
# ────────────────────────────────────────────────────────────────────

def _mod_max_open(gen):
    """At most K facilities may be opened."""
    K = gen._mod_params["max_open_K"]
    row = gen._next_row()
    for i in range(gen.n_facilities):
        gen.A[row, gen._idx_open(i)] = 1.0
    gen.b[row] = float(K)
    gen.senses.append("<=")
    gen.constr_names.append("MaxOpen")
    return 1

def _desc_max_open(gen):
    K = gen._mod_params["max_open_K"]
    return (f"Due to management oversight capacity, the company can operate "
            f"at most {K} distribution centers at any given time.")

def _mod_budget_cap(gen):
    """Total opening cost <= B."""
    B = gen._mod_params["budget_cap_B"]
    row = gen._next_row()
    for i in range(gen.n_facilities):
        gen.A[row, gen._idx_open(i)] = gen.fixed_costs[i]
    gen.b[row] = float(B)
    gen.senses.append("<=")
    gen.constr_names.append("BudgetCap")
    return 1

def _desc_budget_cap(gen):
    B = gen._mod_params["budget_cap_B"]
    return (f"The capital expenditure budget for facility activation is "
            f"capped at ${B:.0f}. The combined fixed costs of all opened "
            f"facilities must stay within this limit.")

def _mod_min_throughput(gen):
    """Each opened facility must ship at least T units total."""
    T = gen._mod_params["min_throughput_T"]
    count = 0
    for i in range(gen.n_facilities):
        row = gen._next_row()
        for j in range(gen.n_customers):
            gen.A[row, gen._idx_ship(i, j)] = 1.0
        gen.A[row, gen._idx_open(i)] = -float(T)
        gen.b[row] = 0.0
        gen.senses.append(">=")
        gen.constr_names.append(f"MinThru_F{i}")
        count += 1
    return count

def _desc_min_throughput(gen):
    T = gen._mod_params["min_throughput_T"]
    return (f"To justify its operating overhead, every opened facility must "
            f"handle at least {T:.0f} units of total outbound shipments.")

def _mod_facility_linking(gen):
    """If facility A opens, facility B must also open."""
    A, B = gen._mod_params["linking_pair"]
    row = gen._next_row()
    gen.A[row, gen._idx_open(A)] = 1.0
    gen.A[row, gen._idx_open(B)] = -1.0
    gen.b[row] = 0.0
    gen.senses.append("<=")
    gen.constr_names.append(f"Link_F{A}_F{B}")
    return 1

def _desc_facility_linking(gen):
    A, B = gen._mod_params["linking_pair"]
    cA = gen.facility_coords[A]
    cB = gen.facility_coords[B]
    return (f"The facility at location F{A} ({cA[0]}, {cA[1]}) relies on "
            f"the regional hub at F{B} ({cB[0]}, {cB[1]}). "
            f"If F{A} is activated, F{B} must also be activated.")

def _mod_anti_affinity(gen):
    """Facilities A and B cannot both be open."""
    A, B = gen._mod_params["anti_affinity_pair"]
    row = gen._next_row()
    gen.A[row, gen._idx_open(A)] = 1.0
    gen.A[row, gen._idx_open(B)] = 1.0
    gen.b[row] = 1.0
    gen.senses.append("<=")
    gen.constr_names.append(f"AntiAff_F{A}_F{B}")
    return 1

def _desc_anti_affinity(gen):
    A, B = gen._mod_params["anti_affinity_pair"]
    cA = gen.facility_coords[A]
    cB = gen.facility_coords[B]
    return (f"Regulatory restrictions prevent the facilities at "
            f"F{A} ({cA[0]}, {cA[1]}) and F{B} ({cB[0]}, {cB[1]}) "
            f"from operating simultaneously due to overlapping market zones.")

def _mod_demand_cap(gen):
    """No single facility can supply more than P% of total demand."""
    P = gen._mod_params["demand_cap_P"]
    total_demand = sum(gen.customer_demands)
    cap = (P / 100.0) * total_demand
    count = 0
    for i in range(gen.n_facilities):
        row = gen._next_row()
        for j in range(gen.n_customers):
            gen.A[row, gen._idx_ship(i, j)] = 1.0
        gen.b[row] = float(cap)
        gen.senses.append("<=")
        gen.constr_names.append(f"DemCap_F{i}")
        count += 1
    return count

def _desc_demand_cap(gen):
    P = gen._mod_params["demand_cap_P"]
    return (f"To avoid over-reliance on any single facility, no distribution "
            f"center may fulfill more than {P:.0f}% of the total system demand.")


MODIFIER_REGISTRY = {
    "max_open":           (_mod_max_open, _desc_max_open),
    "budget_cap":         (_mod_budget_cap, _desc_budget_cap),
    "min_throughput":     (_mod_min_throughput, _desc_min_throughput),
    "facility_linking":   (_mod_facility_linking, _desc_facility_linking),
    "anti_affinity":      (_mod_anti_affinity, _desc_anti_affinity),
    "demand_cap":         (_mod_demand_cap, _desc_demand_cap),
}


class ModifiedFacilityLocationGenerator(BaseProblem):
    """
    Facility Location with Non-Standard Business Rules (MILP).

    Base formulation is standard facility location (textbook):
    - Open_F{i}: Binary — open facility i
    - Ship_F{i}_C{j}: Continuous — ship from i to j
    - Demand satisfaction, capacity linking, distance-based costs

    TRAP: 2-3 randomly selected "business rules" (modifiers) are added
    as extra constraints. These are described in natural language in the
    problem description. An LLM that pattern-matches to textbook facility
    location will MISS the modifiers and get a wrong objective.
    """

    SYSTEM_PROMPT = """You are a Supply Chain Consultant writing a business memo about a distribution network decision.

**TASK:** Write a natural, business-like problem description. Include ALL business rules and policies described below — missing any rule will lead to an incorrect solution.

**TONE:** Professional business memo. Make the additional policies feel like natural business requirements, not mathematical "extra constraints."

**REQUIRED PLACEHOLDERS (include ALL exactly once):**
{FACILITY_COORDINATES}, {CUSTOMER_COORDINATES}, {FACILITY_CAPACITIES}, {CUSTOMER_DEMANDS}, {FIXED_COSTS}, {TRANSPORT_RATE}

**CRITICAL - DATA HANDLING:**
- Use ONLY the placeholders above for numeric data. Do NOT write out sample values.
- Each placeholder should appear EXACTLY ONCE in an Annex section.

**WHAT TO INCLUDE:**
- The standard logistics challenge (which warehouses to open, how to ship)
- Opening a facility has a fixed cost; shipping depends on distance and volume
- Each facility has limited capacity; each customer has demand
- ALL the additional company policies listed in the scenario below — these are CRITICAL
- Goal: minimize total cost

**WHAT TO AVOID:**
- Mathematical formulas
- Technical terms like "constraints", "objective function", "binary variables"
- Labeling rules as "additional constraints" or "non-standard" — just state them as company policy
- Do NOT say "compute Euclidean distance" or give the sqrt formula
- Do NOT list sample data values

**IMPORTANT:** The additional company policies MUST be included in the body of the memo, not in the annex. They should read as natural business requirements."""

    def __init__(
        self,
        n_facilities: int = 5,
        n_customers: int = 8,
        n_modifiers: int = 3,
        coord_range: Tuple[float, float] = (0.0, 100.0),
        capacity_range: Tuple[float, float] = (50.0, 200.0),
        demand_range: Tuple[float, float] = (10.0, 50.0),
        fixed_cost_range: Tuple[float, float] = (100.0, 500.0),
        transport_rate: float = 0.5,
        prob_type: str = "LP",
        goal: str = "MINIMIZE",
    ):
        self.n_facilities = n_facilities
        self.n_customers = n_customers
        self.n_modifiers_target = n_modifiers
        self.coord_range = coord_range
        self.capacity_range = capacity_range
        self.demand_range = demand_range
        self.fixed_cost_range = fixed_cost_range
        self.transport_rate = transport_rate

        # Variable counts (same as standard facility location)
        self.num_open = n_facilities
        self.num_ship = n_facilities * n_customers
        n_vars = self.num_open + self.num_ship

        # Base constraint count
        base_constrs = n_customers + n_facilities  # demand + capacity

        # Select modifiers and compute total constraint count
        self._select_modifiers()
        extra_constrs = self._count_modifier_constraints()
        n_constrs = base_constrs + extra_constrs

        super().__init__(
            n_vars, n_constrs,
            prob_type=prob_type, goal=goal,
            problem_type="modified facility location",
        )

        # Instance data placeholders
        self.facility_coords = None
        self.customer_coords = None
        self.facility_capacities = None
        self.customer_demands = None
        self.fixed_costs = None
        self.distance_matrix = None
        self._row_cursor = 0  # for modifier constraint building

    def _select_modifiers(self):
        """Pick n_modifiers non-conflicting modifiers."""
        available = list(MODIFIER_REGISTRY.keys())
        np.random.shuffle(available)

        selected = []
        used_types = set()
        self._mod_params = {}

        for mod_key in available:
            if len(selected) >= self.n_modifiers_target:
                break

            # Avoid conflicts
            if mod_key == "max_open" and "budget_cap" in used_types:
                continue
            if mod_key == "budget_cap" and "max_open" in used_types:
                continue
            if mod_key == "facility_linking" and "anti_affinity" in used_types:
                continue
            if mod_key == "anti_affinity" and "facility_linking" in used_types:
                continue
            if mod_key == "min_throughput" and "demand_cap" in used_types:
                continue
            if mod_key == "demand_cap" and "min_throughput" in used_types:
                continue

            selected.append(mod_key)
            used_types.add(mod_key)

        self._selected_modifiers = selected

    def _count_modifier_constraints(self):
        """Count how many constraint rows the selected modifiers will add."""
        count = 0
        for mod_key in self._selected_modifiers:
            if mod_key in ("max_open", "budget_cap", "facility_linking", "anti_affinity"):
                count += 1
            elif mod_key in ("min_throughput", "demand_cap"):
                count += self.n_facilities
        return count

    def _next_row(self):
        """Get the next available constraint row index."""
        row = self._row_cursor
        self._row_cursor += 1
        return row

    def _idx_open(self, i):
        return i

    def _idx_ship(self, i, j):
        return self.num_open + i * self.n_customers + j

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, time_limit: float = 300.0):
        self._generate_instance_data()
        self._compute_distance_matrix()
        self._generate_modifier_params()
        self._build_matrices()
        self.solve(time_limit=time_limit)
        return self

    def _generate_instance_data(self):
        # Same as standard facility location
        self.facility_coords = [
            (int(np.random.uniform(*self.coord_range)),
             int(np.random.uniform(*self.coord_range)))
            for _ in range(self.n_facilities)
        ]
        self.customer_coords = [
            (int(np.random.uniform(*self.coord_range)),
             int(np.random.uniform(*self.coord_range)))
            for _ in range(self.n_customers)
        ]
        self.customer_demands = np.random.randint(
            int(self.demand_range[0]), int(self.demand_range[1]) + 1,
            size=self.n_customers
        ).astype(float)
        total_demand = np.sum(self.customer_demands)

        min_total_cap = total_demand * 1.3
        avg_cap = min_total_cap / self.n_facilities
        self.facility_capacities = np.random.randint(
            int(max(self.capacity_range[0], avg_cap * 0.8)),
            int(max(self.capacity_range[1], avg_cap * 1.5)) + 1,
            size=self.n_facilities
        ).astype(float)

        self.fixed_costs = np.random.randint(
            int(self.fixed_cost_range[0]), int(self.fixed_cost_range[1]) + 1,
            size=self.n_facilities
        ).astype(float)

    def _compute_distance_matrix(self):
        self.distance_matrix = np.zeros((self.n_facilities, self.n_customers))
        for i in range(self.n_facilities):
            fx, fy = self.facility_coords[i]
            for j in range(self.n_customers):
                cx, cy = self.customer_coords[j]
                self.distance_matrix[i, j] = np.sqrt(
                    (fx - cx) ** 2 + (fy - cy) ** 2
                )

    def _generate_modifier_params(self):
        """Generate feasible parameters for each selected modifier."""
        total_demand = float(np.sum(self.customer_demands))
        sorted_costs = sorted(self.fixed_costs)

        for mod_key in self._selected_modifiers:
            if mod_key == "max_open":
                # Allow opening 2 to n-1 facilities
                K = int(np.random.randint(2, self.n_facilities))
                self._mod_params["max_open_K"] = K

            elif mod_key == "budget_cap":
                # Budget allows opening 2-3 cheapest facilities
                n_affordable = np.random.randint(2, min(4, self.n_facilities) + 1)
                B = float(sum(sorted_costs[:n_affordable]) * np.random.uniform(1.05, 1.3))
                self._mod_params["budget_cap_B"] = round(B, 0)

            elif mod_key == "min_throughput":
                # Each open facility must ship at least T units
                avg_demand_per_facility = total_demand / self.n_facilities
                T = round(avg_demand_per_facility * np.random.uniform(0.15, 0.35), 0)
                self._mod_params["min_throughput_T"] = T

            elif mod_key == "facility_linking":
                # Random pair (A, B): if A opens, B must open
                pair = list(np.random.choice(self.n_facilities, 2, replace=False))
                self._mod_params["linking_pair"] = [int(pair[0]), int(pair[1])]

            elif mod_key == "anti_affinity":
                # Random pair that can't both open
                pair = list(np.random.choice(self.n_facilities, 2, replace=False))
                self._mod_params["anti_affinity_pair"] = [int(pair[0]), int(pair[1])]

            elif mod_key == "demand_cap":
                # No facility serves more than P% of total demand
                P = int(np.random.randint(40, 70))
                self._mod_params["demand_cap_P"] = P

    # ------------------------------------------------------------------
    # Build A, b, c
    # ------------------------------------------------------------------

    def _build_matrices(self):
        nf = self.n_facilities
        nc = self.n_customers

        # Variable names
        self.var_names = []
        for i in range(nf):
            self.var_names.append(f"Open_F{i}")
        for i in range(nf):
            for j in range(nc):
                self.var_names.append(f"Ship_F{i}_C{j}")

        # Variable types
        self.var_types = (
            ["Binary"] * self.num_open +
            ["Continuous"] * self.num_ship
        )

        # Variable bounds
        self.var_bounds = []
        for _ in range(self.num_open):
            self.var_bounds.append((0.0, 1.0))
        for _ in range(self.num_ship):
            self.var_bounds.append((0.0, None))

        # Objective
        self.c = np.zeros(self.n)
        for i in range(nf):
            self.c[self._idx_open(i)] = self.fixed_costs[i]
        for i in range(nf):
            for j in range(nc):
                self.c[self._idx_ship(i, j)] = (
                    self.distance_matrix[i, j] * self.transport_rate
                )

        # Constraints
        self.A = np.zeros((self.m, self.n))
        self.b = np.zeros(self.m)
        self.senses = []
        self.constr_names = []
        self._row_cursor = 0

        # C1: Demand satisfaction — sum_i Ship_ij >= Demand_j
        for j in range(nc):
            row = self._next_row()
            for i in range(nf):
                self.A[row, self._idx_ship(i, j)] = 1.0
            self.b[row] = self.customer_demands[j]
            self.senses.append(">=")
            self.constr_names.append(f"Demand_C{j}")

        # C2: Capacity — sum_j Ship_ij <= Cap_i * Open_i
        for i in range(nf):
            row = self._next_row()
            for j in range(nc):
                self.A[row, self._idx_ship(i, j)] = 1.0
            self.A[row, self._idx_open(i)] = -self.facility_capacities[i]
            self.b[row] = 0.0
            self.senses.append("<=")
            self.constr_names.append(f"Capacity_F{i}")

        # C3+: Modifier constraints
        for mod_key in self._selected_modifiers:
            apply_fn, _ = MODIFIER_REGISTRY[mod_key]
            apply_fn(self)

        assert self._row_cursor == self.m, (
            f"Expected {self.m} constraints, built {self._row_cursor}"
        )

    # ------------------------------------------------------------------
    # Instance data
    # ------------------------------------------------------------------

    def _get_instance_data(self):
        return {
            "n_facilities": self.n_facilities,
            "n_customers": self.n_customers,
            "facility_coordinates": self.facility_coords,
            "customer_coordinates": self.customer_coords,
            "facility_capacities": self.facility_capacities.tolist(),
            "customer_demands": self.customer_demands.tolist(),
            "fixed_costs": self.fixed_costs.tolist(),
            "transport_rate": self.transport_rate,
            # Modifier info for gold solution
            "modifiers": self._selected_modifiers,
            "modifier_params": self._mod_params,
        }

    # ------------------------------------------------------------------
    # Gold solution code
    # ------------------------------------------------------------------

    def generate_gurobi_code_reference(self, data: dict) -> str:
        inst = data.get("instance_data", {})
        meta = data.get("meta", {})

        nf = inst.get("n_facilities", 0)
        nc = inst.get("n_customers", 0)
        fc = inst.get("facility_coordinates", [])
        cc = inst.get("customer_coordinates", [])
        caps = inst.get("facility_capacities", [])
        demands = inst.get("customer_demands", [])
        fixed = inst.get("fixed_costs", [])
        rate = inst.get("transport_rate", 0.5)
        mods = inst.get("modifiers", [])
        mod_params = inst.get("modifier_params", {})

        sense = "GRB.MINIMIZE" if meta.get("goal") == "MINIMIZE" else "GRB.MAXIMIZE"

        lines = []
        lines.append("```python")
        lines.append("import math")
        lines.append("from gurobipy import *")
        lines.append("")
        lines.append("def solve_problem():")
        lines.append('    m = Model("Modified_Facility_Location")')
        lines.append("    m.setParam('TimeLimit', 300)")
        lines.append("")
        lines.append(f"    nf = {nf}")
        lines.append(f"    nc = {nc}")
        lines.append(f"    facility_coords = {fc}")
        lines.append(f"    customer_coords = {cc}")
        lines.append(f"    capacities = {caps}")
        lines.append(f"    demands = {demands}")
        lines.append(f"    fixed_costs = {fixed}")
        lines.append(f"    rate = {rate}")
        lines.append("")
        lines.append("    # Compute distances")
        lines.append("    dist = {}")
        lines.append("    for i in range(nf):")
        lines.append("        for j in range(nc):")
        lines.append("            dx = facility_coords[i][0] - customer_coords[j][0]")
        lines.append("            dy = facility_coords[i][1] - customer_coords[j][1]")
        lines.append("            dist[(i,j)] = math.sqrt(dx*dx + dy*dy)")
        lines.append("")
        lines.append("    # Variables")
        lines.append("    open_f = {i: m.addVar(vtype=GRB.BINARY, name=f'Open_F{i}') for i in range(nf)}")
        lines.append("    ship = {(i,j): m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'Ship_F{i}_C{j}')")
        lines.append("            for i in range(nf) for j in range(nc)}")
        lines.append("")
        lines.append("    # Objective")
        lines.append("    obj = quicksum(fixed_costs[i]*open_f[i] for i in range(nf))")
        lines.append("    obj += quicksum(dist[(i,j)]*rate*ship[(i,j)] for i in range(nf) for j in range(nc))")
        lines.append(f"    m.setObjective(obj, {sense})")
        lines.append("")
        lines.append("    # Demand satisfaction")
        lines.append("    for j in range(nc):")
        lines.append("        m.addConstr(quicksum(ship[(i,j)] for i in range(nf)) >= demands[j])")
        lines.append("")
        lines.append("    # Capacity linking")
        lines.append("    for i in range(nf):")
        lines.append("        m.addConstr(quicksum(ship[(i,j)] for j in range(nc)) <= capacities[i]*open_f[i])")

        # Modifier constraints in gold solution
        for mod_key in mods:
            lines.append("")
            if mod_key == "max_open":
                K = mod_params["max_open_K"]
                lines.append(f"    # Policy: at most {K} facilities open")
                lines.append(f"    m.addConstr(quicksum(open_f[i] for i in range(nf)) <= {K})")

            elif mod_key == "budget_cap":
                B = mod_params["budget_cap_B"]
                lines.append(f"    # Policy: opening budget cap ${B:.0f}")
                lines.append(f"    m.addConstr(quicksum(fixed_costs[i]*open_f[i] for i in range(nf)) <= {B})")

            elif mod_key == "min_throughput":
                T = mod_params["min_throughput_T"]
                lines.append(f"    # Policy: min throughput {T:.0f} per open facility")
                lines.append("    for i in range(nf):")
                lines.append(f"        m.addConstr(quicksum(ship[(i,j)] for j in range(nc)) >= {T}*open_f[i])")

            elif mod_key == "facility_linking":
                A, B = mod_params["linking_pair"]
                lines.append(f"    # Policy: F{A} requires F{B}")
                lines.append(f"    m.addConstr(open_f[{A}] <= open_f[{B}])")

            elif mod_key == "anti_affinity":
                A, B = mod_params["anti_affinity_pair"]
                lines.append(f"    # Policy: F{A} and F{B} cannot both open")
                lines.append(f"    m.addConstr(open_f[{A}] + open_f[{B}] <= 1)")

            elif mod_key == "demand_cap":
                P = mod_params["demand_cap_P"]
                td = sum(demands)
                cap_val = (P / 100.0) * td
                lines.append(f"    # Policy: no facility serves > {P}% of total demand")
                lines.append(f"    total_demand = {td}")
                lines.append("    for i in range(nf):")
                lines.append(f"        m.addConstr(quicksum(ship[(i,j)] for j in range(nc)) <= {P/100.0}*total_demand)")

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
        clean_data = cls._sanitize_for_prompt(problem_data)
        system_prompt = cls.get_system_prompt()
        inst = clean_data.get("instance_data", {})

        nf = inst.get("n_facilities", 0)
        nc = inst.get("n_customers", 0)
        rate = inst.get("transport_rate", 0.5)
        mods = inst.get("modifiers", [])
        mod_params = inst.get("modifier_params", {})
        facility_coords = inst.get("facility_coordinates", [])

        # Build modifier descriptions
        # We instantiate a dummy generator just to access coord data for descriptions
        mod_descriptions = []
        for mod_key in mods:
            if mod_key == "max_open":
                K = mod_params.get("max_open_K", 3)
                mod_descriptions.append(
                    f"- The company can operate at most {K} distribution centers at any given time "
                    f"due to management oversight capacity.")
            elif mod_key == "budget_cap":
                B = mod_params.get("budget_cap_B", 1000)
                mod_descriptions.append(
                    f"- The capital budget for facility activation is capped at ${B:.0f}. "
                    f"The combined fixed costs of all opened facilities must not exceed this amount.")
            elif mod_key == "min_throughput":
                T = mod_params.get("min_throughput_T", 20)
                mod_descriptions.append(
                    f"- Every opened facility must handle at least {T:.0f} units of total outbound "
                    f"shipments to justify its operating overhead.")
            elif mod_key == "facility_linking":
                A, B = mod_params.get("linking_pair", [0, 1])
                cA = facility_coords[A] if A < len(facility_coords) else (0, 0)
                cB = facility_coords[B] if B < len(facility_coords) else (0, 0)
                mod_descriptions.append(
                    f"- The facility at F{A} ({cA[0]}, {cA[1]}) depends on the regional hub "
                    f"at F{B} ({cB[0]}, {cB[1]}). If F{A} is activated, F{B} must also be activated.")
            elif mod_key == "anti_affinity":
                A, B = mod_params.get("anti_affinity_pair", [0, 1])
                cA = facility_coords[A] if A < len(facility_coords) else (0, 0)
                cB = facility_coords[B] if B < len(facility_coords) else (0, 0)
                mod_descriptions.append(
                    f"- Regulatory restrictions prevent facilities F{A} ({cA[0]}, {cA[1]}) "
                    f"and F{B} ({cB[0]}, {cB[1]}) from operating simultaneously "
                    f"due to overlapping market zones.")
            elif mod_key == "demand_cap":
                P = mod_params.get("demand_cap_P", 50)
                mod_descriptions.append(
                    f"- To avoid over-reliance on any single facility, no distribution center "
                    f"may fulfill more than {P}% of the total system demand.")

        mod_block = "\n".join(mod_descriptions)

        user_content = f"""Write a business memo for a facility location decision problem WITH additional company policies.

**Scenario:**
A company needs to decide which warehouses to open and how to ship products to customers.
- {nf} candidate facility locations (with map coordinates)
- {nc} customer locations (with map coordinates)
- Transport rate: ${rate} per km per unit shipped

**Standard Requirements:**
- Each customer's demand must be fully satisfied
- A facility can only ship if it's opened
- Opening a facility has a fixed cost
- Shipping costs depend on distance traveled and volume shipped

**CRITICAL — Additional Company Policies (MUST be included in the memo):**
{mod_block}

These policies are MANDATORY requirements, not optional suggestions. Each one must appear clearly in the memo body so a reader can implement them.

**Your Task:**
Write a professional business memo.
- Standard data goes in an Annex using ONLY these placeholders:
  {{FACILITY_COORDINATES}}, {{CUSTOMER_COORDINATES}}, {{FACILITY_CAPACITIES}},
  {{CUSTOMER_DEMANDS}}, {{FIXED_COSTS}}, {{TRANSPORT_RATE}}
- The additional company policies must be in the BODY of the memo with their specific numbers.
- Do NOT put policy details in the annex.
- Do NOT label policies as "constraints" or "additional constraints."
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
            "FACILITY_COORDINATES", "CUSTOMER_COORDINATES", "FACILITY_CAPACITIES",
            "CUSTOMER_DEMANDS", "FIXED_COSTS", "TRANSPORT_RATE",
        ]

        has_placeholders = any(f"{{{p}}}" in llm_description for p in required)
        if not has_placeholders:
            return llm_description

        inst = problem_data.get("instance_data", {})
        if not inst:
            return llm_description

        missing = [p for p in required if f"{{{p}}}" not in llm_description]
        if missing:
            print(f"Missing placeholders: {missing}")
            return None

        fmt = {}

        # Facility coordinates
        fc = inst.get("facility_coordinates", [])
        lines = ["Facility Coordinates:"]
        for i, (x, y) in enumerate(fc):
            lines.append(f"  - F{i}: ({x}, {y})")
        fmt["FACILITY_COORDINATES"] = "\n".join(lines)

        # Customer coordinates
        cc = inst.get("customer_coordinates", [])
        lines = ["Customer Coordinates:"]
        for j, (x, y) in enumerate(cc):
            lines.append(f"  - C{j}: ({x}, {y})")
        fmt["CUSTOMER_COORDINATES"] = "\n".join(lines)

        fmt["FACILITY_CAPACITIES"] = str(inst.get("facility_capacities", []))
        fmt["CUSTOMER_DEMANDS"] = str(inst.get("customer_demands", []))
        fmt["FIXED_COSTS"] = str(inst.get("fixed_costs", []))
        fmt["TRANSPORT_RATE"] = str(inst.get("transport_rate", 0.5))

        result = llm_description
        for ph, val in fmt.items():
            result = result.replace(f"{{{ph}}}", val)

        remaining = [p for p in required if f"{{{p}}}" in result]
        if remaining:
            print(f"Placeholders not filled: {remaining}")
            return None

        return result
