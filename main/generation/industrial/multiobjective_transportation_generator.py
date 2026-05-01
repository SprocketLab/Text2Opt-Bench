import numpy as np
from main.generation.base_problem import BaseProblem


class MultiObjectiveTransportationGenerator(BaseProblem):
    """
    Multi-Objective Transportation with Industrial Side Constraints (MILP).

    Ship goods from sources to destinations with two conflicting objectives
    (cost vs CO2 emissions), scalarized via weighted sum, PLUS industrial
    constraints that introduce binary variables and make this a real MILP:

      1. Fixed-charge costs: opening a route (i,j) incurs a one-time setup
         cost f_ij, modeled with binary y_ij and Big-M linking.
      2. Minimum order quantities (MOQ): if route (i,j) is used (y_ij=1),
         must ship at least MOQ_ij units.
      3. Max suppliers per store: each destination can receive from at most
         K warehouses (limits the number of active inbound routes).

    Variables:
      x_ij  — continuous shipment quantity on route (i,j), >= 0
      y_ij  — binary indicator: 1 if route (i,j) is used at all

    Objective: MINIMIZE
      w_cost * Σ c_ij * x_ij  +  w_emission * Σ e_ij * x_ij  +  Σ f_ij * y_ij

    Constraints:
      Supply:      Σ_j x_ij <= supply_i                       for all i
      Demand:      Σ_i x_ij >= demand_j                        for all j
      Linking:     x_ij <= M_ij * y_ij                         for all i,j
      MOQ:         x_ij >= MOQ_ij * y_ij                       for all i,j
      MaxSupplier: Σ_i y_ij <= K_j                             for all j
    """

    SYSTEM_PROMPT = """You are a Sustainability & Logistics Director writing a concise memo about an industrial shipping challenge that balances cost and environmental impact, with real-world operational constraints.

**TASK:** Write a professional business memo describing a transportation problem with two competing goals (cost vs emissions) AND practical operational rules.

**CRITICAL INSTRUCTION - TEMPLATE GENERATION:**
- You must generate a **text template** that uses specific placeholders for data.
- **DO NOT** attempt to describe the specific data values (numbers, matrices) in the narrative.
- **DO NOT** invent numbers. Use the placeholders.
- Place all data placeholders in a dedicated "ANNEX" or "DATA" section at the end of the memo.

**TONE:** Professional business memo. Clear and practical. Something a VP of Supply Chain would write.

**REQUIRED PLACEHOLDERS (include ALL exactly once):**
{NUM_SOURCES}, {NUM_DESTINATIONS}, {SUPPLIES}, {DEMANDS},
{COST_MATRIX}, {EMISSION_MATRIX}, {WEIGHT_COST}, {WEIGHT_EMISSION},
{FIXED_COSTS}, {MIN_ORDER_QUANTITIES}, {MAX_SUPPLIERS_PER_STORE}

**SCENARIO TO DESCRIBE:**
- Company ships products from multiple warehouses to multiple stores.
- Each shipping route has TWO metrics: a monetary cost ($/unit) and a carbon emission rate (kg CO2/unit).
- These goals conflict: the cheapest routes are often the most polluting.
- Management has set importance weights for cost vs emissions.

**INDUSTRIAL OPERATIONAL RULES (describe ALL three naturally):**
1. **Setup/activation costs:** Opening any shipping route requires a one-time setup cost (contract fees, customs paperwork, loading dock setup). A route is either active or not — if active, the setup cost is charged regardless of volume shipped.
2. **Minimum order quantities:** If a route is activated, a minimum number of units must be shipped on it (to justify the logistics overhead — partial truck loads are not accepted).
3. **Supplier limits per store:** Each store can receive deliveries from at most a limited number of warehouses (due to receiving dock capacity, vendor management complexity, or contractual agreements).

**WHAT TO INCLUDE:**
- Business context: why balancing cost and sustainability matters
- The three operational rules described naturally (NOT as math)
- Supply limits at warehouses and demand requirements at stores
- The goal: find shipping plan that minimizes a combined objective. The combined objective is: (weight_cost × variable shipping cost) + (weight_emission × total emissions) + setup costs for activated routes. Note: setup costs are added at full value (NOT weighted), only the per-unit variable costs and emissions are weighted.

**WHAT TO AVOID:**
- Mathematical notation or formulas
- Technical terms like "objective function", "constraints", "decision variables", "binary", "indicator", "Big-M", "MILP"
- Phrases like "MINIMIZE", "subject to", "<=", ">="
- Anything that sounds like an optimization textbook

**SUGGESTED STRUCTURE:**
1. Subject/Date: Standard memo header
2. Context: Balancing cost and sustainability in shipping
3. Operational rules: Setup costs, minimum orders, supplier limits
4. Management priority: Cost vs emissions weights
5. Goal: Find the best shipping plan under these priorities and rules
6. Annex - Data: All placeholders"""

    def __init__(
        self,
        num_sources: int = 5,
        num_dests: int = 8,
        supply_range: tuple = (80, 250),
        demand_range: tuple = (10, 60),
        cost_range: tuple = (3, 30),
        emission_range: tuple = (1, 20),
        fixed_cost_range: tuple = (50, 300),
        moq_range: tuple = (5, 20),
        max_suppliers_per_store: int = 3,
        weight_cost: float = 0.6,
        weight_emission: float = 0.4,
        cost_emission_correlation: float = -0.3,
    ):
        self.num_sources = num_sources
        self.num_dests = num_dests
        self.supply_range = supply_range
        self.demand_range = demand_range
        self.cost_range = cost_range
        self.emission_range = emission_range
        self.fixed_cost_range = fixed_cost_range
        self.moq_range = moq_range
        self.max_suppliers_per_store = max_suppliers_per_store
        self.weight_cost = weight_cost
        self.weight_emission = weight_emission
        self.cost_emission_correlation = cost_emission_correlation

        I, J = num_sources, num_dests
        self.n_x = I * J       # continuous shipment vars
        self.n_y = I * J       # binary route-open vars
        n_vars = self.n_x + self.n_y

        # Constraints: supply(I) + demand(J) + linking(I*J) + MOQ(I*J) + maxSupplier(J)
        n_constrs = I + J + I * J + I * J + J

        super().__init__(
            n_vars=n_vars,
            n_constrs=n_constrs,
            prob_type="MILP",
            goal="MINIMIZE",
            problem_type="multi-objective transportation",
        )

        # Instance data (populated by _generate_instance_data)
        self.supplies = None
        self.demands = None
        self.cost_matrix = None
        self.emission_matrix = None
        self.fixed_costs = None
        self.moq_matrix = None
        self.big_M = None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, time_limit: float = 300.0):
        self._generate_instance_data()
        self._build_var_metadata()
        self._solve_direct(time_limit=time_limit)
        self._build_sparse_serialization()
        return self

    def _generate_instance_data(self):
        I = self.num_sources
        J = self.num_dests

        # Supplies
        self.supplies = np.random.randint(
            self.supply_range[0], self.supply_range[1] + 1, size=I
        ).tolist()

        # Demands — ensure total supply >= total demand with margin
        raw_demands = np.random.randint(
            self.demand_range[0], self.demand_range[1] + 1, size=J
        )
        total_supply = sum(self.supplies)
        total_demand = sum(raw_demands)
        if total_demand > total_supply * 0.85:
            scale = (total_supply * 0.75) / total_demand
            raw_demands = np.maximum(1, np.round(raw_demands * scale).astype(int))
        self.demands = raw_demands.tolist()

        # Cost matrix
        self.cost_matrix = np.random.uniform(
            self.cost_range[0], self.cost_range[1], size=(I, J)
        ).round(2).tolist()

        # Emission matrix — negatively correlated with cost
        corr = self.cost_emission_correlation
        cost_arr = np.array(self.cost_matrix)
        cost_norm = (cost_arr - cost_arr.mean()) / (cost_arr.std() + 1e-9)

        emission_base = np.random.uniform(
            self.emission_range[0], self.emission_range[1], size=(I, J)
        )
        emission_norm = (emission_base - emission_base.mean()) / (emission_base.std() + 1e-9)
        emission_correlated = corr * cost_norm + np.sqrt(1 - corr**2) * emission_norm
        emission_scaled = (
            emission_correlated - emission_correlated.min()
        ) / (emission_correlated.max() - emission_correlated.min() + 1e-9)
        emission_final = (
            self.emission_range[0]
            + emission_scaled * (self.emission_range[1] - self.emission_range[0])
        )
        self.emission_matrix = np.round(emission_final, 2).tolist()

        # Fixed costs per route
        self.fixed_costs = np.random.randint(
            self.fixed_cost_range[0], self.fixed_cost_range[1] + 1, size=(I, J)
        ).tolist()

        # Minimum order quantities per route
        self.moq_matrix = np.random.randint(
            self.moq_range[0], self.moq_range[1] + 1, size=(I, J)
        ).tolist()

        # Big-M for linking: max a single route could carry = min(supply_i, total_demand)
        self.big_M = [[
            min(self.supplies[i], sum(self.demands))
            for j in range(J)]
            for i in range(I)
        ]

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _x_idx(self, i, j):
        """Index for continuous shipment x_ij."""
        return i * self.num_dests + j

    def _y_idx(self, i, j):
        """Index for binary route-open indicator y_ij."""
        return self.n_x + i * self.num_dests + j

    # ------------------------------------------------------------------
    # Variable metadata
    # ------------------------------------------------------------------

    def _build_var_metadata(self):
        I = self.num_sources
        J = self.num_dests

        self.var_names = []
        # Continuous shipment variables
        for i in range(I):
            for j in range(J):
                self.var_names.append(f"Ship_S{i}_D{j}")
        # Binary route-open indicators
        for i in range(I):
            for j in range(J):
                self.var_names.append(f"RouteOpen_S{i}_D{j}")

        self.var_types = (
            ["Continuous"] * self.n_x
            + ["Binary"] * self.n_y
        )
        self.var_bounds = (
            [(0.0, None)] * self.n_x
            + [(0.0, 1.0)] * self.n_y
        )

        # Objective coefficients
        self.c = np.zeros(self.n)
        for i in range(I):
            for j in range(J):
                # Weighted variable cost on x_ij
                self.c[self._x_idx(i, j)] = (
                    self.weight_cost * self.cost_matrix[i][j]
                    + self.weight_emission * self.emission_matrix[i][j]
                )
                # Fixed cost on y_ij
                self.c[self._y_idx(i, j)] = self.fixed_costs[i][j]

    # ------------------------------------------------------------------
    # Direct Gurobi solve
    # ------------------------------------------------------------------

    def _solve_direct(self, time_limit: float = 300.0):
        import gurobipy as gp
        from gurobipy import GRB

        I = self.num_sources
        J = self.num_dests
        K = self.max_suppliers_per_store

        m = gp.Model("MultiObjective_Transportation")
        m.setParam("OutputFlag", 0)
        m.setParam("LogToConsole", 0)
        if time_limit:
            m.setParam("TimeLimit", time_limit)

        # Continuous shipment variables
        x = {}
        for i in range(I):
            for j in range(J):
                x[(i, j)] = m.addVar(lb=0, vtype=GRB.CONTINUOUS,
                                     name=f"Ship_S{i}_D{j}")

        # Binary route-open indicators
        y = {}
        for i in range(I):
            for j in range(J):
                y[(i, j)] = m.addVar(vtype=GRB.BINARY,
                                     name=f"RouteOpen_S{i}_D{j}")

        # Objective: weighted variable cost + fixed costs
        m.setObjective(
            gp.quicksum(
                (self.weight_cost * self.cost_matrix[i][j]
                 + self.weight_emission * self.emission_matrix[i][j])
                * x[(i, j)]
                for i in range(I) for j in range(J)
            )
            + gp.quicksum(
                self.fixed_costs[i][j] * y[(i, j)]
                for i in range(I) for j in range(J)
            ),
            GRB.MINIMIZE,
        )

        # C1: Supply
        for i in range(I):
            m.addConstr(
                gp.quicksum(x[(i, j)] for j in range(J)) <= self.supplies[i],
                name=f"Supply_S{i}",
            )

        # C2: Demand
        for j in range(J):
            m.addConstr(
                gp.quicksum(x[(i, j)] for i in range(I)) >= self.demands[j],
                name=f"Demand_D{j}",
            )

        # C3: Linking (Big-M) — x_ij <= M_ij * y_ij
        for i in range(I):
            for j in range(J):
                m.addConstr(
                    x[(i, j)] <= self.big_M[i][j] * y[(i, j)],
                    name=f"Link_S{i}_D{j}",
                )

        # C4: Minimum order quantity — x_ij >= MOQ_ij * y_ij
        for i in range(I):
            for j in range(J):
                m.addConstr(
                    x[(i, j)] >= self.moq_matrix[i][j] * y[(i, j)],
                    name=f"MOQ_S{i}_D{j}",
                )

        # C5: Max suppliers per store — Σ_i y_ij <= K
        for j in range(J):
            m.addConstr(
                gp.quicksum(y[(i, j)] for i in range(I)) <= K,
                name=f"MaxSupplier_D{j}",
            )

        m.optimize()

        self.solver_status = int(m.Status)
        if m.Status == GRB.TIME_LIMIT:
            raise TimeoutError(f"Gurobi hit the {time_limit}s time limit")
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi failed with status {m.Status}")

        self.x_star = np.array([m.getVarByName(n).X for n in self.var_names])
        self.obj_val = float(m.ObjVal)

    # ------------------------------------------------------------------
    # Sparse serialization
    # ------------------------------------------------------------------

    def _build_sparse_serialization(self):
        I = self.num_sources
        J = self.num_dests
        K = self.max_suppliers_per_store

        self.constr_names = []
        self.senses = []
        b_list = []
        self._A_sparse = {}
        row = 0

        # C1: Supply
        for i in range(I):
            self._A_sparse[row] = {self._x_idx(i, j): 1.0 for j in range(J)}
            b_list.append(float(self.supplies[i]))
            self.senses.append("<=")
            self.constr_names.append(f"Supply_S{i}")
            row += 1

        # C2: Demand
        for j in range(J):
            self._A_sparse[row] = {self._x_idx(i, j): 1.0 for i in range(I)}
            b_list.append(float(self.demands[j]))
            self.senses.append(">=")
            self.constr_names.append(f"Demand_D{j}")
            row += 1

        # C3: Linking — x_ij - M_ij * y_ij <= 0
        for i in range(I):
            for j in range(J):
                self._A_sparse[row] = {
                    self._x_idx(i, j): 1.0,
                    self._y_idx(i, j): -float(self.big_M[i][j]),
                }
                b_list.append(0.0)
                self.senses.append("<=")
                self.constr_names.append(f"Link_S{i}_D{j}")
                row += 1

        # C4: MOQ — x_ij - MOQ_ij * y_ij >= 0  →  -x_ij + MOQ_ij * y_ij <= 0
        for i in range(I):
            for j in range(J):
                self._A_sparse[row] = {
                    self._x_idx(i, j): 1.0,
                    self._y_idx(i, j): -float(self.moq_matrix[i][j]),
                }
                b_list.append(0.0)
                self.senses.append(">=")
                self.constr_names.append(f"MOQ_S{i}_D{j}")
                row += 1

        # C5: MaxSupplier — Σ_i y_ij <= K
        for j in range(J):
            self._A_sparse[row] = {self._y_idx(i, j): 1.0 for i in range(I)}
            b_list.append(float(K))
            self.senses.append("<=")
            self.constr_names.append(f"MaxSupplier_D{j}")
            row += 1

        self.b = np.array(b_list)
        self.A = None

    def to_json_dict(self):
        """Override to use sparse A instead of dense matrix."""
        if self.x_star is None or self.obj_val is None:
            raise ValueError("Problem not solved yet.")

        is_integer = any(vt in ("Integer", "Binary") for vt in self.var_types)
        optimal_values = {
            name: float(self.x_star[i]) for i, name in enumerate(self.var_names)
        }
        instance_data = self._get_instance_data()

        # Compute individual objective values
        cost_val = sum(
            self.cost_matrix[i][j] * self.x_star[self._x_idx(i, j)]
            for i in range(self.num_sources) for j in range(self.num_dests)
        )
        emission_val = sum(
            self.emission_matrix[i][j] * self.x_star[self._x_idx(i, j)]
            for i in range(self.num_sources) for j in range(self.num_dests)
        )
        fixed_val = sum(
            self.fixed_costs[i][j] * self.x_star[self._y_idx(i, j)]
            for i in range(self.num_sources) for j in range(self.num_dests)
        )

        data = {
            "meta": {
                "optimization_type": self.prob_type,
                "problem_type": self.problem_type,
                "goal": self.goal,
                "num_vars": self.n,
                "num_constraints": self.m,
                "is_integer": is_integer,
            },
            "instance_data": instance_data,
            "gurobi_result": {
                "solver_status": self.solver_status,
                "theoretical_optimum": float(self.obj_val),
                "optimal_values": optimal_values,
                "individual_objectives": {
                    "total_cost": round(float(cost_val), 4),
                    "total_emissions": round(float(emission_val), 4),
                    "total_fixed_costs": round(float(fixed_val), 4),
                },
            },
            "constraints": {},
            "variables": {},
        }

        for i, cname in enumerate(self.constr_names):
            data["constraints"][cname] = {
                "type": "Constraint",
                "sense": self.senses[i],
                "rhs": float(self.b[i]),
            }

        for j, vname in enumerate(self.var_names):
            resource_costs = {}
            if self._A_sparse is not None:
                for row_idx, coeffs in self._A_sparse.items():
                    if j in coeffs and abs(coeffs[j]) > 1e-9:
                        resource_costs[self.constr_names[row_idx]] = coeffs[j]

            if self.var_types[j] == "Binary":
                lb, ub = 0.0, 1.0
            else:
                lb, ub = self.var_bounds[j] if self.var_bounds else (0.0, None)
            ub_val = "inf" if ub is None else float(ub)

            data["variables"][vname] = {
                "type": self.var_types[j],
                "range": [float(lb), ub_val],
                "objective_linear_coefficient": float(self.c[j]) if self.c is not None else 0.0,
                "resource_costs": resource_costs,
            }

        return data

    # ------------------------------------------------------------------
    # Instance data
    # ------------------------------------------------------------------

    def _get_instance_data(self):
        return {
            "num_sources": self.num_sources,
            "num_destinations": self.num_dests,
            "supplies": self.supplies,
            "demands": self.demands,
            "cost_matrix": self.cost_matrix,
            "emission_matrix": self.emission_matrix,
            "fixed_costs": self.fixed_costs,
            "min_order_quantities": self.moq_matrix,
            "max_suppliers_per_store": self.max_suppliers_per_store,
            "weight_cost": self.weight_cost,
            "weight_emission": self.weight_emission,
        }

    # ------------------------------------------------------------------
    # Gold solution code
    # ------------------------------------------------------------------

    def generate_gurobi_code_reference(self, data: dict) -> str:
        inst = data.get("instance_data", {})
        meta = data.get("meta", {})

        I = inst.get("num_sources", 0)
        J = inst.get("num_destinations", 0)
        supplies = inst.get("supplies", [])
        demands = inst.get("demands", [])
        cost_matrix = inst.get("cost_matrix", [])
        emission_matrix = inst.get("emission_matrix", [])
        fixed_costs = inst.get("fixed_costs", [])
        moq = inst.get("min_order_quantities", [])
        K = inst.get("max_suppliers_per_store", 3)
        w_cost = inst.get("weight_cost", 0.6)
        w_emission = inst.get("weight_emission", 0.4)

        sense = "GRB.MINIMIZE" if meta.get("goal") == "MINIMIZE" else "GRB.MAXIMIZE"

        lines = []
        lines.append("```python")
        lines.append("from gurobipy import *")
        lines.append("")
        lines.append("def solve_problem():")
        lines.append('    m = Model("MultiObjective_Transportation")')
        lines.append("    m.setParam('TimeLimit', 300)")
        lines.append("")
        lines.append("    # --- Data ---")
        lines.append(f"    I = {I}  # sources")
        lines.append(f"    J = {J}  # destinations")
        lines.append(f"    K = {K}  # max suppliers per store")
        lines.append(f"    w_cost = {w_cost}")
        lines.append(f"    w_emission = {w_emission}")
        lines.append("")
        lines.append(f"    supplies = {supplies}")
        lines.append(f"    demands = {demands}")
        lines.append(f"    cost_matrix = {cost_matrix}")
        lines.append(f"    emission_matrix = {emission_matrix}")
        lines.append(f"    fixed_costs = {fixed_costs}")
        lines.append(f"    moq = {moq}")
        lines.append("")
        lines.append("    # Big-M per route")
        lines.append("    total_demand = sum(demands)")
        lines.append("    big_M = [[min(supplies[i], total_demand) for j in range(J)] for i in range(I)]")
        lines.append("")
        lines.append("    # --- Variables ---")
        lines.append("    # Continuous: shipment quantities")
        lines.append("    x = {}")
        lines.append("    for i in range(I):")
        lines.append("        for j in range(J):")
        lines.append('            x[(i,j)] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"Ship_S{i}_D{j}")')
        lines.append("")
        lines.append("    # Binary: route-open indicators")
        lines.append("    y = {}")
        lines.append("    for i in range(I):")
        lines.append("        for j in range(J):")
        lines.append('            y[(i,j)] = m.addVar(vtype=GRB.BINARY, name=f"RouteOpen_S{i}_D{j}")')
        lines.append("")
        lines.append("    # --- Objective ---")
        lines.append("    # Weighted variable cost + fixed setup costs")
        lines.append("    m.setObjective(")
        lines.append("        quicksum(")
        lines.append("            (w_cost * cost_matrix[i][j] + w_emission * emission_matrix[i][j]) * x[(i,j)]")
        lines.append("            for i in range(I) for j in range(J)")
        lines.append("        )")
        lines.append("        + quicksum(fixed_costs[i][j] * y[(i,j)] for i in range(I) for j in range(J)),")
        lines.append(f"        {sense}")
        lines.append("    )")
        lines.append("")
        lines.append("    # --- Constraints ---")
        lines.append("    # C1: Supply")
        lines.append("    for i in range(I):")
        lines.append("        m.addConstr(")
        lines.append("            quicksum(x[(i,j)] for j in range(J)) <= supplies[i],")
        lines.append('            name=f"Supply_S{i}")')
        lines.append("")
        lines.append("    # C2: Demand")
        lines.append("    for j in range(J):")
        lines.append("        m.addConstr(")
        lines.append("            quicksum(x[(i,j)] for i in range(I)) >= demands[j],")
        lines.append('            name=f"Demand_D{j}")')
        lines.append("")
        lines.append("    # C3: Linking — can only ship if route is open")
        lines.append("    for i in range(I):")
        lines.append("        for j in range(J):")
        lines.append("            m.addConstr(")
        lines.append("                x[(i,j)] <= big_M[i][j] * y[(i,j)],")
        lines.append('                name=f"Link_S{i}_D{j}")')
        lines.append("")
        lines.append("    # C4: Minimum order quantity")
        lines.append("    for i in range(I):")
        lines.append("        for j in range(J):")
        lines.append("            m.addConstr(")
        lines.append("                x[(i,j)] >= moq[i][j] * y[(i,j)],")
        lines.append('                name=f"MOQ_S{i}_D{j}")')
        lines.append("")
        lines.append("    # C5: Max suppliers per store")
        lines.append("    for j in range(J):")
        lines.append("        m.addConstr(")
        lines.append("            quicksum(y[(i,j)] for i in range(I)) <= K,")
        lines.append('            name=f"MaxSupplier_D{j}")')
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
        I = inst.get("num_sources", 0)
        J = inst.get("num_destinations", 0)
        K = inst.get("max_suppliers_per_store", 3)
        w_cost = inst.get("weight_cost", 0.6)
        w_emission = inst.get("weight_emission", 0.4)

        user_content = f"""Write a business memo for a shipping problem that balances cost and environmental impact, with real-world operational rules.

**Company Situation:**
- We have {I} source warehouses and {J} destination stores.
- Each shipping route has two metrics: a dollar cost per unit AND a CO2 emission rate per unit.
- These often conflict — the cheapest routes tend to produce the most emissions.

**Operational Rules:**
1. **Setup costs:** Activating any shipping route requires a one-time setup cost (regardless of volume). A route is either active or not.
2. **Minimum order quantities:** If a route is activated, a minimum number of units must be shipped on it.
3. **Supplier limits:** Each store can receive deliveries from at most {K} warehouses.

**Management Priority:**
- Cost importance: {w_cost*100:.0f}%
- Emission importance: {w_emission*100:.0f}%
- We want to minimize a blend of total variable cost, total emissions, and total setup costs.

**Business Goal:**
Find which routes to activate and how much to ship on each, such that:
- Warehouse inventory is not exceeded
- Each store's minimum demand is met
- Activated routes ship at least the minimum order
- Each store receives from at most {K} warehouses
- Total weighted cost (variable + emissions + setup) is minimized

**Your Task:**
Write a professional business memo describing this challenge.
IMPORTANT: Do NOT include any specific numbers or data values in your text.
Use ONLY these placeholders in an Annex section:
{{NUM_SOURCES}}, {{NUM_DESTINATIONS}}, {{SUPPLIES}}, {{DEMANDS}},
{{COST_MATRIX}}, {{EMISSION_MATRIX}}, {{WEIGHT_COST}}, {{WEIGHT_EMISSION}},
{{FIXED_COSTS}}, {{MIN_ORDER_QUANTITIES}}, {{MAX_SUPPLIERS_PER_STORE}}

Each placeholder should appear exactly ONCE.
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
            "NUM_SOURCES", "NUM_DESTINATIONS", "SUPPLIES", "DEMANDS",
            "COST_MATRIX", "EMISSION_MATRIX", "WEIGHT_COST", "WEIGHT_EMISSION",
            "FIXED_COSTS", "MIN_ORDER_QUANTITIES", "MAX_SUPPLIERS_PER_STORE",
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
        fmt["NUM_SOURCES"] = str(inst.get("num_sources", ""))
        fmt["NUM_DESTINATIONS"] = str(inst.get("num_destinations", ""))
        fmt["WEIGHT_COST"] = f"{inst.get('weight_cost', 0.6) * 100:.0f}%"
        fmt["WEIGHT_EMISSION"] = f"{inst.get('weight_emission', 0.4) * 100:.0f}%"
        fmt["MAX_SUPPLIERS_PER_STORE"] = str(inst.get("max_suppliers_per_store", 3))

        # Supplies
        fmt["SUPPLIES"] = f"Warehouse capacities: {inst.get('supplies', [])}"

        # Demands
        fmt["DEMANDS"] = f"Store demand requirements: {inst.get('demands', [])}"

        # Cost matrix
        cost_matrix = inst.get("cost_matrix", [])
        I = len(cost_matrix)
        J = len(cost_matrix[0]) if cost_matrix else 0
        header = "Source\\Dest | " + " | ".join(f"D{j}" for j in range(J))

        cm_lines = [f"Shipping Costs ($/unit, {I} sources x {J} destinations):"]
        cm_lines.append(header)
        cm_lines.append("-" * len(header))
        for i, row in enumerate(cost_matrix):
            row_str = f"S{i:<9} | " + " | ".join(f"{v:6.2f}" for v in row)
            cm_lines.append(row_str)
        fmt["COST_MATRIX"] = "\n".join(cm_lines)

        # Emission matrix
        emission_matrix = inst.get("emission_matrix", [])
        em_lines = [f"CO2 Emissions (kg CO2/unit, {I} sources x {J} destinations):"]
        em_lines.append(header)
        em_lines.append("-" * len(header))
        for i, row in enumerate(emission_matrix):
            row_str = f"S{i:<9} | " + " | ".join(f"{v:6.2f}" for v in row)
            em_lines.append(row_str)
        fmt["EMISSION_MATRIX"] = "\n".join(em_lines)

        # Fixed costs
        fc = inst.get("fixed_costs", [])
        fc_lines = [f"Route Setup Costs ($, {I} sources x {J} destinations):"]
        fc_lines.append(header)
        fc_lines.append("-" * len(header))
        for i, row in enumerate(fc):
            row_str = f"S{i:<9} | " + " | ".join(f"{v:>6}" for v in row)
            fc_lines.append(row_str)
        fmt["FIXED_COSTS"] = "\n".join(fc_lines)

        # MOQ matrix
        moq = inst.get("min_order_quantities", [])
        moq_lines = [f"Minimum Order Quantities (units, {I} sources x {J} destinations):"]
        moq_lines.append(header)
        moq_lines.append("-" * len(header))
        for i, row in enumerate(moq):
            row_str = f"S{i:<9} | " + " | ".join(f"{v:>6}" for v in row)
            moq_lines.append(row_str)
        fmt["MIN_ORDER_QUANTITIES"] = "\n".join(moq_lines)

        result = llm_description
        for ph, val in fmt.items():
            result = result.replace(f"{{{ph}}}", val)

        remaining = [p for p in required if f"{{{p}}}" in result]
        if remaining:
            print(f"Placeholders not filled: {remaining}")
            return None

        return result
