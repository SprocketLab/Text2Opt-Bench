import numpy as np
from main.generation.base_problem import BaseProblem


class StochasticTransportationGenerator(BaseProblem):
    """
    Two-Stage Stochastic Transportation Problem with SAA-based chance constraints.

    Stage 1 (here-and-now): Decide base shipment quantities x_ij from source i
    to destination j BEFORE demand is known.

    Stage 2 (recourse): After demand scenario s is revealed, decide emergency
    shipment quantities y_ij_s at higher cost to cover any shortfall.

    Chance constraint (SAA): Demand at each destination must be fully met in
    at least ceil(alpha * S) of the S scenarios. A binary indicator z_js = 1
    flags scenario s where destination j's demand is NOT fully met.

    Deterministic equivalent (single MILP):
      Variables:
        x_ij          — base shipment (Continuous, >= 0)
        y_ij_s        — emergency recourse per scenario (Continuous, >= 0)
        z_js          — violation indicator (Binary)

      Objective: MINIMIZE
        sum_ij c_ij * x_ij                                   (base cost)
      + (1/S) * sum_s sum_ij q_ij * y_ij_s                  (expected recourse)

      Constraints:
        Supply (base):    sum_j x_ij <= Supply_i               for all i
        Supply (recourse): sum_j y_ij_s <= RecourseCapacity_i  for all i, s
        Demand (chance):  sum_i (x_ij + y_ij_s) >= D_js - M * z_js
                                                               for all j, s
        Violation budget: sum_s z_js <= floor((1-alpha) * S)   for all j
    """

    SYSTEM_PROMPT = """You are a Supply Chain Director writing a concise memo about a logistics challenge under demand uncertainty.

**TASK:** Write a professional business memo describing a two-stage shipping problem where demand is uncertain.

**CRITICAL INSTRUCTION - TEMPLATE GENERATION:**
- You must generate a **text template** that uses specific placeholders for data.
- **DO NOT** attempt to describe the specific data values (numbers, matrices) in the narrative.
- **DO NOT** invent numbers. Use the placeholders.
- Place all data placeholders in a dedicated "ANNEX" or "DATA" section at the end of the memo.

**TONE:** Professional business memo. Clear and practical. Something a logistics manager would write.

**REQUIRED PLACEHOLDERS (include ALL exactly once):**
{NUM_SOURCES}, {NUM_DESTINATIONS}, {NUM_SCENARIOS}, {CONFIDENCE_LEVEL},
{SUPPLIES}, {RECOURSE_CAPACITIES}, {BASE_COST_MATRIX}, {RECOURSE_COST_MATRIX},
{DEMAND_SCENARIOS}

**SCENARIO TO DESCRIBE:**
- Company ships goods from multiple warehouses to multiple stores.
- Customer demand is uncertain — we have multiple demand scenarios based on forecasts.
- Stage 1: Lock in base shipping plan BEFORE knowing actual demand.
- Stage 2: After demand is revealed, use more expensive emergency shipments to cover shortfalls.
- Reliability requirement: each store's demand must be fully met in at least a certain percentage of scenarios (e.g., 90% or 95%).
- Goal: minimize total expected cost = base shipping cost + average emergency cost across scenarios.

**WHAT TO INCLUDE:**
- Business context: why planning under uncertainty matters (avoiding stockouts, customer satisfaction)
- Two-phase decision: initial shipping plan locked in advance, emergency orders if needed
- Reliability target: demand must be met in most scenarios (not necessarily all)
- The goal: minimize total expected cost while maintaining reliability

**WHAT TO AVOID:**
- Mathematical notation or formulas
- Technical terms like "objective function", "constraints", "decision variables", "SAA", "chance constraint"
- Phrases like "MINIMIZE", "subject to", "<=", ">="
- Anything that sounds like an optimization textbook

**SUGGESTED STRUCTURE:**
1. Subject/Date: Standard memo header
2. Challenge: Uncertain customer demand and the need for a reliable shipping plan
3. Two-phase approach: Commit to a base plan, then adjust with emergency shipments
4. Reliability target: Describe the service level requirement naturally
5. Goal: Minimize total expected cost
6. Annex - Data: All placeholders"""

    def __init__(
        self,
        num_sources: int = 5,
        num_dests: int = 8,
        num_scenarios: int = 20,
        supply_range: tuple = (80, 200),
        base_demand_range: tuple = (10, 50),
        base_cost_range: tuple = (5, 25),
        recourse_cost_multiplier: float = 2.0,
        demand_noise_pct: float = 0.30,
        alpha: float = 0.90,
        recourse_capacity_fraction: float = 0.4,
    ):
        self.num_sources = num_sources
        self.num_dests = num_dests
        self.num_scenarios = num_scenarios
        self.supply_range = supply_range
        self.base_demand_range = base_demand_range
        self.base_cost_range = base_cost_range
        self.recourse_cost_multiplier = recourse_cost_multiplier
        self.demand_noise_pct = demand_noise_pct
        self.alpha = alpha
        self.recourse_capacity_fraction = recourse_capacity_fraction

        S = num_scenarios
        I = num_sources
        J = num_dests

        # Variable counts
        self.n_x = I * J                  # base shipments
        self.n_y = I * J * S              # recourse shipments
        self.n_z = J * S                  # violation indicators

        n_vars = self.n_x + self.n_y + self.n_z

        # Constraint counts
        self.n_supply_base = I            # supply constraints (base)
        self.n_supply_recourse = I * S    # supply constraints (recourse, per scenario)
        self.n_demand_chance = J * S      # demand chance constraints
        self.n_violation_budget = J       # violation budget (per destination)

        n_constrs = (self.n_supply_base + self.n_supply_recourse
                     + self.n_demand_chance + self.n_violation_budget)

        super().__init__(
            n_vars=n_vars,
            n_constrs=n_constrs,
            prob_type="MILP",
            goal="MINIMIZE",
            problem_type="stochastic transportation",
        )

        # Instance data (populated by _generate_instance_data)
        self.supplies = None
        self.recourse_capacities = None
        self.base_demands = None
        self.base_costs = None
        self.recourse_costs = None
        self.demand_scenarios = None  # [scenario][destination]
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
        S = self.num_scenarios

        # Supplies
        self.supplies = np.random.randint(
            self.supply_range[0], self.supply_range[1] + 1, size=I
        ).tolist()

        # Recourse capacity per source (fraction of supply)
        self.recourse_capacities = [
            int(s * self.recourse_capacity_fraction) for s in self.supplies
        ]

        # Base demands (mean)
        self.base_demands = np.random.randint(
            self.base_demand_range[0], self.base_demand_range[1] + 1, size=J
        ).tolist()

        # Ensure feasibility: total supply >= max total demand across scenarios
        total_supply = sum(self.supplies) + sum(self.recourse_capacities)
        max_demand = sum(d * (1 + self.demand_noise_pct) for d in self.base_demands)
        if total_supply < max_demand * 1.2:
            scale = (max_demand * 1.3) / total_supply
            self.supplies = [int(s * scale) for s in self.supplies]
            self.recourse_capacities = [
                int(s * self.recourse_capacity_fraction) for s in self.supplies
            ]

        # Cost matrices
        self.base_costs = np.random.uniform(
            self.base_cost_range[0], self.base_cost_range[1],
            size=(I, J)
        ).round(2).tolist()

        self.recourse_costs = [
            [round(c * self.recourse_cost_multiplier, 2) for c in row]
            for row in self.base_costs
        ]

        # Generate demand scenarios (perturbations of base demand)
        self.demand_scenarios = []
        for _ in range(S):
            noise = np.random.uniform(
                1 - self.demand_noise_pct, 1 + self.demand_noise_pct, size=J
            )
            scenario_demands = [
                max(1, round(self.base_demands[j] * noise[j]))
                for j in range(J)
            ]
            self.demand_scenarios.append(scenario_demands)

        # Big-M for chance constraints: max possible demand in any scenario
        self.big_M = max(
            max(self.demand_scenarios[s][j] for s in range(S))
            for j in range(J)
        ) * 2

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _x_idx(self, i, j):
        """Index for base shipment x_ij."""
        return i * self.num_dests + j

    def _y_idx(self, i, j, s):
        """Index for recourse shipment y_ij_s."""
        return self.n_x + s * (self.num_sources * self.num_dests) + i * self.num_dests + j

    def _z_idx(self, j, s):
        """Index for violation indicator z_js."""
        return self.n_x + self.n_y + j * self.num_scenarios + s

    # ------------------------------------------------------------------
    # Variable metadata (names, types, bounds, objective) — no dense A
    # ------------------------------------------------------------------

    def _build_var_metadata(self):
        I = self.num_sources
        J = self.num_dests
        S = self.num_scenarios

        self.var_names = []
        for i in range(I):
            for j in range(J):
                self.var_names.append(f"Base_S{i}_D{j}")
        for s in range(S):
            for i in range(I):
                for j in range(J):
                    self.var_names.append(f"Recourse_S{i}_D{j}_Sc{s}")
        for j in range(J):
            for s in range(S):
                self.var_names.append(f"Violate_D{j}_Sc{s}")

        self.var_types = (
            ["Continuous"] * self.n_x
            + ["Continuous"] * self.n_y
            + ["Binary"] * self.n_z
        )
        self.var_bounds = (
            [(0.0, None)] * self.n_x
            + [(0.0, None)] * self.n_y
            + [(0.0, 1.0)] * self.n_z
        )

        self.c = np.zeros(self.n)
        for i in range(I):
            for j in range(J):
                self.c[self._x_idx(i, j)] = self.base_costs[i][j]
        for s in range(S):
            for i in range(I):
                for j in range(J):
                    self.c[self._y_idx(i, j, s)] = self.recourse_costs[i][j] / S

    # ------------------------------------------------------------------
    # Direct Gurobi solve (sparse construction, bypasses dense A)
    # ------------------------------------------------------------------

    def _solve_direct(self, time_limit: float = 300.0):
        import gurobipy as gp
        from gurobipy import GRB

        I = self.num_sources
        J = self.num_dests
        S = self.num_scenarios
        M = self.big_M
        max_violations = int(np.floor((1 - self.alpha) * S))

        m = gp.Model("Stochastic_Transportation")
        m.setParam("OutputFlag", 0)
        m.setParam("LogToConsole", 0)
        if time_limit:
            m.setParam("TimeLimit", time_limit)

        # Variables
        x = {}
        for i in range(I):
            for j in range(J):
                x[(i, j)] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"Base_S{i}_D{j}")

        y = {}
        for s in range(S):
            for i in range(I):
                for j in range(J):
                    y[(i, j, s)] = m.addVar(lb=0, vtype=GRB.CONTINUOUS,
                                            name=f"Recourse_S{i}_D{j}_Sc{s}")

        z = {}
        for j in range(J):
            for s in range(S):
                z[(j, s)] = m.addVar(vtype=GRB.BINARY, name=f"Violate_D{j}_Sc{s}")

        # Objective
        m.setObjective(
            gp.quicksum(self.base_costs[i][j] * x[(i, j)]
                        for i in range(I) for j in range(J))
            + (1.0 / S) * gp.quicksum(self.recourse_costs[i][j] * y[(i, j, s)]
                                       for s in range(S) for i in range(I) for j in range(J)),
            GRB.MINIMIZE,
        )

        # C1: Supply (base)
        for i in range(I):
            m.addConstr(
                gp.quicksum(x[(i, j)] for j in range(J)) <= self.supplies[i],
                name=f"SupplyBase_S{i}",
            )

        # C2: Supply (recourse)
        for i in range(I):
            for s in range(S):
                m.addConstr(
                    gp.quicksum(y[(i, j, s)] for j in range(J)) <= self.recourse_capacities[i],
                    name=f"SupplyRecourse_S{i}_Sc{s}",
                )

        # C3: Demand chance
        for j in range(J):
            for s in range(S):
                m.addConstr(
                    gp.quicksum(x[(i, j)] + y[(i, j, s)] for i in range(I))
                    + M * z[(j, s)] >= self.demand_scenarios[s][j],
                    name=f"DemandChance_D{j}_Sc{s}",
                )

        # C4: Violation budget
        for j in range(J):
            m.addConstr(
                gp.quicksum(z[(j, s)] for s in range(S)) <= max_violations,
                name=f"ViolationBudget_D{j}",
            )

        m.optimize()

        self.solver_status = int(m.Status)
        if m.Status == GRB.TIME_LIMIT:
            raise TimeoutError(f"Gurobi hit the {time_limit}s time limit")
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi failed with status {m.Status}")

        # Extract solution in var_names order
        self.x_star = np.array([m.getVarByName(n).X for n in self.var_names])
        self.obj_val = float(m.ObjVal)

    # ------------------------------------------------------------------
    # Build sparse serialization for to_json_dict (constraint/variable data)
    # ------------------------------------------------------------------

    def _build_sparse_serialization(self):
        """Build senses, constr_names, b, and a sparse A for to_json_dict."""
        I = self.num_sources
        J = self.num_dests
        S = self.num_scenarios
        M = self.big_M
        max_violations = int(np.floor((1 - self.alpha) * S))

        self.constr_names = []
        self.senses = []
        b_list = []

        # We store A as a dict-of-dicts: row_idx -> {col_idx: coeff}
        self._A_sparse = {}
        row = 0

        # C1: Supply (base)
        for i in range(I):
            self._A_sparse[row] = {self._x_idx(i, j): 1.0 for j in range(J)}
            b_list.append(float(self.supplies[i]))
            self.senses.append("<=")
            self.constr_names.append(f"SupplyBase_S{i}")
            row += 1

        # C2: Supply (recourse)
        for i in range(I):
            for s in range(S):
                self._A_sparse[row] = {self._y_idx(i, j, s): 1.0 for j in range(J)}
                b_list.append(float(self.recourse_capacities[i]))
                self.senses.append("<=")
                self.constr_names.append(f"SupplyRecourse_S{i}_Sc{s}")
                row += 1

        # C3: Demand chance
        for j in range(J):
            for s in range(S):
                coeffs = {}
                for i in range(I):
                    coeffs[self._x_idx(i, j)] = 1.0
                    coeffs[self._y_idx(i, j, s)] = 1.0
                coeffs[self._z_idx(j, s)] = float(M)
                self._A_sparse[row] = coeffs
                b_list.append(float(self.demand_scenarios[s][j]))
                self.senses.append(">=")
                self.constr_names.append(f"DemandChance_D{j}_Sc{s}")
                row += 1

        # C4: Violation budget
        for j in range(J):
            self._A_sparse[row] = {self._z_idx(j, s): 1.0 for s in range(S)}
            b_list.append(float(max_violations))
            self.senses.append("<=")
            self.constr_names.append(f"ViolationBudget_D{j}")
            row += 1

        self.b = np.array(b_list)
        # Set A to None — override to_json_dict to use sparse version
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
                lb, ub = (0.0, 1.0)
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
    # Instance data for JSON
    # ------------------------------------------------------------------

    def _get_instance_data(self):
        return {
            "num_sources": self.num_sources,
            "num_destinations": self.num_dests,
            "num_scenarios": self.num_scenarios,
            "confidence_level": self.alpha,
            "max_violations_per_dest": int(np.floor((1 - self.alpha) * self.num_scenarios)),
            "supplies": self.supplies,
            "recourse_capacities": self.recourse_capacities,
            "base_demands": self.base_demands,
            "base_costs": self.base_costs,
            "recourse_costs": self.recourse_costs,
            "demand_scenarios": self.demand_scenarios,
        }

    # ------------------------------------------------------------------
    # Gold solution code
    # ------------------------------------------------------------------

    def generate_gurobi_code_reference(self, data: dict) -> str:
        inst = data.get("instance_data", {})
        meta = data.get("meta", {})

        I = inst.get("num_sources", 0)
        J = inst.get("num_destinations", 0)
        S = inst.get("num_scenarios", 0)
        alpha = inst.get("confidence_level", 0.9)
        supplies = inst.get("supplies", [])
        recourse_caps = inst.get("recourse_capacities", [])
        base_costs = inst.get("base_costs", [])
        recourse_costs = inst.get("recourse_costs", [])
        demand_scenarios = inst.get("demand_scenarios", [])

        sense = "GRB.MINIMIZE" if meta.get("goal") == "MINIMIZE" else "GRB.MAXIMIZE"

        lines = []
        lines.append("```python")
        lines.append("import math")
        lines.append("from gurobipy import *")
        lines.append("")
        lines.append("def solve_problem():")
        lines.append('    m = Model("Stochastic_Transportation")')
        lines.append("    m.setParam('TimeLimit', 300)")
        lines.append("")
        lines.append("    # --- Data ---")
        lines.append(f"    I = {I}  # sources")
        lines.append(f"    J = {J}  # destinations")
        lines.append(f"    S = {S}  # scenarios")
        lines.append(f"    alpha = {alpha}  # confidence level")
        lines.append(f"    max_violations = int(math.floor((1 - alpha) * S))")
        lines.append("")
        lines.append(f"    supplies = {supplies}")
        lines.append(f"    recourse_caps = {recourse_caps}")
        lines.append(f"    base_costs = {base_costs}")
        lines.append(f"    recourse_costs = {recourse_costs}")
        lines.append(f"    demand_scenarios = {demand_scenarios}")
        lines.append("")
        lines.append("    # Big-M: largest demand * 2")
        lines.append("    M = max(demand_scenarios[s][j] for s in range(S) for j in range(J)) * 2")
        lines.append("")
        lines.append("    # --- Variables ---")
        lines.append("    # Stage 1: base shipments")
        lines.append("    x = {}")
        lines.append("    for i in range(I):")
        lines.append("        for j in range(J):")
        lines.append('            x[(i,j)] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"Base_S{i}_D{j}")')
        lines.append("")
        lines.append("    # Stage 2: recourse shipments per scenario")
        lines.append("    y = {}")
        lines.append("    for s in range(S):")
        lines.append("        for i in range(I):")
        lines.append("            for j in range(J):")
        lines.append('                y[(i,j,s)] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"Recourse_S{i}_D{j}_Sc{s}")')
        lines.append("")
        lines.append("    # Violation indicators")
        lines.append("    z = {}")
        lines.append("    for j in range(J):")
        lines.append("        for s in range(S):")
        lines.append('            z[(j,s)] = m.addVar(vtype=GRB.BINARY, name=f"Violate_D{j}_Sc{s}")')
        lines.append("")
        lines.append("    # --- Objective: minimize base cost + expected recourse cost ---")
        lines.append("    m.setObjective(")
        lines.append("        quicksum(base_costs[i][j] * x[(i,j)] for i in range(I) for j in range(J))")
        lines.append("        + (1.0 / S) * quicksum(recourse_costs[i][j] * y[(i,j,s)]")
        lines.append("                               for s in range(S) for i in range(I) for j in range(J)),")
        lines.append(f"        {sense}")
        lines.append("    )")
        lines.append("")
        lines.append("    # --- Constraints ---")
        lines.append("    # C1: Supply (base)")
        lines.append("    for i in range(I):")
        lines.append("        m.addConstr(")
        lines.append("            quicksum(x[(i,j)] for j in range(J)) <= supplies[i],")
        lines.append('            name=f"SupplyBase_S{i}")')
        lines.append("")
        lines.append("    # C2: Supply (recourse per scenario)")
        lines.append("    for i in range(I):")
        lines.append("        for s in range(S):")
        lines.append("            m.addConstr(")
        lines.append("                quicksum(y[(i,j,s)] for j in range(J)) <= recourse_caps[i],")
        lines.append('                name=f"SupplyRecourse_S{i}_Sc{s}")')
        lines.append("")
        lines.append("    # C3: Demand chance constraint (SAA)")
        lines.append("    for j in range(J):")
        lines.append("        for s in range(S):")
        lines.append("            m.addConstr(")
        lines.append("                quicksum(x[(i,j)] + y[(i,j,s)] for i in range(I))")
        lines.append("                + M * z[(j,s)] >= demand_scenarios[s][j],")
        lines.append('                name=f"DemandChance_D{j}_Sc{s}")')
        lines.append("")
        lines.append("    # C4: Violation budget per destination")
        lines.append("    for j in range(J):")
        lines.append("        m.addConstr(")
        lines.append("            quicksum(z[(j,s)] for s in range(S)) <= max_violations,")
        lines.append('            name=f"ViolationBudget_D{j}")')
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
        S = inst.get("num_scenarios", 0)
        alpha = inst.get("confidence_level", 0.9)

        user_content = f"""Write a business memo for a shipping problem under demand uncertainty.

**Company Situation:**
- We have {I} source warehouses and {J} destination stores.
- Customer demand is uncertain — we have {S} forecast scenarios.
- Confidence level for meeting demand: {alpha*100:.0f}%

**Two-Phase Decision:**
- Phase 1: Lock in a base shipping plan before demand is known.
- Phase 2: Use emergency (more expensive) shipments to cover shortfalls after demand is revealed.

**Reliability Requirement:**
- Each store's demand must be fully satisfied in at least {alpha*100:.0f}% of the scenarios.
- It is acceptable to miss demand in up to {(1-alpha)*100:.0f}% of scenarios.

**Business Goal:**
Find the most cost-effective plan that:
- Doesn't exceed warehouse inventory (base and emergency separately)
- Meets each store's demand in at least {alpha*100:.0f}% of scenarios
- Minimizes total expected cost = base shipping + average emergency shipping

**Your Task:**
Write a professional business memo describing this challenge.
IMPORTANT: Do NOT include any specific numbers or data values in your text.
Use ONLY these placeholders in an Annex section:
{{NUM_SOURCES}}, {{NUM_DESTINATIONS}}, {{NUM_SCENARIOS}}, {{CONFIDENCE_LEVEL}},
{{SUPPLIES}}, {{RECOURSE_CAPACITIES}}, {{BASE_COST_MATRIX}}, {{RECOURSE_COST_MATRIX}},
{{DEMAND_SCENARIOS}}

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
            "NUM_SOURCES", "NUM_DESTINATIONS", "NUM_SCENARIOS", "CONFIDENCE_LEVEL",
            "SUPPLIES", "RECOURSE_CAPACITIES", "BASE_COST_MATRIX",
            "RECOURSE_COST_MATRIX", "DEMAND_SCENARIOS",
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
        fmt["NUM_SCENARIOS"] = str(inst.get("num_scenarios", ""))
        fmt["CONFIDENCE_LEVEL"] = f"{inst.get('confidence_level', 0.9) * 100:.0f}%"

        # Supplies
        fmt["SUPPLIES"] = f"Warehouse capacities (base): {inst.get('supplies', [])}"

        # Recourse capacities
        fmt["RECOURSE_CAPACITIES"] = f"Emergency shipping capacities: {inst.get('recourse_capacities', [])}"

        # Base cost matrix
        base_costs = inst.get("base_costs", [])
        I = len(base_costs)
        J = len(base_costs[0]) if base_costs else 0
        bc_lines = [f"Base Shipping Costs ({I} sources × {J} destinations):"]
        header = "Source\\Dest | " + " | ".join(f"D{j}" for j in range(J))
        bc_lines.append(header)
        bc_lines.append("-" * len(header))
        for i, row in enumerate(base_costs):
            row_str = f"S{i:<9} | " + " | ".join(f"{v:6.2f}" for v in row)
            bc_lines.append(row_str)
        fmt["BASE_COST_MATRIX"] = "\n".join(bc_lines)

        # Recourse cost matrix
        recourse_costs = inst.get("recourse_costs", [])
        rc_lines = [f"Emergency Shipping Costs ({I} sources × {J} destinations):"]
        rc_lines.append(header)
        rc_lines.append("-" * len(header))
        for i, row in enumerate(recourse_costs):
            row_str = f"S{i:<9} | " + " | ".join(f"{v:6.2f}" for v in row)
            rc_lines.append(row_str)
        fmt["RECOURSE_COST_MATRIX"] = "\n".join(rc_lines)

        # Demand scenarios
        demand_scenarios = inst.get("demand_scenarios", [])
        S = len(demand_scenarios)
        ds_lines = [f"Demand Scenarios ({S} scenarios × {J} destinations):"]
        ds_header = "Scenario   | " + " | ".join(f"D{j}" for j in range(J))
        ds_lines.append(ds_header)
        ds_lines.append("-" * len(ds_header))
        for s, row in enumerate(demand_scenarios):
            row_str = f"Sc{s:<8} | " + " | ".join(f"{v:>4}" for v in row)
            ds_lines.append(row_str)
        fmt["DEMAND_SCENARIOS"] = "\n".join(ds_lines)

        result = llm_description
        for ph, val in fmt.items():
            result = result.replace(f"{{{ph}}}", val)

        remaining = [p for p in required if f"{{{p}}}" in result]
        if remaining:
            print(f"Placeholders not filled: {remaining}")
            return None

        return result
