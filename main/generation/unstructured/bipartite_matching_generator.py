import numpy as np
from main.generation.base_problem import BaseProblem


class BipartiteMatchingGenerator(BaseProblem):
    """
    Generator for Bipartite Matching / Transportation problems.
    Structure: Bipartite block matrix (Sources x Destinations).
    - Variables: x_ij (flow from source i to destination j).
    - Supply Constraints: sum_j x_ij <= Supply_i
    - Demand Constraints: sum_i x_ij >= Demand_j
    - Objective: Minimize sum c_ij * x_ij
    """

    SYSTEM_PROMPT = """You are a Supply Chain Director writing a concise memo about a shipping logistics challenge.

**TASK:** Write a concise and professional business memo describing a transportation/distribution problem.

**CRITICAL INSTRUCTION - TEMPLATE GENERATION:**
- You must generate a **text template** that uses specific placeholders for data.
- **DO NOT** attempt to describe the specific data values (numbers, matrices) in the narrative.
- **DO NOT** invent numbers. Use the placeholders.
- **DO NOT** write headers or titles for the data placeholders. The placeholders will be replaced by formatted tables/lists including headers.
- **DO NOT** embed large data matrices in the middle of sentences.
- Place all data placeholders in a dedicated "ANNEX" or "DATA" section at the end of the memo to maintain realism.

**TONE:** Professional business memo. Clear and practical. Something a logistics manager would actually write.

**REQUIRED PLACEHOLDERS (include ALL - these will be filled with data):**
{SUPPLIES}, {DEMANDS}, {COST_MATRIX}

**SCENARIO TO DESCRIBE:**
- Company has multiple source locations (warehouses, factories, distribution centers)
- Need to ship products to multiple destination locations (stores, customers, regional hubs)
- Each source has limited inventory/capacity
- Each destination has a required quantity they need
- Shipping costs vary by route
- Goal: Figure out the cheapest way to meet all demands

**WHAT TO INCLUDE:**
- Business context: Why this matters (cost savings, customer satisfaction)
- The logistics challenge in plain language
- The goal: minimize total shipping cost while meeting all requirements

**WHAT TO AVOID:**
- Mathematical notation or formulas
- Technical terms like "objective function", "constraints", "decision variables"
- Phrases like "MINIMIZE", "subject to", "≤", "≥"
- Anything that sounds like an optimization textbook
- The word "continuous" or "integer" when describing shipping quantities (just describe naturally)

**VARIABLE TYPE NOTE:**
You'll be told if shipments must be in whole units (like pallets or truckloads) or can be fractional. Mention this naturally, e.g., "ship in whole pallet quantities" or "partial shipments are allowed".

**SUGGESTED STRUCTURE:**
1. **Subject/Date:** Standard memo header.
2. **Situation:** Brief context on the supply chain need.
3. **Problem:** Concise statement of the distribution challenge.
4. **Annex - Shipping Data:**
   {SUPPLIES}
   {DEMANDS}
   {COST_MATRIX}
"""

    def __init__(
        self,
        num_sources: int,
        num_dests: int,
        supply_range: tuple = (20, 100),
        demand_range: tuple = (10, 80),
        cost_range: tuple = (5, 30),
        is_integer: bool = False,
    ):
        self.num_sources = num_sources
        self.num_dests = num_dests
        self.supply_range = supply_range
        self.demand_range = demand_range
        self.cost_range = cost_range
        self.is_integer = is_integer

        n_vars = num_sources * num_dests
        n_constrs = num_sources + num_dests

        problem_label = "Transportation Problem"
        prob_type = "LP"

        super().__init__(
            n_vars=n_vars,
            n_constrs=n_constrs,
            prob_type=prob_type,
            goal="MINIMIZE",
            problem_type=problem_label,
        )

        self.supplies = None
        self.demands = None
        self.cost_matrix = None

    def generate(self, time_limit: float = 300.0):
        """Build the Bipartite Matching / Transportation problem instance."""
        self._build_parameters()
        self._build_model_matrices()

        self.solve(time_limit=time_limit)

        data = self.to_json_dict()
        data["solution_code"] = self.generate_gurobi_code_reference(data)
        return data

    def _build_parameters(self):
        """Generate feasible supply and demand vectors."""
        # 1. Generate Demand first
        self.demands = np.random.randint(
            self.demand_range[0], self.demand_range[1] + 1, size=self.num_dests
        )
        total_demand = self.demands.sum()

        # 2. Generate Supply
        # To guarantee feasibility (Supply >= Demand), we generate supply and then
        # scale/adjust if necessary.
        self.supplies = np.random.randint(
            self.supply_range[0], self.supply_range[1] + 1, size=self.num_sources
        )
        total_supply = self.supplies.sum()

        # Enforce Total Supply >= Total Demand
        if total_supply < total_demand:
            # Add difference to random sources or scale
            diff = total_demand - total_supply
            # Distribute diff + buffer
            buffer = int(diff * 1.1) + 1 if diff > 0 else 0
            # Randomly add to sources
            for _ in range(diff + buffer):
                idx = np.random.randint(0, self.num_sources)
                self.supplies[idx] += 1

        # Costs
        self.cost_matrix = np.random.uniform(
            self.cost_range[0], self.cost_range[1], size=(self.num_sources, self.num_dests)
        ).round(2)

    def _build_model_matrices(self):
        """Construct A, b, c, etc."""
        # Variable Names: Flow_S{i}_D{j}
        self.var_names = []
        for i in range(self.num_sources):
            for j in range(self.num_dests):
                self.var_names.append(f"Flow_S{i}_D{j}")

        # Constraint Names: Supply_S{i}, Demand_D{j}
        self.constr_names = [f"Supply_S{i}" for i in range(self.num_sources)] + \
                            [f"Demand_D{j}" for j in range(self.num_dests)]

        # Matrix A: (m x n)
        # Rows 0..S-1: Supply constraints (sum_j x_ij <= S_i)
        # Rows S..S+D-1: Demand constraints (sum_i x_ij >= D_j)
        self.A = np.zeros((self.m, self.n))

        # Supply rows
        for i in range(self.num_sources):
            # Variables for source i are x_i0, x_i1, ..., x_iD-1
            # Indices in flat list: i*num_dests + j
            start_idx = i * self.num_dests
            end_idx = start_idx + self.num_dests
            self.A[i, start_idx:end_idx] = 1.0

        # Demand rows
        for j in range(self.num_dests):
            # Variables for dest j are x_0j, x_1j, ...
            # Indices: k * num_dests + j for k in 0..num_sources-1
            row_idx = self.num_sources + j
            for i in range(self.num_sources):
                col_idx = i * self.num_dests + j
                self.A[row_idx, col_idx] = 1.0

        # RHS b
        self.b = np.concatenate([self.supplies, self.demands])

        # Senses
        # Supply: <=
        # Demand: >=
        self.senses = ["<="] * self.num_sources + [">="] * self.num_dests

        # Objective c (flattened cost matrix)
        self.c = self.cost_matrix.flatten()

        # Types and Bounds
        if self.is_integer:
            self.var_types = ["Integer"] * self.n
            # Bounds: 0 to infinity (or min(S_i, D_j))
            # Usually safe to leave UB as None or huge
            self.var_bounds = [(0.0, None) for _ in range(self.n)]
        else:
            self.var_types = ["Continuous"] * self.n
            self.var_bounds = [(0.0, None) for _ in range(self.n)]

    def _get_instance_data(self):
        return {
            "integer_flows": self.is_integer,
            "num_sources": self.num_sources,
            "num_destinations": self.num_dests,
            "supplies": self.supplies.tolist(),
            "demands": self.demands.tolist(),
            "cost_matrix": self.cost_matrix.tolist(),
        }

    def generate_gurobi_code_reference(self, data):
        """
        Generate a compact, matrix-based Gurobi script for bipartite matching problems.
        
        Overrides the base class method to produce more readable, matrix-structured code
        using loops and quicksum instead of individual variable declarations.
        """
        instance_data = data.get("instance_data", {})
        meta = data.get("meta", {})
        
        num_sources = instance_data.get("num_sources", 0)
        num_dests = instance_data.get("num_destinations", 0)
        supplies = instance_data.get("supplies", [])
        demands = instance_data.get("demands", [])
        cost_matrix = instance_data.get("cost_matrix", [])
        is_integer = instance_data.get("integer_flows", False)
        goal = meta.get("goal", "MINIMIZE")
        
        vtype = "GRB.INTEGER" if is_integer else "GRB.CONTINUOUS"
        sense = "GRB.MINIMIZE" if goal == "MINIMIZE" else "GRB.MAXIMIZE"
        
        lines = []
        lines.append("```python")
        lines.append("from gurobipy import *")
        lines.append("")
        lines.append("def solve_problem():")
        lines.append(f"    m = Model(\"Transportation_Problem\")")
        lines.append("    m.setParam(\"TimeLimit\", 300)  # Set time limit to 300 seconds")
        lines.append("")
        lines.append("    # Data")
        lines.append(f"    sources = list(range({num_sources}))  # S0..S{num_sources-1}")
        lines.append(f"    destinations = list(range({num_dests}))  # D0..D{num_dests-1}")
        lines.append("")
        lines.append(f"    supplies = {supplies}")
        lines.append(f"    demands = {demands}")
        lines.append("")
        lines.append("    # Cost matrix: rows correspond to sources S0..S{}, columns to destinations D0..D{}".format(
            num_sources-1, num_dests-1))
        lines.append("    costs = [")
        for i, row in enumerate(cost_matrix):
            # Format row with proper alignment
            row_str = "        ["
            row_str += ", ".join(f"{val:6.2f}" for val in row)
            row_str += f"],  # S{i}"
            lines.append(row_str)
        lines.append("    ]")
        lines.append("")
        lines.append("    # Decision variables:")
        lines.append("    # Flow_S{i}_D{j} >= 0 is the quantity shipped from source i to destination j")
        lines.append("    flows = {}")
        lines.append("    for i in sources:")
        lines.append("        for j in destinations:")
        lines.append(f"            flows[(i, j)] = m.addVar(lb=0.0, vtype={vtype}, name=f\"Flow_S{{i}}_D{{j}}\")")
        lines.append("")
        lines.append("    # Objective: Minimize total transportation cost")
        lines.append("    m.setObjective(")
        lines.append("        quicksum(costs[i][j] * flows[(i, j)] for i in sources for j in destinations),")
        lines.append(f"        {sense}")
        lines.append("    )")
        lines.append("")
        lines.append("    # Supply constraints: Shipments from each source <= its supply")
        lines.append("    for i in sources:")
        lines.append("        m.addConstr(")
        lines.append("            quicksum(flows[(i, j)] for j in destinations) <= supplies[i],")
        lines.append(f"            name=f\"Supply_S{{i}}\"")
        lines.append("        )")
        lines.append("")
        lines.append("    # Demand constraints: Shipments to each destination >= its demand")
        lines.append("    for j in destinations:")
        lines.append("        m.addConstr(")
        lines.append("            quicksum(flows[(i, j)] for i in sources) >= demands[j],")
        lines.append(f"            name=f\"Demand_D{{j}}\"")
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
        The LLM should provide a compact description while the matrix structure remains directly accessible.
        """
        import json
        
        clean_data = cls._sanitize_for_prompt(problem_data)
        system_prompt = cls.get_system_prompt()
        
        # Extract instance_data for clear presentation
        instance_data = clean_data.get("instance_data", {})
        meta = clean_data.get("meta", {})
        
        # Build a structured prompt that guides template creation
        num_sources = instance_data.get('num_sources', 0)
        num_dests = instance_data.get('num_destinations', 0)
        # Check both field names for compatibility
        is_integer = instance_data.get('integer_flows', False)
        var_type_str = 'Integer' if is_integer else 'Continuous'
        
        # Extract actual data for examples
        supplies = instance_data.get('supplies', [])
        demands = instance_data.get('demands', [])
        cost_matrix = instance_data.get('cost_matrix', [])
        
        # Format supplies and demands
        supplies_str = str(supplies) if supplies else "[]"
        demands_str = str(demands) if demands else "[]"
        
        # Format first few rows of cost matrix (show first 3 rows, or all if fewer)
        cost_matrix_preview_rows = min(3, len(cost_matrix))
        cost_matrix_preview = []
        for i in range(cost_matrix_preview_rows):
            if i < len(cost_matrix):
                cost_matrix_preview.append(cost_matrix[i])
        
        # Format cost matrix preview as a readable table
        cost_matrix_str = ""
        if cost_matrix_preview and len(cost_matrix_preview) > 0:
            num_cols = len(cost_matrix_preview[0]) if cost_matrix_preview[0] else num_dests
            # Header row
            header = "Source\\Dest | " + " | ".join(f"D{j:2d}" for j in range(min(num_dests, num_cols)))
            cost_matrix_str = f"First {cost_matrix_preview_rows} rows of Cost Matrix:\n{header}\n" + "-" * len(header) + "\n"
            # Data rows
            for i, row in enumerate(cost_matrix_preview):
                row_str = f"S{i:2d}        | " + " | ".join(f"{val:6.2f}" for val in row)
                cost_matrix_str += row_str + "\n"
            if len(cost_matrix) > cost_matrix_preview_rows:
                cost_matrix_str += f"... (and {len(cost_matrix) - cost_matrix_preview_rows} more rows)\n"
        
        # Determine natural language for variable type
        if is_integer:
            shipment_type_desc = "Shipments must be in whole units (no partial loads allowed)"
        else:
            shipment_type_desc = "Partial shipments are allowed (any quantity can be shipped)"

        user_content = f"""Write a business memo for the following logistics situation.

**Company Situation:**
- We have {num_sources} source locations (warehouses/factories)
- We need to supply {num_dests} destination locations (stores/customers)
- {shipment_type_desc}

**Business Goal:**
Find the most cost-effective shipping plan that:
- Doesn't exceed available inventory at any source
- Meets the required quantity at every destination
- Minimizes total shipping costs

**Your Task:**
Write a professional business memo describing this challenge.
IMPORTANT: Do NOT include any specific numbers or data values in your text. You MUST use the provided placeholders in an Annex section.
Placeholders: {{SUPPLIES}}, {{DEMANDS}}, {{COST_MATRIX}}
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
        
        # Required placeholders for bipartite matching
        required_placeholders = ["SUPPLIES", "DEMANDS", "COST_MATRIX"]
        
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
        required_fields = ["supplies", "demands", "cost_matrix"]
        missing_fields = [field for field in required_fields if field not in instance_data]
        
        if missing_fields:
            print(f"❌ Error: Missing required fields in instance_data: {missing_fields}")
            return None
        
        # Format instance data for insertion
        formatted_data = {}
        
        # Format supplies
        supplies = instance_data["supplies"]
        formatted_data["SUPPLIES"] = str(supplies)
        
        # Format demands
        demands = instance_data["demands"]
        formatted_data["DEMANDS"] = str(demands)
        
        # Format cost matrix as a readable table
        cost_matrix = instance_data["cost_matrix"]
        num_sources = len(cost_matrix)
        num_dests = len(cost_matrix[0]) if cost_matrix else 0
        
        # Create a formatted table
        lines = [f"Cost Matrix ({num_sources} sources × {num_dests} destinations):"]
        lines.append("")
        # Header row
        header = "Source\\Dest | " + " | ".join(f"D{j:2d}" for j in range(num_dests))
        lines.append(header)
        lines.append("-" * len(header))
        # Data rows
        for i, row in enumerate(cost_matrix):
            row_str = f"S{i:2d}        | " + " | ".join(f"{val:6.2f}" for val in row)
            lines.append(row_str)
        
        formatted_data["COST_MATRIX"] = "\n".join(lines)
        
        # Fill in placeholders using targeted string replacement
        # This avoids issues with other braces like {i} and {j} in the text
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

    # Inherit to_json_dict from BaseProblem since it now handles instance_data correctly
