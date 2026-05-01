import numpy as np
import time

from main.generation.base_problem import BaseProblem


class CapitalBudgetingGenerator(BaseProblem):
    """
    Multi-Constraint Capital Budgeting / Knapsack Problem (MIP).

    Select a subset of projects/investments to maximize total return,
    subject to multiple resource constraints (budget, labor, risk, etc.).

    Key differences from resource_allocation LP:
    - Binary or integer variables (select/reject decisions)
    - Multiple knapsack-style constraints (all <=, positive coefficients)
    - MAXIMIZE total return/profit
    - No instance_data (unstructured — all data embedded in narrative)

    Decision Variables:
    - Var_i: Binary (1 = select project i, 0 = reject) or
             Integer (quantity of investment i to purchase, bounded)

    Constraints:
    - Resource constraints: sum of resource usage <= budget/capacity
    - Optional: at-most/at-least selection constraints

    Objective: Maximize sum of returns
    """

    SYSTEM_PROMPT = """You are a Senior Investment Portfolio Manager writing a memo about capital allocation decisions.

**CORE PHILOSOPHY: TOTAL ABSTRACTION**
The output must look exactly like a memo from a business executive describing investment opportunities.
It must NOT look like a math problem. There should be no references to 'Var_0', 'C1', or 'Objective Functions'.

**Naming & Creativity Rules:**
1. **Invent Your Own Names**: Give each investment/project a vivid thematic name
   (e.g., 'Meridian Solar Farm Expansion', 'Lake District Warehouse Retrofit', 'Alpine Data Center Build').
2. **Strict Implicit Ordering**: Describe investments in the **exact order** they appear in the JSON.
   The first investment you describe MUST correspond to Var_0, the second to Var_1, etc.
3. **No Technical IDs**: Do NOT include mapping tables or JSON references.

**Structure Strategy (Entity-Centric):**
1. **The Introduction**: Set the scene — a firm evaluating capital projects.
   State the goal (maximize total returns/NPV/value).
2. **The Opportunities (Variables)**: Dedicate one paragraph or bullet to each project.
   For Binary: describe as a go/no-go decision.
   For Integer: describe as a quantity (number of units, lots, or tranches to acquire).
   Include: expected return per unit, and how much of each shared resource it consumes.
   Embed all numerical coefficients naturally in business language.
3. **The Constraints (Resource Limits)**: Conclude with a 'Budget & Resource Limits' section
   listing the total capacity for each shared resource (budget, labor hours, risk exposure, etc.).

**Non-Redundancy**: No constraint or variable described more than once.

**Completeness Check**: Include every coefficient, bound, and RHS value from the JSON, disguised as business data."""

    def __init__(
        self,
        n_vars: int = 8,
        n_constrs: int = 4,
        integer_fraction: float = 0.3,
        sparsity: float = 0.15,
        goal: str = "MAXIMIZE",
    ):
        """
        Args:
            n_vars: Number of investment opportunities.
            n_constrs: Number of resource constraints.
            integer_fraction: Fraction of variables that are Integer (rest Binary).
            sparsity: Fraction of zero entries in constraint matrix.
        """
        # Determine which variables are Integer vs Binary
        n_integer = max(0, int(n_vars * integer_fraction))
        integer_indices = sorted(np.random.choice(n_vars, n_integer, replace=False).tolist()) if n_integer > 0 else []

        prob_type = "LP"  # BaseProblem uses LP for mixed problems too
        super().__init__(n_vars, n_constrs, prob_type=prob_type, goal=goal,
                         problem_type="capital budgeting")

        self.integer_indices = integer_indices
        self.sparsity = np.clip(sparsity, 0.0, 0.5)  # Keep sparsity moderate

    def generate(self, time_limit: float = 300.0):
        max_retries = 20
        start_time = time.time()

        for _ in range(max_retries):
            if time.time() - start_time > time_limit:
                break

            self._build_problem()

            try:
                self.solve(time_limit=time_limit)
            except Exception:
                continue

            # Ensure non-trivial solution
            if self.obj_val is not None and abs(self.obj_val) > 1e-4:
                return self

        if self.obj_val is None:
            pass
        print(f"Warning: Could not generate non-trivial problem after {max_retries} retries.")
        return self

    def _build_problem(self):
        n = self.n
        m = self.m

        # ---- Variable types ----
        self.var_types = []
        for i in range(n):
            if i in self.integer_indices:
                self.var_types.append("Integer")
            else:
                self.var_types.append("Binary")

        # ---- Variable bounds ----
        self.var_bounds = []
        for i in range(n):
            if i in self.integer_indices:
                ub = float(np.random.randint(3, 12))
                self.var_bounds.append((0.0, ub))
            else:
                self.var_bounds.append((0.0, 1.0))

        # ---- Objective coefficients (returns/profits) ----
        # Positive returns, varying magnitude
        self.c = np.round(np.random.uniform(2.0, 25.0, size=n), 2)

        # ---- Constraint matrix (resource usage) ----
        # All positive coefficients (resource consumption), knapsack-style
        A = np.random.uniform(1.0, 15.0, size=(m, n))

        # Apply sparsity
        if self.sparsity > 0:
            mask = np.random.rand(m, n) < self.sparsity
            A[mask] = 0.0

        # Ensure no all-zero rows
        for i in range(m):
            if np.allclose(A[i], 0.0):
                col = np.random.randint(0, n)
                A[i, col] = np.random.uniform(1.0, 15.0)

        self.A = np.round(A, 2)

        # ---- Constraint senses (all <=, knapsack-style) ----
        self.senses = ["<="] * m

        # ---- RHS (resource budgets) ----
        # Set budgets so that roughly 50-70% of items can be selected
        # Use anchor-based approach for feasibility
        anchor = np.zeros(n)
        for i in range(n):
            if self.var_types[i] == "Binary":
                anchor[i] = 1.0 if np.random.random() < 0.6 else 0.0
            else:
                anchor[i] = np.random.uniform(1.0, self.var_bounds[i][1] * 0.6)

        dot = self.A @ anchor
        # Add moderate slack so the anchor is feasible but constraints are binding
        slack = np.random.uniform(1.0, 8.0, size=m)
        self.b = np.round(dot + slack, 2)

        # ---- Constraint names ----
        self.constr_names = [f"C{i}" for i in range(m)]
