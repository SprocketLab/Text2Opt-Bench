import numpy as np
import time

from main.generation.base_problem import BaseProblem


class ResourceAllocationGenerator(BaseProblem):
    """
    Forward/Anchor resource allocation generator (LP only).
    - Build A with requested sparsity (fraction of zeros).
    - Sample a nonnegative anchor x_anchor.
    - Set b = A @ x_anchor + positive slack to guarantee feasibility.
    - Random objective c in [0, 10].
    """

    SYSTEM_PROMPT = """You are a Senior Operations Research Consultant.
Your task is to translate a structured JSON resource allocation optimization dataset into a strictly natural language business problem. This dataset represents a Linear Programming (LP) model (no quadratic or nonlinear terms).

**CORE PHILOSOPHY: TOTAL ABSTRACTION**
The output must look exactly like an email or memo from a business client.
It must NOT look like a math problem. There should be no references to 'Var_0', 'C1_Res', or 'Objective Functions'.

**Naming & Creativity Rules:**
1. **Invent Your Own Names**: You have total freedom to name the decision variables and constraints to fit the scenario.
Use thematic names (e.g., 'Blue Bond Portfolio', 'Distillation Column B', 'Union Overtime Shifts').
2. **Strict Implicit Ordering**: While you must use creative names, you must describe the entities in the **exact order** they appear in the JSON.
The first entity you describe MUST correspond to the data for Var_0, the second for Var_1, etc. This is crucial for the logic to hold.
3. **No Technical IDs**: Do NOT include a mapping table or any reference to the original JSON IDs. The text should be purely qualitative.

**Structure Strategy (Entity-Centric):**
1. **The Introduction**: Set the scene for a resource allocation decision (e.g., allocating budgets, equipment, or manpower). State the goal (minimize cost/maximize benefit).
2. **The Levers (Variables)**: Dedicate one distinct paragraph (or bullet point) to each decision variable.
Describe its costs, revenues, and how it impacts every single constraint (resource usage, pollution, quality, etc.).
Embed the numerical coefficients naturally into sentences.
3. **The Limits (Constraints)**: Conclude with a 'Facility Specs' or 'Market Limits' section that lists the total capacities (RHS values) for the resources mentioned above.

**Non-Redundancy**: Ensure that no constraint or variable is described more than once. The description should be concise and avoid repetition.

**Completeness Check**: You must include every single coefficient, bound, and RHS value from the JSON, disguised as business data."""

    def __init__(
        self,
        n_vars: int,
        n_constrs: int,
        integer_indices=None,
        sparsity: float = 0.3,
        value_low: float = -10.0,
        value_high: float = 10.0,
        goal: str = "MAXIMIZE",
        ub_min: float = 5.0,
        ub_max: float = 20.0,
    ):
        # Resource allocation is LP-only in this forward/anchor generator.
        super().__init__(n_vars, n_constrs, prob_type="LP", goal=goal, problem_type="resource allocation")
        self.integer_indices = integer_indices if integer_indices else []
        self.sparsity = np.clip(sparsity, 0.0, 1.0)
        self.value_low = value_low
        self.value_high = value_high
        self.ub_min = ub_min
        self.ub_max = max(ub_min, ub_max)

    def generate(self, time_limit: float = 300.0):
        max_retries = 20
        start_time = time.time()
        
        for _ in range(max_retries):
            # Check total time limit
            if time.time() - start_time > time_limit:
                print(f"Warning: Generation time limit ({time_limit}s) exceeded.")
                break
                
            self._build_matrix_A()
            self._build_anchor_and_b()
            self._build_objective()

            # Solve and store optimal solution
            try:
                # Respect a hard time limit so generation never gets stuck.
                self.solve(time_limit=time_limit)
            except Exception as exc:
                # If solve fails (unbounded/infeasible/time limit), retry
                if isinstance(exc, TimeoutError):
                    print(f"    [timeout] solver exceeded {time_limit}s, retrying with new sample...")
                continue
            
            # Check for trivial zero solution (if theoretical optimum is 0)
            # We want non-trivial problems. 
            # Note: For some problems, 0 might be the only feasible solution, 
            # but we try to avoid it if possible.
            
            # Use self.obj_val directly as self.solve() sets it
            opt_val = self.obj_val if self.obj_val is not None else 0.0
            
            # Check if non-trivial (absolute value > epsilon)
            if abs(opt_val) > 1e-4:
                return self
        
        # If we exhausted retries or time, we might return a trivial or potentially failed state.
        # But if self.obj_val is None, we haven't solved any successfully.
        if self.obj_val is None:
            # Try one last build that is guaranteed simple if possible, or just fail gracefully?
            # For now, print warning.
            pass

        print(f"Warning: Could not generate non-trivial problem after {max_retries} retries.")
        return self

    def _build_matrix_A(self):
        # Start with dense values
        A = np.random.uniform(self.value_low, self.value_high, size=(self.m, self.n))

        # Apply sparsity mask: probability of zero = sparsity
        if self.sparsity > 0:
            mask = np.random.rand(self.m, self.n) < self.sparsity
            A[mask] = 0.0

        # Ensure no all-zero rows
        for i in range(self.m):
            if np.allclose(A[i], 0.0):
                # Replace one random column with a non-zero coefficient
                col = np.random.randint(0, self.n)
                A[i, col] = np.random.uniform(
                    self.value_low if self.value_low < 0 else -1.0,
                    self.value_high,
                )

        self.A = np.round(A, 2)
        
        # Mixed constraints logic
        self.senses = []
        if self.goal == "MINIMIZE":
            # 70% >=, 25% <=, 5% =
            choices = [">=", "<=", "="]
            probs = [0.70, 0.25, 0.05]
        else:
            # 70% <=, 25% >=, 5% =
            choices = ["<=", ">=", "="]
            probs = [0.70, 0.25, 0.05]
            
        self.senses = list(np.random.choice(choices, size=self.m, p=probs))

        self.var_types = [
            "Integer" if i in self.integer_indices else "Continuous"
            for i in range(self.n)
        ]

    def _build_anchor_and_b(self):
        # Anchor is nonnegative; integer vars are rounded to keep feasibility for MIP
        anchor = np.random.uniform(0.0, 10.0, size=self.n)
        for idx in self.integer_indices:
            anchor[idx] = np.round(anchor[idx])
        self.anchor_x = np.round(anchor, 2)

        # Tighter slack to avoid overly loose constraints (keeps more vars active)
        slack = np.random.uniform(0.5, 3.0, size=self.m)
        self.slack = np.round(slack, 2)

        dot_product = np.dot(self.A, self.anchor_x)
        
        # Calculate b based on individual constraint senses
        b_vals = []
        for i, sense in enumerate(self.senses):
            if sense == "<=":
                # Ax <= b  =>  b = Ax + slack
                val = dot_product[i] + self.slack[i]
            elif sense == ">=":
                # Ax >= b  =>  b = Ax - slack
                val = dot_product[i] - self.slack[i]
            else: # "="
                # Ax = b   =>  b = Ax
                val = dot_product[i]
            b_vals.append(val)
            
        self.b = np.round(np.array(b_vals), 2)

        # Variable bounds: finite, above anchor to maintain feasibility,
        # with a positive lower bound to reduce zero-valued optima.
        var_bounds = []
        for i in range(self.n):
            cand_ub = np.random.uniform(self.ub_min, self.ub_max)
            ub = max(cand_ub, self.anchor_x[i] + 1.0)
            var_bounds.append((0.0, round(ub, 2)))
        self.var_bounds = var_bounds

    def _build_objective(self):
        # Positive coefficients with maximize goal encourages utilization (fewer zeros).
        c = np.random.uniform(1.0, 10.0, size=self.n)
        self.c = np.round(c, 2)

