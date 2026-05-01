import copy
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from main.utils import save_json


class BaseProblem:
    """
    Shared logic for forward/anchor problem generation and serialization.
    Subclasses should populate:
        - self.A (m x n), self.b (m,), self.c (n,)
        - self.senses: list[str] with entries in {'<=','>=','='}
        - self.var_types: list[str] with entries 'Continuous' or 'Integer'
        - self.anchor_x, self.slack (optional metadata)
    
    For prompt generation, subclasses should override:
        - SYSTEM_PROMPT: class attribute with the LLM system prompt
    """
    
    # Default system prompt - subclasses should override
    SYSTEM_PROMPT = """You are an Operations Research expert.
Your task is to translate a structured JSON optimization dataset into a natural language business problem.
The output should read like a memo from a business client, not a math problem."""

    def __init__(
        self,
        n_vars: int,
        n_constrs: int,
        prob_type: str = "LP",
        goal: str = "MINIMIZE",
        problem_type: str = "generic",
    ):
        self.n = n_vars
        self.m = n_constrs
        self.prob_type = prob_type
        self.goal = goal
        self.problem_type = problem_type
        self.var_names = [f"Var_{i}" for i in range(self.n)]
        self.constr_names = [f"C{i}" for i in range(self.m)]

        # Filled by subclasses
        self.A = None
        self.b = None
        self.c = None
        self.Q = None  # Quadratic objective matrix (n x n)
        self.senses = None
        self.var_types = None
        self.anchor_x = None
        self.slack = None
        self.var_bounds = None  # list of (lb, ub or None)

        # Results
        self.x_star = None
        self.obj_val = None
        self.solver_status = None

    def _build_model(self) -> gp.Model:
        """Construct a Gurobi model from A, b, c, senses, var_types."""
        m = gp.Model("Optimization_Problem")
        m.setParam("OutputFlag", 0)
        m.setParam("LogToConsole", 0)

        gurobi_vars = {}
        for i, name in enumerate(self.var_names):
            vtype_raw = self.var_types[i]
            if vtype_raw == "Binary":
                gurobi_vars[name] = m.addVar(vtype=GRB.BINARY, name=name)
                continue

            if vtype_raw == "Integer":
                vtype = GRB.INTEGER
            else:
                vtype = GRB.CONTINUOUS

            lb, ub = (0.0, None)
            if self.var_bounds is not None:
                lb, ub = self.var_bounds[i]
                if vtype == GRB.INTEGER:
                    lb = np.ceil(lb)
                    if ub is not None:
                        ub = np.floor(ub)
            
            gurobi_vars[name] = m.addVar(lb=lb, ub=ub if ub is not None else GRB.INFINITY, vtype=vtype, name=name)

        # Linear objective
        obj_expr = 0
        if self.c is not None:
            for i, name in enumerate(self.var_names):
                obj_expr += float(self.c[i]) * gurobi_vars[name]

        # Quadratic objective (Standard form: 0.5 * x.T * Q * x)
        if self.Q is not None:
            rows, cols = self.Q.shape
            for i in range(rows):
                for j in range(cols):
                    coeff = float(self.Q[i, j])
                    if abs(coeff) > 1e-9:
                        v_i = gurobi_vars[self.var_names[i]]
                        v_j = gurobi_vars[self.var_names[j]]
                        obj_expr += 0.5 * coeff * v_i * v_j

        sense = GRB.MINIMIZE if self.goal == "MINIMIZE" else GRB.MAXIMIZE
        m.setObjective(obj_expr, sense)

        # Constraints
        for row_idx, cname in enumerate(self.constr_names):
            lhs = 0
            for col_idx, vname in enumerate(self.var_names):
                coeff = float(self.A[row_idx, col_idx])
                if abs(coeff) > 1e-9:
                    lhs += coeff * gurobi_vars[vname]

            rhs = float(self.b[row_idx])
            s = self.senses[row_idx]
            if s == "<=":
                m.addConstr(lhs <= rhs, name=cname)
            elif s == ">=":
                m.addConstr(lhs >= rhs, name=cname)
            else:
                m.addConstr(lhs == rhs, name=cname)

        return m

    def solve(self, time_limit: float | None = 300.0):
        """Solve the generated problem and store x_star and obj_val."""
        model = self._build_model()
        if time_limit is not None:
            # Ensure we never spend more than the provided wall-clock limit.
            model.setParam("TimeLimit", time_limit)
        model.optimize()

        self.solver_status = int(model.Status)

        if model.Status == GRB.TIME_LIMIT:
            raise TimeoutError(f"Gurobi hit the {time_limit}s time limit")
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi failed with status {model.Status}")

        self.x_star = np.array([model.getVarByName(n).X for n in self.var_names])
        self.obj_val = float(model.ObjVal)
        return self.x_star, self.obj_val

    def _get_instance_data(self):
        """Hook for subclasses to provide problem-specific data."""
        return {}

    def to_json_dict(self):
        """Serialize to the JSON structure expected by downstream code."""
        if self.x_star is None or self.obj_val is None:
            raise ValueError("Problem not solved yet. Call solve() first.")

        is_integer = any(vt in ("Integer", "Binary") for vt in self.var_types)

        optimal_values = {
            name: float(self.x_star[i]) for i, name in enumerate(self.var_names)
        }
        
        # Get instance-specific data (e.g. geometric matrices, casualty counts)
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
        
        # If no instance data, we can either keep empty dict or remove it.
        # Keeping it is consistent.

        for i, cname in enumerate(self.constr_names):
            data["constraints"][cname] = {
                "type": "Constraint",
                "sense": self.senses[i],
                "rhs": float(self.b[i]),
            }

        for j, vname in enumerate(self.var_names):
            resource_costs = {}
            for i, cname in enumerate(self.constr_names):
                coeff = float(self.A[i, j])
                if abs(coeff) > 1e-9:
                    resource_costs[cname] = coeff

            # Binary variables must have range [0.0, 1.0]
            if self.var_types[j] == "Binary":
                lb, ub = (0.0, 1.0)
            else:
                lb, ub = (0.0, None)
                if self.var_bounds is not None:
                    lb, ub = self.var_bounds[j]
                    # Ensure integer variables have integer bounds for cleanliness
                    if self.var_types[j] == "Integer":
                        lb = np.ceil(lb)
                        if ub is not None:
                            ub = np.floor(ub)

            ub_val = "inf" if ub is None else float(ub)

            data["variables"][vname] = {
                "type": self.var_types[j],
                "range": [float(lb), ub_val],
                "objective_linear_coefficient": float(self.c[j]) if self.c is not None else 0.0,
                "resource_costs": resource_costs,
            }
        
        if self.Q is not None:
            data["quadratic_matrix_q"] = self.Q.tolist()

        return data

    def generate_gurobi_code_reference(self, data: dict) -> str:
        """
        Generate Gurobi script from problem data.
        
        This is the base implementation that generates verbose, variable-by-variable code.
        Subclasses can override this method to generate more compact, problem-specific code
        (e.g., using matrix structures and loops).
        
        Args:
            data: Problem data dictionary with 'variables', 'constraints', 'meta' keys
            
        Returns:
            String containing the Gurobi Python script
        """
        lines = []
        lines.append("```python")
        lines.append("from gurobipy import *")
        lines.append("")
        lines.append("def solve_problem():")
        lines.append("    # Create a new model")
        lines.append("    m = Model(\"Optimization_Problem\")")
        
        lines.append("")
        
        lines.append("    # 2. Variables")
        lines.append("    vars = {}")
        lines.append("    lin_expr = 0")
        
        # Sort variables for deterministic output
        sorted_vars = sorted(data["variables"].keys())
        
        for v_name in sorted_vars:
            v_info = data["variables"][v_name]
            v_type_raw = v_info["type"]
            if v_type_raw == "Binary":
                lines.append(
                    f"    vars['{v_name}'] = m.addVar(vtype=GRB.BINARY, name='{v_name}')"
                )
                c = v_info["objective_linear_coefficient"]
                if abs(c) > 1e-9:
                    lines.append(f"    lin_expr += {c} * vars['{v_name}']")
                continue
            elif v_type_raw == "Integer":
                v_type = "GRB.INTEGER"
            else:
                v_type = "GRB.CONTINUOUS"

            lb = v_info["range"][0]
            ub_raw = v_info["range"][1]
            
            # Format UB for Gurobi call
            ub = "GRB.INFINITY" if ub_raw == "inf" else ub_raw
            
            lines.append(
                f"    vars['{v_name}'] = m.addVar(lb={lb}, ub={ub}, vtype={v_type}, name='{v_name}')"
            )
            
            c = v_info["objective_linear_coefficient"]
            if abs(c) > 1e-9:
                  lines.append(f"    lin_expr += {c} * vars['{v_name}']")
        
        quad_terms = None
        if data["meta"]["optimization_type"] == "QP":
            lines.append("")
            lines.append("    # Quadratic Terms (using symmetric Q, merged upper triangle)")
            lines.append("    quad_expr = 0")
            quad_terms = []
            Q = data["quadratic_matrix_q"]
            n_q = len(Q)
            for i in range(n_q):
                # Diagonal term: 0.5 * Q_ii * x_i^2
                diag = Q[i][i]
                if abs(diag) > 1e-9:
                    quad_terms.append(
                        f"    quad_expr += 0.5 * {diag} * vars['Var_{i}'] * vars['Var_{i}']"
                    )
                # Off-diagonal terms: Q_ij * x_i * x_j  (since 0.5 * 2 * Q_ij = Q_ij)
                for j in range(i + 1, n_q):
                    val = Q[i][j]
                    if abs(val) > 1e-9:
                        quad_terms.append(
                            f"    quad_expr += {val} * vars['Var_{i}'] * vars['Var_{j}']"
                        )
            lines.extend(quad_terms)

        lines.append("")
        sense = "GRB.MINIMIZE" if data["meta"]["goal"] == "MINIMIZE" else "GRB.MAXIMIZE"
        if data["meta"]["optimization_type"] == "QP":
            lines.append(f"    m.setObjective(lin_expr + quad_expr, {sense})")
        else:
            lines.append(f"    m.setObjective(lin_expr, {sense})")
        lines.append("")
        
        lines.append("    # 3. Constraints")
        constr_lhss = {name: [] for name in data["constraints"].keys()}
        
        for v_name in sorted_vars:
            v_info = data["variables"][v_name]
            for c_name, coeff in v_info["resource_costs"].items():
                if abs(coeff) > 1e-9:
                    constr_lhss[c_name].append(f"{coeff} * vars['{v_name}']")
                    
        sorted_constrs = sorted(data["constraints"].keys())
        for c_name in sorted_constrs:
            c_info = data["constraints"][c_name]
            lhs_parts = constr_lhss[c_name]
            if not lhs_parts:
                lhs_str = "0"
            else:
                lhs_str = " + ".join(lhs_parts)
                
            rhs = c_info["rhs"]
            sense = c_info["sense"]
            if sense == "=":
                sense = "=="
            lines.append(f"    m.addConstr({lhs_str} {sense} {rhs}, name='{c_name}')")

        lines.append("")
        lines.append("    # 4. Optimize")
        lines.append("    m.optimize()")
        lines.append("    return m")
        lines.append("```")

        return "\n".join(lines)

    def save_to_file(self, filepath: str):
        data = self.to_json_dict()
        # Attach gold solution code; fail fast if code generation fails
        data["gold_solution"] = self.generate_gurobi_code_reference(data)
        save_json(filepath, data)

    @classmethod
    def get_system_prompt(cls) -> str:
        """Return the system prompt for LLM description generation."""
        return cls.SYSTEM_PROMPT

    def prepare_prompt_data(self) -> dict:
        """
        Prepare sanitized problem data for LLM prompt generation.
        Removes answer-bearing fields (gurobi_result, gold_solution, etc.).
        """
        data = self.to_json_dict()
        return self._sanitize_for_prompt(data)

    @staticmethod
    def _sanitize_for_prompt(data: dict) -> dict:
        """Remove answer-bearing fields before sending to LLM."""
        sanitized = copy.deepcopy(data) if isinstance(data, dict) else {}
        
        # Top-level fields to drop
        for key in ["gurobi_result", "result", "gold_solution"]:
            sanitized.pop(key, None)
        
        return sanitized

    @classmethod
    def build_prompt_messages(cls, problem_data: dict) -> list[dict]:
        """
        Build the complete messages list for LLM prompt generation.
        
        Subclasses can override this to fully customize the prompt structure.
        
        Args:
            problem_data: The problem data dictionary (will be sanitized).
            
        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        import json
        
        clean_data = cls._sanitize_for_prompt(problem_data)
        system_prompt = cls.get_system_prompt()
        
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Here is the problem data:\n{json.dumps(clean_data, indent=2)}\n\nWrite the client memo.",
            },
        ]

    @classmethod
    def finish_description(cls, llm_description: str, problem_data: dict) -> str:
        """
        Post-process the LLM-generated description to fill in any placeholders or finalize the format.
        
        This method is called after receiving the LLM description and can be used to:
        - Fill in template placeholders with actual data
        - Format numerical data
        - Add any final touches to the description
        
        Subclasses should override this method if they need custom post-processing.
        
        Args:
            llm_description: The raw description returned by the LLM.
            problem_data: The full problem data dictionary (including instance_data).
            
        Returns:
            The finalized description ready to be stored.
        """
        # Default implementation: return description as-is
        return llm_description

