#!/usr/bin/env python3
"""
Deterministic Phase 2 template solvers for template categories.

Given a path to an extracted instance_data JSON file (Phase 1 output),
generate complete Gurobi Python code that solves the problem.

These are the template-code counterpart to the binding specialist:
  Phase 1 (binder) → instance_data JSON → Phase 2 (this module) → Gurobi code
"""

import textwrap


def build_transportation_code(extracted_json_path: str) -> str:
    """Generate Gurobi code that reads transportation instance_data from a JSON file."""
    return textwrap.dedent(f'''\
        import json
        from gurobipy import *

        def solve_problem():
            with open(r"{extracted_json_path}", "r") as f:
                d = json.load(f)

            num_sources = d["num_sources"]
            num_dests = d["num_destinations"]
            supplies = d["supplies"]
            demands = d["demands"]
            costs = d["cost_matrix"]
            is_integer = d.get("integer_flows", False)

            m = Model("Transportation_Problem")
            m.setParam("TimeLimit", 300)

            sources = list(range(num_sources))
            destinations = list(range(num_dests))
            vtype = GRB.INTEGER if is_integer else GRB.CONTINUOUS

            # Decision variables: flow from source i to destination j
            flows = {{}}
            for i in sources:
                for j in destinations:
                    flows[(i, j)] = m.addVar(lb=0.0, vtype=vtype, name=f"Flow_S{{i}}_D{{j}}")

            # Objective: minimize total transportation cost
            m.setObjective(
                quicksum(costs[i][j] * flows[(i, j)] for i in sources for j in destinations),
                GRB.MINIMIZE
            )

            # Supply constraints
            for i in sources:
                m.addConstr(
                    quicksum(flows[(i, j)] for j in destinations) <= supplies[i],
                    name=f"Supply_S{{i}}"
                )

            # Demand constraints
            for j in destinations:
                m.addConstr(
                    quicksum(flows[(i, j)] for i in sources) >= demands[j],
                    name=f"Demand_D{{j}}"
                )

            m.optimize()
            return m
    ''')


def build_jssp_code(extracted_json_path: str) -> str:
    """Generate Gurobi code that reads JSSP instance_data from a JSON file."""
    return textwrap.dedent(f'''\
        import json
        from gurobipy import *

        def solve_problem():
            with open(r"{extracted_json_path}", "r") as f:
                d = json.load(f)

            n_jobs = d["n_jobs"]
            n_machines = d["n_machines"]
            processing_times = d["processing_times"]
            machine_assignments = d["machine_assignments"]

            # Big-M: sum of all processing times
            M = sum(processing_times[j][o] for j in range(n_jobs) for o in range(n_machines))

            m = Model("JSSP")
            m.setParam("TimeLimit", 300)

            # Start time variables
            S = {{}}
            for j in range(n_jobs):
                for o in range(n_machines):
                    S[(j, o)] = m.addVar(lb=0, ub=M, vtype=GRB.CONTINUOUS, name=f"Start_{{j}}_{{o}}")

            # Makespan variable
            Cmax = m.addVar(lb=0, ub=M, vtype=GRB.CONTINUOUS, name="Cmax")

            # Build machine -> [(job, op)] mapping
            machine_ops = {{mach: [] for mach in range(n_machines)}}
            for j in range(n_jobs):
                for o in range(n_machines):
                    machine_ops[machine_assignments[j][o]].append((j, o))

            # Disjunctive binary variables for each pair on same machine
            Y = {{}}
            for mach in range(n_machines):
                ops = machine_ops[mach]
                for i in range(len(ops)):
                    for k in range(i + 1, len(ops)):
                        j1, o1 = ops[i]
                        j2, o2 = ops[k]
                        Y[(j1, o1, j2, o2)] = m.addVar(vtype=GRB.BINARY, name=f"Y_{{j1}}_{{o1}}_{{j2}}_{{o2}}")

            # Objective: minimize makespan
            m.setObjective(Cmax, GRB.MINIMIZE)

            # 1. Precedence constraints
            for j in range(n_jobs):
                for o in range(n_machines - 1):
                    m.addConstr(
                        S[(j, o + 1)] >= S[(j, o)] + processing_times[j][o],
                        name=f"Prec_{{j}}_{{o}}"
                    )

            # 2. Disjunctive constraints
            for mach in range(n_machines):
                ops = machine_ops[mach]
                for i in range(len(ops)):
                    for k in range(i + 1, len(ops)):
                        j1, o1 = ops[i]
                        j2, o2 = ops[k]
                        p1 = processing_times[j1][o1]
                        p2 = processing_times[j2][o2]
                        y = Y[(j1, o1, j2, o2)]
                        m.addConstr(
                            S[(j1, o1)] + p1 <= S[(j2, o2)] + M * (1 - y),
                            name=f"Disj_A_{{j1}}_{{o1}}_{{j2}}_{{o2}}"
                        )
                        m.addConstr(
                            S[(j2, o2)] + p2 <= S[(j1, o1)] + M * y,
                            name=f"Disj_B_{{j1}}_{{o1}}_{{j2}}_{{o2}}"
                        )

            # 3. Makespan constraints
            for j in range(n_jobs):
                last_op = n_machines - 1
                m.addConstr(
                    Cmax >= S[(j, last_op)] + processing_times[j][last_op],
                    name=f"Makespan_J{{j}}"
                )

            m.optimize()
            return m
    ''')


TEMPLATE_BUILDERS = {
    "transportation": build_transportation_code,
    "jssp": build_jssp_code,
}


def build_template_code(extracted_json_path: str, category: str) -> str:
    """Dispatch to the right template builder for the given category."""
    if category not in TEMPLATE_BUILDERS:
        raise ValueError(f"Unknown category: {category}. Available: {list(TEMPLATE_BUILDERS.keys())}")
    return TEMPLATE_BUILDERS[category](extracted_json_path)


if __name__ == "__main__":
    # Quick test: print generated code for each category
    for cat in TEMPLATE_BUILDERS:
        print(f"=== {cat} ===")
        code = build_template_code("/tmp/test_extracted.json", cat)
        print(code[:500])
        print("...")
        print()
