import numpy as np
from main.generation.base_problem import BaseProblem


class JSSPGenerator(BaseProblem):
    """
    Job Shop Scheduling Problem (JSSP) - MILP.

    N jobs must be processed on M machines. Each job has M operations
    that must be performed in a fixed order. Each operation runs on a
    specific machine for a given duration. No two operations can run
    simultaneously on the same machine.

    Decision Variables:
    - Start_j_o:  Continuous — start time of operation o of job j
    - Cmax:       Continuous — makespan (completion time of last operation)
    - Y_j1_o1_j2_o2: Binary — 1 if operation (j1,o1) precedes (j2,o2) on shared machine

    Constraints:
    1. Precedence:    S[j,o+1] >= S[j,o] + p[j,o]   (operation order within jobs)
    2. Disjunctive-A: S[j1,o1] >= S[j2,o2] + p[j2,o2] - M*(1-Y)
    3. Disjunctive-B: S[j2,o2] >= S[j1,o1] + p[j1,o1] - M*Y
    4. Makespan:      Cmax >= S[j,last_op] + p[j,last_op]

    Objective: Minimize Cmax
    """

    SYSTEM_PROMPT = """You are a Manufacturing Operations Manager writing a memo about a production scheduling challenge.

**TASK:** Write a natural, business-like problem description for a job shop scheduling problem.

**TONE:** Professional operations memo. The reader should understand the scheduling challenge intuitively.

**REQUIRED PLACEHOLDERS (include ALL exactly once):**
{NUM_JOBS}, {NUM_MACHINES}, {PROCESSING_TIMES}, {MACHINE_ASSIGNMENTS}

**CRITICAL - DATA HANDLING:**
- Use ONLY the placeholders above for data. Do NOT write out sample values.
- Each placeholder should appear EXACTLY ONCE.
- Place all data placeholders in a dedicated "ANNEX" or "DATA" section at the end.

**WHAT TO INCLUDE:**
- A manufacturing facility with multiple specialized machines (workstations)
- Multiple jobs (products/orders) need to be manufactured
- Each job requires a series of operations in a fixed sequence
- Each operation must be performed on a specific machine for a known duration
- A machine can only handle one operation at a time (no preemption)
- Operations within the same job must be completed in order
- Once an operation starts, it runs to completion without interruption
- Goal: Schedule all operations to minimize the time to complete all jobs (makespan)

**WHAT TO AVOID:**
- Mathematical formulas or technical jargon (no "Big-M", "disjunctive constraints", "MILP")
- Do NOT use OR terminology like "objective function" or "decision variables"
- Do NOT list sample data - only use placeholders

**SUGGESTED STRUCTURE:**
1. Subject/Context: Overview of the manufacturing operation
2. Challenge: The scheduling problem
3. Requirements: Machine exclusivity, operation ordering
4. Goal: Minimize total production time
5. Annex - Production Data: All placeholders"""

    def __init__(
        self,
        n_jobs: int = 4,
        n_machines: int = 3,
        processing_time_range: tuple = (5, 30),
        prob_type: str = "LP",
        goal: str = "MINIMIZE",
    ):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.processing_time_range = processing_time_range

        # Each job has n_machines operations (one per machine, in some order)
        self.n_ops_per_job = n_machines

        # Variable counts
        # Start times: n_jobs * n_machines
        self.num_starts = n_jobs * n_machines
        # Makespan: 1
        self.num_cmax = 1
        # Disjunctive binary variables: for each machine, C(jobs_on_machine, 2) pairs
        # Each machine has n_jobs operations → n_jobs*(n_jobs-1)/2 pairs per machine
        self.pairs_per_machine = n_jobs * (n_jobs - 1) // 2
        self.num_binary = n_machines * self.pairs_per_machine

        n_vars = self.num_starts + self.num_cmax + self.num_binary

        # Constraint counts
        # Precedence: n_jobs * (n_machines - 1)
        self.n_precedence = n_jobs * (n_machines - 1)
        # Disjunctive (2 per pair): 2 * n_machines * pairs_per_machine
        self.n_disjunctive = 2 * self.num_binary
        # Makespan: n_jobs
        self.n_makespan = n_jobs

        n_constrs = self.n_precedence + self.n_disjunctive + self.n_makespan

        super().__init__(
            n_vars, n_constrs,
            prob_type=prob_type, goal=goal,
            problem_type="job shop scheduling",
        )

        # Instance data (populated by _generate_instance_data)
        self.processing_times = None   # [job][operation] -> duration
        self.machine_assignments = None  # [job][operation] -> machine index
        self.big_M = None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, time_limit: float = 300.0):
        self._generate_instance_data()
        self._build_matrices()
        self.solve(time_limit=time_limit)
        return self

    def _generate_instance_data(self):
        n_j = self.n_jobs
        n_m = self.n_machines
        lo, hi = self.processing_time_range

        # Processing times: random integers
        self.processing_times = [
            [int(np.random.randint(lo, hi + 1)) for _ in range(n_m)]
            for _ in range(n_j)
        ]

        # Machine assignments: each job visits all machines in a random permutation
        self.machine_assignments = [
            [int(x) for x in np.random.permutation(n_m)]
            for _ in range(n_j)
        ]

        # Big-M: sum of all processing times (upper bound on makespan)
        self.big_M = sum(
            self.processing_times[j][o]
            for j in range(n_j)
            for o in range(n_m)
        )

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _get_start_idx(self, job, op):
        """Index for Start_{job}_{op}."""
        return job * self.n_machines + op

    def _get_cmax_idx(self):
        """Index for Cmax variable."""
        return self.num_starts

    def _get_binary_idx(self, machine, pair_idx):
        """Index for disjunctive binary variable on given machine and pair."""
        return self.num_starts + self.num_cmax + machine * self.pairs_per_machine + pair_idx

    # ------------------------------------------------------------------
    # Build A, b, c
    # ------------------------------------------------------------------

    def _build_matrices(self):
        n_j = self.n_jobs
        n_m = self.n_machines
        M = self.big_M

        # ---- Variable names ----
        self.var_names = []
        for j in range(n_j):
            for o in range(n_m):
                self.var_names.append(f"Start_{j}_{o}")
        self.var_names.append("Cmax")

        # Build machine → [(job, op)] mapping for disjunctive pairs
        machine_ops = {m: [] for m in range(n_m)}
        for j in range(n_j):
            for o in range(n_m):
                m_id = self.machine_assignments[j][o]
                machine_ops[m_id].append((j, o))

        # Binary variable names for disjunctive pairs
        self._disjunctive_pairs = {}  # machine -> list of ((j1,o1),(j2,o2))
        for m_id in range(n_m):
            ops = machine_ops[m_id]
            pairs = []
            for idx1 in range(len(ops)):
                for idx2 in range(idx1 + 1, len(ops)):
                    j1, o1 = ops[idx1]
                    j2, o2 = ops[idx2]
                    pairs.append(((j1, o1), (j2, o2)))
                    self.var_names.append(f"Y_{j1}_{o1}_{j2}_{o2}")
            self._disjunctive_pairs[m_id] = pairs

        # ---- Variable types ----
        self.var_types = (
            ["Continuous"] * self.num_starts +
            ["Continuous"] * self.num_cmax +
            ["Binary"] * self.num_binary
        )

        # ---- Variable bounds ----
        self.var_bounds = []
        # Start times: [0, big_M]
        for _ in range(self.num_starts):
            self.var_bounds.append((0.0, float(M)))
        # Cmax: [0, big_M]
        self.var_bounds.append((0.0, float(M)))
        # Binary: [0, 1]
        for _ in range(self.num_binary):
            self.var_bounds.append((0.0, 1.0))

        # ---- Objective: minimize Cmax ----
        self.c = np.zeros(self.n)
        self.c[self._get_cmax_idx()] = 1.0

        # ---- Constraints ----
        self.A = np.zeros((self.m, self.n))
        self.b_vec = np.zeros(self.m)
        self.senses = []
        self.constr_names = []
        row = 0

        # C1: Precedence — S[j,o+1] >= S[j,o] + p[j,o]
        #     ⟹ S[j,o] - S[j,o+1] <= -p[j,o]
        for j in range(n_j):
            for o in range(n_m - 1):
                self.A[row, self._get_start_idx(j, o)] = 1.0
                self.A[row, self._get_start_idx(j, o + 1)] = -1.0
                self.b_vec[row] = -float(self.processing_times[j][o])
                self.senses.append("<=")
                self.constr_names.append(f"Prec_{j}_{o}")
                row += 1

        # C2 & C3: Disjunctive constraints for each machine
        # For each pair ((j1,o1),(j2,o2)) on the same machine:
        #   Y=1 means j1 before j2: S[j1,o1] + p1 <= S[j2,o2] + M*(1-Y)
        #     ⟹ S[j1,o1] - S[j2,o2] + M*Y <= M - p[j1,o1]
        #   Y=0 means j2 before j1: S[j2,o2] + p2 <= S[j1,o1] + M*Y
        #     ⟹ S[j2,o2] - S[j1,o1] - M*Y <= -p[j2,o2]
        for m_id in range(n_m):
            for pair_idx, ((j1, o1), (j2, o2)) in enumerate(self._disjunctive_pairs[m_id]):
                bin_idx = self._get_binary_idx(m_id, pair_idx)
                p1 = self.processing_times[j1][o1]
                p2 = self.processing_times[j2][o2]

                # Disjunctive A: S[j1,o1] - S[j2,o2] + M*Y <= M - p[j1,o1]
                self.A[row, self._get_start_idx(j1, o1)] = 1.0
                self.A[row, self._get_start_idx(j2, o2)] = -1.0
                self.A[row, bin_idx] = float(M)
                self.b_vec[row] = float(M - p1)
                self.senses.append("<=")
                self.constr_names.append(f"Disj_A_{j1}_{o1}_{j2}_{o2}")
                row += 1

                # Disjunctive B: S[j2,o2] - S[j1,o1] - M*Y <= -p[j2,o2]
                self.A[row, self._get_start_idx(j2, o2)] = 1.0
                self.A[row, self._get_start_idx(j1, o1)] = -1.0
                self.A[row, bin_idx] = -float(M)
                self.b_vec[row] = -float(p2)
                self.senses.append("<=")
                self.constr_names.append(f"Disj_B_{j1}_{o1}_{j2}_{o2}")
                row += 1

        # C4: Makespan — Cmax >= S[j, last_op] + p[j, last_op]
        #     ⟹ S[j, last_op] - Cmax <= -p[j, last_op]
        for j in range(n_j):
            last_op = n_m - 1
            self.A[row, self._get_start_idx(j, last_op)] = 1.0
            self.A[row, self._get_cmax_idx()] = -1.0
            self.b_vec[row] = -float(self.processing_times[j][last_op])
            self.senses.append("<=")
            self.constr_names.append(f"Makespan_J{j}")
            row += 1

        assert row == self.m, f"Expected {self.m} constraints, built {row}"
        self.b = self.b_vec

    # ------------------------------------------------------------------
    # Instance data for JSON
    # ------------------------------------------------------------------

    def _get_instance_data(self):
        return {
            "n_jobs": self.n_jobs,
            "n_machines": self.n_machines,
            "processing_times": self.processing_times,
            "machine_assignments": self.machine_assignments,
        }

    # ------------------------------------------------------------------
    # Gold solution code
    # ------------------------------------------------------------------

    def generate_gurobi_code_reference(self, data: dict) -> str:
        inst = data.get("instance_data", {})
        meta = data.get("meta", {})

        n_jobs = inst.get("n_jobs", 0)
        n_machines = inst.get("n_machines", 0)
        proc_times = inst.get("processing_times", [])
        machine_asgn = inst.get("machine_assignments", [])

        sense = "GRB.MINIMIZE" if meta.get("goal") == "MINIMIZE" else "GRB.MAXIMIZE"

        lines = []
        lines.append("```python")
        lines.append("from gurobipy import *")
        lines.append("")
        lines.append("def solve_problem():")
        lines.append('    m = Model("JSSP")')
        lines.append("    m.setParam('TimeLimit', 300)")
        lines.append("")
        lines.append("    # --- Data ---")
        lines.append(f"    n_jobs = {n_jobs}")
        lines.append(f"    n_machines = {n_machines}")
        lines.append(f"    processing_times = {proc_times}")
        lines.append(f"    machine_assignments = {machine_asgn}")
        lines.append("")
        lines.append("    # Big-M: sum of all processing times")
        lines.append("    M = sum(processing_times[j][o] for j in range(n_jobs) for o in range(n_machines))")
        lines.append("")
        lines.append("    # --- Variables ---")
        lines.append("    # Start times for each operation")
        lines.append("    S = {}")
        lines.append("    for j in range(n_jobs):")
        lines.append("        for o in range(n_machines):")
        lines.append('            S[(j,o)] = m.addVar(lb=0, ub=M, vtype=GRB.CONTINUOUS, name=f"Start_{j}_{o}")')
        lines.append("")
        lines.append("    # Makespan variable")
        lines.append('    Cmax = m.addVar(lb=0, ub=M, vtype=GRB.CONTINUOUS, name="Cmax")')
        lines.append("")
        lines.append("    # Build machine -> [(job, op)] mapping")
        lines.append("    machine_ops = {mach: [] for mach in range(n_machines)}")
        lines.append("    for j in range(n_jobs):")
        lines.append("        for o in range(n_machines):")
        lines.append("            machine_ops[machine_assignments[j][o]].append((j, o))")
        lines.append("")
        lines.append("    # Disjunctive binary variables for each pair on same machine")
        lines.append("    Y = {}")
        lines.append("    for mach in range(n_machines):")
        lines.append("        ops = machine_ops[mach]")
        lines.append("        for i in range(len(ops)):")
        lines.append("            for k in range(i+1, len(ops)):")
        lines.append("                j1, o1 = ops[i]")
        lines.append("                j2, o2 = ops[k]")
        lines.append('                Y[(j1,o1,j2,o2)] = m.addVar(vtype=GRB.BINARY, name=f"Y_{j1}_{o1}_{j2}_{o2}")')
        lines.append("")
        lines.append("    # --- Objective: minimize makespan ---")
        lines.append(f"    m.setObjective(Cmax, {sense})")
        lines.append("")
        lines.append("    # --- Constraints ---")
        lines.append("    # 1. Precedence: operations within a job must follow order")
        lines.append("    for j in range(n_jobs):")
        lines.append("        for o in range(n_machines - 1):")
        lines.append("            m.addConstr(")
        lines.append("                S[(j,o+1)] >= S[(j,o)] + processing_times[j][o],")
        lines.append('                name=f"Prec_{j}_{o}")')
        lines.append("")
        lines.append("    # 2. Disjunctive: no two operations on the same machine overlap")
        lines.append("    for mach in range(n_machines):")
        lines.append("        ops = machine_ops[mach]")
        lines.append("        for i in range(len(ops)):")
        lines.append("            for k in range(i+1, len(ops)):")
        lines.append("                j1, o1 = ops[i]")
        lines.append("                j2, o2 = ops[k]")
        lines.append("                p1 = processing_times[j1][o1]")
        lines.append("                p2 = processing_times[j2][o2]")
        lines.append("                y = Y[(j1,o1,j2,o2)]")
        lines.append("                m.addConstr(")
        lines.append("                    S[(j1,o1)] + p1 <= S[(j2,o2)] + M * (1 - y),")
        lines.append('                    name=f"Disj_A_{j1}_{o1}_{j2}_{o2}")')
        lines.append("                m.addConstr(")
        lines.append("                    S[(j2,o2)] + p2 <= S[(j1,o1)] + M * y,")
        lines.append('                    name=f"Disj_B_{j1}_{o1}_{j2}_{o2}")')
        lines.append("")
        lines.append("    # 3. Makespan: Cmax >= completion of last operation of each job")
        lines.append("    for j in range(n_jobs):")
        lines.append("        last_op = n_machines - 1")
        lines.append("        m.addConstr(")
        lines.append("            Cmax >= S[(j, last_op)] + processing_times[j][last_op],")
        lines.append('            name=f"Makespan_J{j}")')
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

        n_jobs = inst.get("n_jobs", 0)
        n_machines = inst.get("n_machines", 0)
        proc_times = inst.get("processing_times", [])

        # Compute average processing time for context
        all_times = [t for row in proc_times for t in row]
        avg_time = sum(all_times) / len(all_times) if all_times else 0

        user_content = f"""Write a professional operations memo for a job shop scheduling problem.

**Scenario:**
A manufacturing facility has {n_machines} specialized machines (workstations).
{n_jobs} production jobs need to be scheduled, each requiring {n_machines} operations.

**Problem Scale:**
- {n_jobs} jobs × {n_machines} operations each = {n_jobs * n_machines} total operations
- {n_machines} machines, each handling one operation at a time
- Processing times range from {min(all_times)} to {max(all_times)} time units (avg {avg_time:.0f})

**Key Rules:**
- Each job has a fixed sequence of operations (must be done in order)
- Each operation is assigned to a specific machine
- A machine can only work on one operation at a time
- No preemption (once started, an operation runs to completion)
- Goal: Complete ALL jobs in the shortest possible total time (minimize makespan)

**Your Task:**
Write a business memo describing this scheduling challenge.
IMPORTANT: Use ONLY these placeholders for data:
{{NUM_JOBS}}, {{NUM_MACHINES}}, {{PROCESSING_TIMES}}, {{MACHINE_ASSIGNMENTS}}

Each placeholder should appear exactly ONCE in an Annex section.
Do NOT duplicate data by writing sample values AND using placeholders.
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
            "NUM_JOBS", "NUM_MACHINES", "PROCESSING_TIMES", "MACHINE_ASSIGNMENTS",
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

        fmt["NUM_JOBS"] = str(inst.get("n_jobs", ""))
        fmt["NUM_MACHINES"] = str(inst.get("n_machines", ""))

        # Processing times table
        proc_times = inst.get("processing_times", [])
        machine_asgn = inst.get("machine_assignments", [])
        n_machines = inst.get("n_machines", 0)

        pt_lines = [f"Processing Times (time units per operation):"]
        pt_lines.append(f"  Each job has {n_machines} operations performed in sequence.")
        for j, times in enumerate(proc_times):
            ops_str = ", ".join(
                f"Op{o+1}: {t}" for o, t in enumerate(times)
            )
            pt_lines.append(f"  - Job {j+1}: [{ops_str}]")
        fmt["PROCESSING_TIMES"] = "\n".join(pt_lines)

        # Machine assignments table
        ma_lines = ["Machine Assignments (which machine each operation uses):"]
        ma_lines.append(f"  Machine indices: 0 to {n_machines - 1}")
        for j, machines in enumerate(machine_asgn):
            ops_str = ", ".join(
                f"Op{o+1}: Machine {m}" for o, m in enumerate(machines)
            )
            ma_lines.append(f"  - Job {j+1}: [{ops_str}]")
        fmt["MACHINE_ASSIGNMENTS"] = "\n".join(ma_lines)

        result = llm_description
        for ph, val in fmt.items():
            result = result.replace(f"{{{ph}}}", val)

        remaining = [p for p in required if f"{{{p}}}" in result]
        if remaining:
            print(f"Placeholders not filled: {remaining}")
            return None

        return result
