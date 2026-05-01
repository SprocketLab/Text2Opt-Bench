import numpy as np
from main.generation.base_problem import BaseProblem


class RCPSPGenerator(BaseProblem):
    """
    Multi-Mode Resource-Constrained Project Scheduling Problem (MRCPSP) - MILP.

    N activities must be scheduled subject to precedence constraints (a DAG),
    renewable resource capacity constraints, a non-renewable budget constraint,
    time lags (min/max delays between dependent activities), and per-activity
    release dates and deadlines.  Each activity can be executed in one of
    several modes, each with different duration, resource requirements, and cost.

    Formulation (continuous-time disjunctive):

    Decision Variables:
    - S[i]:      Continuous — start time of activity i
    - x[i,m]:    Binary — 1 if activity i uses execution mode m
    - Cmax:      Continuous — makespan
    - Y[i,j]:    Binary — 1 if activity i is scheduled before activity j
                 (only for "conflicting" pairs that are not precedence-related
                  and whose combined resource usage in SOME mode combination
                  exceeds some resource capacity)

    Constraints:
    1. Mode selection:    sum_m x[i,m] = 1  for each activity i
    2. Precedence + min lag: S[j] >= S[i] + d_eff[i] + lag_min
       where d_eff[i] = sum_m dur[i][m]*x[i,m]
    3. Max lag (some edges): S[j] <= S[i] + d_eff[i] + lag_max
    4. Disjunctive-A:    S[i] + d_eff[i] <= S[j] + M*(1-Y[i,j])
    5. Disjunctive-B:    S[j] + d_eff[j] <= S[i] + M*Y[i,j]
    6. Budget:           sum_i sum_m cost[i][m]*x[i,m] <= budget
    7. Deadlines:        S[i] + d_eff[i] <= deadline[i]  (for activities with deadlines)
    8. Makespan:         Cmax >= S[i] + d_eff[i]  for all activities

    Objective: Minimize Cmax

    What makes this hard for LLMs:
    - Must create mode-selection binary variables and link durations to modes
    - Must identify "conflicting pairs" across ALL mode combinations (4-level nested loop)
    - Must compute transitive closure to exclude precedence-related pairs
    - Must handle two-sided time lags (min AND max) on precedence edges
    - Must incorporate non-renewable budget constraint on mode costs
    - Must enforce deadlines with mode-dependent effective durations
    - Big-M formulation with correct value
    """

    SYSTEM_PROMPT = """You are a Project Manager writing a memo about a project scheduling challenge.

**TASK:** Write a natural, business-like problem description for a multi-mode resource-constrained project scheduling problem.

**TONE:** Professional project management memo. The reader should understand the scheduling challenge intuitively.

**REQUIRED PLACEHOLDERS (include ALL exactly once):**
{NUM_ACTIVITIES}, {NUM_RESOURCES}, {NUM_MODES}, {RESOURCE_CAPACITIES},
{MODE_DURATIONS}, {MODE_RESOURCE_REQUIREMENTS}, {MODE_COSTS}, {BUDGET},
{PRECEDENCE_EDGES_WITH_LAGS}, {RELEASE_DATES}, {DEADLINES}

**CRITICAL - DATA HANDLING:**
- Use ONLY the placeholders above for data. Do NOT write out sample values.
- Each placeholder should appear EXACTLY ONCE.
- Place all data placeholders in a dedicated "ANNEX" or "DATA" section at the end.

**WHAT TO INCLUDE:**
- A project with multiple activities that must be scheduled
- Each activity can be executed in one of several modes — each mode has a different duration, resource requirement, and cost (faster modes cost more)
- Resources are renewable: once an activity finishes, its resources become available again
- There is also a fixed project budget (non-renewable): the sum of all chosen mode costs must not exceed it
- Precedence constraints define which activities must finish before others can start
- Some dependencies have minimum delays (activity B must wait at least X time units after activity A finishes)
- Some dependencies also have maximum delays (activity B must start within Y time units after activity A finishes)
- Each activity may have an earliest allowed start time (release date)
- Some activities have a hard deadline by which they must be completed
- Multiple resource types, each with a fixed capacity
- Activities cannot be preempted (once started, they run to completion)
- RESOURCE CONFLICT RULE: Two activities that are NOT related by precedence must be sequenced (one finishes before the other starts) if there EXISTS at least one combination of their execution modes AND at least one resource where their combined usage would exceed that resource's capacity. This is a conservative safety rule — even if some mode pairs would fit simultaneously, the potential for overload in any mode combination means the activities cannot overlap. The scheduler must decide which one goes first. This is the ONLY resource constraint — there is no separate "at all times" capacity check beyond these pairwise sequencing decisions.
- Goal: Schedule all activities (choosing a mode for each) to minimize the total project duration (makespan) while staying within budget

**WHAT TO AVOID:**
- Mathematical formulas or technical jargon (no "Big-M", "disjunctive constraints", "MILP")
- Do NOT use OR terminology like "objective function" or "decision variables"
- Do NOT list sample data - only use placeholders

**SUGGESTED STRUCTURE:**
1. Subject/Context: Overview of the project
2. Execution modes: Each activity has multiple execution options with cost/speed tradeoffs
3. Challenge: Dependencies with timing constraints, resource limits, and budget
4. Resource conflicts: When two independent activities could compete for resources in any execution mode, one must be delayed
5. Timing constraints: Release dates, deadlines, minimum/maximum delays between dependent tasks
6. Budget: Total spending on execution modes is capped
7. Goal: Minimize total project duration while staying within budget
8. Annex - Project Data: All placeholders"""

    def __init__(
        self,
        n_activities: int = 8,
        n_resources: int = 2,
        n_modes: int = 3,
        capacity_range: tuple = (3, 5),
        duration_range: tuple = (2, 10),
        resource_range: tuple = (0, 3),
        cost_range: tuple = (5, 25),
        precedence_density: float = 1.3,
        lag_min_prob: float = 0.3,
        lag_max_prob: float = 0.25,
        release_prob: float = 0.3,
        deadline_prob: float = 0.2,
        prob_type: str = "LP",
        goal: str = "MINIMIZE",
    ):
        self.n_activities = n_activities
        self.n_resources = n_resources
        self.n_modes = n_modes
        self.capacity_range = capacity_range
        self.duration_range = duration_range
        self.resource_range = resource_range
        self.cost_range = cost_range
        self.precedence_density = precedence_density
        self.lag_min_prob = lag_min_prob
        self.lag_max_prob = lag_max_prob
        self.release_prob = release_prob
        self.deadline_prob = deadline_prob

        # Filled during generation
        self.resource_capacities = None
        self.mode_durations = None          # N x M
        self.mode_resource_requirements = None  # N x M x K
        self.mode_costs = None              # N x M
        self.budget = None
        self.precedence_edges = None        # list of (i, j)
        self.time_lags = None               # list of (i, j, lag_min, lag_max_or_None)
        self.release_dates = None           # list of N ints
        self.deadlines = None               # list of N (int or None)
        self.conflicting_pairs = None
        self.big_M = None

        super().__init__(
            n_vars=1, n_constrs=1,
            prob_type=prob_type, goal=goal,
            problem_type="resource-constrained project scheduling",
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, time_limit: float = 300.0):
        self._generate_instance_data()
        self._compute_conflicting_pairs()
        self._update_counts()
        self._build_matrices()
        self.solve(time_limit=time_limit)
        return self

    def _generate_instance_data(self):
        N = self.n_activities
        K = self.n_resources
        M = self.n_modes

        # Resource capacities (renewable)
        lo_c, hi_c = self.capacity_range
        self.resource_capacities = [int(np.random.randint(lo_c, hi_c + 1)) for _ in range(K)]

        # Mode data for each activity
        lo_d, hi_d = self.duration_range
        lo_r, hi_r = self.resource_range
        lo_cost, hi_cost = self.cost_range

        self.mode_durations = []
        self.mode_resource_requirements = []
        self.mode_costs = []

        for i in range(N):
            # Generate M modes with speed/cost tradeoff:
            # shorter duration → higher cost, higher resource usage
            durs = sorted([int(np.random.randint(lo_d, hi_d + 1)) for _ in range(M)])
            costs = sorted([int(np.random.randint(lo_cost, hi_cost + 1)) for _ in range(M)], reverse=True)

            reqs = []
            for m_idx in range(M):
                r = [int(np.random.randint(lo_r, hi_r + 1)) for _ in range(K)]
                if sum(r) == 0:
                    r[int(np.random.randint(0, K))] = 1
                # Cap at capacity
                r = [min(r[k], self.resource_capacities[k]) for k in range(K)]
                reqs.append(r)

            self.mode_durations.append(durs)
            self.mode_resource_requirements.append(reqs)
            self.mode_costs.append(costs)

        # Budget: between min-cost and max-cost mode sums, biased toward feasible
        min_total_cost = sum(min(self.mode_costs[i]) for i in range(N))
        max_total_cost = sum(max(self.mode_costs[i]) for i in range(N))
        # Set budget so cheapest modes are feasible but not all modes are free
        self.budget = int(min_total_cost + 0.4 * (max_total_cost - min_total_cost))

        # Generate precedence DAG
        self._generate_precedence_dag()

        # Add time lags to edges
        self.time_lags = []
        for (i, j) in self.precedence_edges:
            # Min lag: 0 most of the time, occasionally 1-3
            if np.random.random() < self.lag_min_prob:
                lag_min = int(np.random.randint(1, 4))
            else:
                lag_min = 0

            # Max lag: some edges have an upper bound on the delay
            if np.random.random() < self.lag_max_prob:
                max_dur = max(self.mode_durations[i])
                lag_max = lag_min + int(np.random.randint(max_dur, max_dur * 3 + 5))
            else:
                lag_max = None
            self.time_lags.append((i, j, lag_min, lag_max))

        # Release dates
        self.release_dates = [0] * N
        for i in range(N):
            if np.random.random() < self.release_prob:
                self.release_dates[i] = int(np.random.randint(1, 6))

        # Deadlines (generous to avoid infeasibility)
        max_total_dur = sum(max(self.mode_durations[i]) for i in range(N))
        self.deadlines = [None] * N
        for i in range(N):
            if np.random.random() < self.deadline_prob:
                min_dur = min(self.mode_durations[i])
                self.deadlines[i] = int(self.release_dates[i] + min_dur
                                        + np.random.randint(max_total_dur // 3, max_total_dur))

        # Big-M: upper bound on any start time
        self.big_M = max_total_dur + max(self.release_dates) + 10

    def _generate_precedence_dag(self):
        """Generate a random DAG by assigning activities to layers."""
        N = self.n_activities
        n_layers = max(2, N // 3)

        layers = [[] for _ in range(n_layers)]
        for i in range(N):
            layer = int(np.random.randint(0, n_layers))
            layers[layer].append(i)

        if not layers[0]:
            donor = next(l for l in range(1, n_layers) if layers[l])
            layers[0].append(layers[donor].pop())
        if not layers[-1]:
            donor = next(l for l in range(n_layers - 2, -1, -1) if layers[l])
            layers[-1].append(layers[donor].pop())

        target_edges = int(self.precedence_density * N)
        edges = set()

        for l in range(n_layers - 1):
            if not layers[l] or not layers[l + 1]:
                continue
            for i in layers[l]:
                for j in layers[l + 1]:
                    if np.random.random() < 0.4 and len(edges) < target_edges * 2:
                        edges.add((i, j))

        for l in range(n_layers):
            for l2 in range(l + 2, min(l + 3, n_layers)):
                if not layers[l] or not layers[l2]:
                    continue
                for i in layers[l]:
                    for j in layers[l2]:
                        if np.random.random() < 0.2 and len(edges) < target_edges * 2:
                            edges.add((i, j))

        edges = list(edges)
        if len(edges) > target_edges:
            np.random.shuffle(edges)
            edges = edges[:target_edges]

        all_activities = set(range(N))
        has_pred = {j for _, j in edges}

        layer_of = {}
        for l, acts in enumerate(layers):
            for a in acts:
                layer_of[a] = l

        for act in all_activities:
            if act not in has_pred and layer_of[act] > 0:
                my_layer = layer_of[act]
                candidates = [a for a in all_activities if layer_of[a] < my_layer]
                if candidates:
                    pred = candidates[int(np.random.randint(0, len(candidates)))]
                    edges.append((pred, act))

        self.precedence_edges = sorted(edges)

    def _compute_conflicting_pairs(self):
        """Find pairs of activities that COULD conflict on resources in ANY mode combination.

        Two activities conflict if:
        1. They are NOT related by precedence (neither is ancestor of the other)
        2. There EXISTS some resource k and mode combo (mi, mj) such that
           req[i][mi][k] + req[j][mj][k] > capacity[k]
        """
        N = self.n_activities
        K = self.n_resources
        M = self.n_modes

        # Build transitive closure of precedence
        reachable = set()
        adj = {i: [] for i in range(N)}
        for (i, j) in self.precedence_edges:
            adj[i].append(j)

        for start in range(N):
            visited = set()
            stack = [start]
            while stack:
                node = stack.pop()
                for nxt in adj[node]:
                    if nxt not in visited:
                        visited.add(nxt)
                        reachable.add((start, nxt))
                        stack.append(nxt)

        # Find conflicting pairs (check all mode combinations)
        self.conflicting_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                if (i, j) in reachable or (j, i) in reachable:
                    continue
                conflict = False
                for mi in range(M):
                    for mj in range(M):
                        for k in range(K):
                            if (self.mode_resource_requirements[i][mi][k]
                                    + self.mode_resource_requirements[j][mj][k]
                                    > self.resource_capacities[k]):
                                conflict = True
                                break
                        if conflict:
                            break
                    if conflict:
                        break
                if conflict:
                    self.conflicting_pairs.append((i, j))

    def _update_counts(self):
        """Update variable and constraint counts after generation."""
        N = self.n_activities
        M = self.n_modes
        n_conflicts = len(self.conflicting_pairs)
        n_edges = len(self.precedence_edges)
        n_maxlags = sum(1 for _, _, _, lmax in self.time_lags if lmax is not None)
        n_deadlines = sum(1 for d in self.deadlines if d is not None)

        # Variables: S[i] + Cmax + x[i,m] + Y[i,j]
        self.n = N + 1 + N * M + n_conflicts
        # Constraints: mode_sel(N) + prec_min(n_edges) + maxlag(n_maxlags)
        #   + disj(2*n_conflicts) + budget(1) + deadlines(n_deadlines) + makespan(N)
        self.m = N + n_edges + n_maxlags + 2 * n_conflicts + 1 + n_deadlines + N

    # ------------------------------------------------------------------
    # Variable column index helpers
    # ------------------------------------------------------------------

    def _col_S(self, i):
        """Column index for start time S[i]."""
        return i

    def _col_Cmax(self):
        """Column index for Cmax."""
        return self.n_activities

    def _col_x(self, i, m):
        """Column index for mode variable x[i,m]."""
        return self.n_activities + 1 + i * self.n_modes + m

    def _col_Y(self, idx):
        """Column index for ordering variable Y[conflict_idx]."""
        return self.n_activities + 1 + self.n_activities * self.n_modes + idx

    # ------------------------------------------------------------------
    # Build A, b, c
    # ------------------------------------------------------------------

    def _build_matrices(self):
        N = self.n_activities
        M_modes = self.n_modes
        BM = self.big_M
        n_conflicts = len(self.conflicting_pairs)

        # ---- Variable names ----
        self.var_names = [f"S_{i}" for i in range(N)]
        self.var_names.append("Cmax")
        for i in range(N):
            for m in range(M_modes):
                self.var_names.append(f"x_{i}_{m}")
        for idx, (i, j) in enumerate(self.conflicting_pairs):
            self.var_names.append(f"Y_{i}_{j}")

        # ---- Variable types ----
        self.var_types = (
            ["Continuous"] * N +           # start times
            ["Continuous"] +               # Cmax
            ["Binary"] * (N * M_modes) +   # mode selections
            ["Binary"] * n_conflicts       # ordering variables
        )

        # ---- Variable bounds ----
        self.var_bounds = []
        for i in range(N):
            self.var_bounds.append((float(self.release_dates[i]), float(BM)))
        self.var_bounds.append((0.0, float(BM)))  # Cmax
        for _ in range(N * M_modes):
            self.var_bounds.append((0.0, 1.0))
        for _ in range(n_conflicts):
            self.var_bounds.append((0.0, 1.0))

        # ---- Objective: minimize Cmax ----
        self.c = np.zeros(self.n)
        self.c[self._col_Cmax()] = 1.0

        # ---- Constraints ----
        self.A = np.zeros((self.m, self.n))
        self.b = np.zeros(self.m)
        self.senses = []
        self.constr_names = []
        row = 0

        # C1: Mode selection — sum_m x[i,m] = 1
        for i in range(N):
            for m in range(M_modes):
                self.A[row, self._col_x(i, m)] = 1.0
            self.b[row] = 1.0
            self.senses.append("=")
            self.constr_names.append(f"Mode_{i}")
            row += 1

        # C2: Precedence with min lag
        # S[j] >= S[i] + sum_m dur[i][m]*x[i,m] + lag_min
        # => S[i] - S[j] + sum_m dur[i][m]*x[i,m] <= -lag_min
        for edge_idx, (i, j, lag_min, lag_max) in enumerate(self.time_lags):
            self.A[row, self._col_S(i)] = 1.0
            self.A[row, self._col_S(j)] = -1.0
            for m in range(M_modes):
                self.A[row, self._col_x(i, m)] = float(self.mode_durations[i][m])
            self.b[row] = -float(lag_min)
            self.senses.append("<=")
            self.constr_names.append(f"Prec_{i}_{j}")
            row += 1

        # C3: Max lag constraints
        # S[j] <= S[i] + sum_m dur[i][m]*x[i,m] + lag_max
        # => -S[i] + S[j] - sum_m dur[i][m]*x[i,m] <= lag_max
        for edge_idx, (i, j, lag_min, lag_max) in enumerate(self.time_lags):
            if lag_max is None:
                continue
            self.A[row, self._col_S(i)] = -1.0
            self.A[row, self._col_S(j)] = 1.0
            for m in range(M_modes):
                self.A[row, self._col_x(i, m)] = -float(self.mode_durations[i][m])
            self.b[row] = float(lag_max)
            self.senses.append("<=")
            self.constr_names.append(f"MaxLag_{i}_{j}")
            row += 1

        # C4 & C5: Disjunctive constraints for conflicting pairs
        for idx, (i, j) in enumerate(self.conflicting_pairs):
            y_col = self._col_Y(idx)

            # Disj-A: S[i] + d_eff[i] <= S[j] + BM*(1-Y)
            # => S[i] - S[j] + sum_m dur[i][m]*x[i,m] + BM*Y <= BM
            self.A[row, self._col_S(i)] = 1.0
            self.A[row, self._col_S(j)] = -1.0
            for m in range(M_modes):
                self.A[row, self._col_x(i, m)] = float(self.mode_durations[i][m])
            self.A[row, y_col] = float(BM)
            self.b[row] = float(BM)
            self.senses.append("<=")
            self.constr_names.append(f"Disj_A_{i}_{j}")
            row += 1

            # Disj-B: S[j] + d_eff[j] <= S[i] + BM*Y
            # => S[j] - S[i] + sum_m dur[j][m]*x[j,m] - BM*Y <= 0
            self.A[row, self._col_S(j)] = 1.0
            self.A[row, self._col_S(i)] = -1.0
            for m in range(M_modes):
                self.A[row, self._col_x(j, m)] = float(self.mode_durations[j][m])
            self.A[row, y_col] = -float(BM)
            self.b[row] = 0.0
            self.senses.append("<=")
            self.constr_names.append(f"Disj_B_{i}_{j}")
            row += 1

        # C6: Budget — sum_i sum_m cost[i][m]*x[i,m] <= budget
        for i in range(N):
            for m in range(M_modes):
                self.A[row, self._col_x(i, m)] = float(self.mode_costs[i][m])
        self.b[row] = float(self.budget)
        self.senses.append("<=")
        self.constr_names.append("Budget")
        row += 1

        # C7: Deadlines — S[i] + d_eff[i] <= deadline[i]
        # => S[i] + sum_m dur[i][m]*x[i,m] <= deadline[i]
        for i in range(N):
            if self.deadlines[i] is None:
                continue
            self.A[row, self._col_S(i)] = 1.0
            for m in range(M_modes):
                self.A[row, self._col_x(i, m)] = float(self.mode_durations[i][m])
            self.b[row] = float(self.deadlines[i])
            self.senses.append("<=")
            self.constr_names.append(f"Deadline_{i}")
            row += 1

        # C8: Makespan — Cmax >= S[i] + d_eff[i]
        # => S[i] - Cmax + sum_m dur[i][m]*x[i,m] <= 0
        for i in range(N):
            self.A[row, self._col_S(i)] = 1.0
            self.A[row, self._col_Cmax()] = -1.0
            for m in range(M_modes):
                self.A[row, self._col_x(i, m)] = float(self.mode_durations[i][m])
            self.b[row] = 0.0
            self.senses.append("<=")
            self.constr_names.append(f"Makespan_{i}")
            row += 1

        assert row == self.m, f"Expected {self.m} constraints, built {row}"

    # ------------------------------------------------------------------
    # Instance data for JSON
    # ------------------------------------------------------------------

    def _get_instance_data(self):
        return {
            "n_activities": self.n_activities,
            "n_resources": self.n_resources,
            "n_modes": self.n_modes,
            "resource_capacities": self.resource_capacities,
            "mode_durations": self.mode_durations,
            "mode_resource_requirements": self.mode_resource_requirements,
            "mode_costs": self.mode_costs,
            "budget": self.budget,
            "precedence_with_lags": self.time_lags,  # list of (pred, succ, min_lag, max_lag_or_None)
            "release_dates": self.release_dates,
            "deadlines": self.deadlines,
        }

    # ------------------------------------------------------------------
    # Gold solution code
    # ------------------------------------------------------------------

    def generate_gurobi_code_reference(self, data: dict) -> str:
        inst = data.get("instance_data", {})
        meta = data.get("meta", {})
        sense = "GRB.MINIMIZE" if meta.get("goal") == "MINIMIZE" else "GRB.MAXIMIZE"

        # Serialize deadlines with None preserved
        deadlines_repr = "["
        dl_parts = []
        for d in inst["deadlines"]:
            dl_parts.append("None" if d is None else str(d))
        deadlines_repr += ", ".join(dl_parts) + "]"

        # Serialize time_lags with None preserved
        lags_repr = "["
        lag_parts = []
        for (i, j, lmin, lmax) in inst["precedence_with_lags"]:
            lmax_s = "None" if lmax is None else str(lmax)
            lag_parts.append(f"({i}, {j}, {lmin}, {lmax_s})")
        lags_repr += ", ".join(lag_parts) + "]"

        lines = []
        lines.append("```python")
        lines.append("from gurobipy import *")
        lines.append("")
        lines.append("def solve_problem():")
        lines.append('    m = Model("MRCPSP")')
        lines.append("    m.setParam('TimeLimit', 300)")
        lines.append("")
        lines.append("    # --- Data ---")
        lines.append(f"    n_act = {inst['n_activities']}")
        lines.append(f"    n_res = {inst['n_resources']}")
        lines.append(f"    n_modes = {inst['n_modes']}")
        lines.append(f"    capacities = {inst['resource_capacities']}")
        lines.append(f"    mode_dur = {inst['mode_durations']}")
        lines.append(f"    mode_req = {inst['mode_resource_requirements']}")
        lines.append(f"    mode_cost = {inst['mode_costs']}")
        lines.append(f"    budget = {inst['budget']}")
        lines.append(f"    # precedence_with_lags: each entry is (predecessor, successor, min_lag, max_lag_or_None)")
        lines.append(f"    precedence_with_lags = {lags_repr}")
        lines.append(f"    release_dates = {inst['release_dates']}")
        lines.append(f"    deadlines = {deadlines_repr}")
        lines.append("")
        lines.append("    # Derive precedence edges from precedence_with_lags")
        lines.append("    prec_edges = [(i, j) for (i, j, _, _) in precedence_with_lags]")
        lines.append("    time_lags = precedence_with_lags")
        lines.append("")
        lines.append("    # Big-M: sum of max durations + max release + slack")
        lines.append("    M = sum(max(mode_dur[i]) for i in range(n_act)) + max(release_dates) + 10")
        lines.append("")
        lines.append("    # --- Identify conflicting pairs ---")
        lines.append("    # Build transitive closure of precedence")
        lines.append("    adj = {i: [] for i in range(n_act)}")
        lines.append("    for (a, b) in prec_edges:")
        lines.append("        adj[a].append(b)")
        lines.append("    reachable = set()")
        lines.append("    for start in range(n_act):")
        lines.append("        visited = set()")
        lines.append("        stack = [start]")
        lines.append("        while stack:")
        lines.append("            node = stack.pop()")
        lines.append("            for nxt in adj[node]:")
        lines.append("                if nxt not in visited:")
        lines.append("                    visited.add(nxt)")
        lines.append("                    reachable.add((start, nxt))")
        lines.append("                    stack.append(nxt)")
        lines.append("")
        lines.append("    # Check all mode combinations for potential resource conflicts")
        lines.append("    conflicts = []")
        lines.append("    for i in range(n_act):")
        lines.append("        for j in range(i+1, n_act):")
        lines.append("            if (i,j) in reachable or (j,i) in reachable:")
        lines.append("                continue")
        lines.append("            conflict = False")
        lines.append("            for mi in range(n_modes):")
        lines.append("                for mj in range(n_modes):")
        lines.append("                    for k in range(n_res):")
        lines.append("                        if mode_req[i][mi][k] + mode_req[j][mj][k] > capacities[k]:")
        lines.append("                            conflict = True")
        lines.append("                            break")
        lines.append("                    if conflict:")
        lines.append("                        break")
        lines.append("                if conflict:")
        lines.append("                    break")
        lines.append("            if conflict:")
        lines.append("                conflicts.append((i, j))")
        lines.append("")
        lines.append("    # --- Variables ---")
        lines.append("    S = {}")
        lines.append("    for i in range(n_act):")
        lines.append('        S[i] = m.addVar(lb=release_dates[i], ub=M, vtype=GRB.CONTINUOUS, name=f"S_{i}")')
        lines.append('    Cmax = m.addVar(lb=0, ub=M, vtype=GRB.CONTINUOUS, name="Cmax")')
        lines.append("")
        lines.append("    x = {}")
        lines.append("    for i in range(n_act):")
        lines.append("        for md in range(n_modes):")
        lines.append('            x[i,md] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{md}")')
        lines.append("")
        lines.append("    Y = {}")
        lines.append("    for (i, j) in conflicts:")
        lines.append('        Y[(i,j)] = m.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{j}")')
        lines.append("")
        lines.append("    # --- Objective ---")
        lines.append(f"    m.setObjective(Cmax, {sense})")
        lines.append("")
        lines.append("    # --- Constraints ---")
        lines.append("    # 1. Mode selection: exactly one mode per activity")
        lines.append("    for i in range(n_act):")
        lines.append("        m.addConstr(quicksum(x[i,md] for md in range(n_modes)) == 1, name=f\"Mode_{i}\")")
        lines.append("")
        lines.append("    # 2. Precedence with min/max time lags")
        lines.append("    for (a, b, lag_min, lag_max) in time_lags:")
        lines.append("        d_eff_a = quicksum(mode_dur[a][md] * x[a,md] for md in range(n_modes))")
        lines.append("        m.addConstr(S[b] >= S[a] + d_eff_a + lag_min, name=f\"Prec_{a}_{b}\")")
        lines.append("        if lag_max is not None:")
        lines.append("            m.addConstr(S[b] <= S[a] + d_eff_a + lag_max, name=f\"MaxLag_{a}_{b}\")")
        lines.append("")
        lines.append("    # 3. Disjunctive for conflicting pairs")
        lines.append("    for (i, j) in conflicts:")
        lines.append("        y = Y[(i,j)]")
        lines.append("        d_eff_i = quicksum(mode_dur[i][md] * x[i,md] for md in range(n_modes))")
        lines.append("        d_eff_j = quicksum(mode_dur[j][md] * x[j,md] for md in range(n_modes))")
        lines.append("        m.addConstr(S[i] + d_eff_i <= S[j] + M*(1-y), name=f\"Disj_A_{i}_{j}\")")
        lines.append("        m.addConstr(S[j] + d_eff_j <= S[i] + M*y, name=f\"Disj_B_{i}_{j}\")")
        lines.append("")
        lines.append("    # 4. Budget constraint (non-renewable)")
        lines.append("    m.addConstr(quicksum(mode_cost[i][md] * x[i,md] for i in range(n_act) for md in range(n_modes)) <= budget, name=\"Budget\")")
        lines.append("")
        lines.append("    # 5. Deadlines")
        lines.append("    for i in range(n_act):")
        lines.append("        if deadlines[i] is not None:")
        lines.append("            d_eff_i = quicksum(mode_dur[i][md] * x[i,md] for md in range(n_modes))")
        lines.append("            m.addConstr(S[i] + d_eff_i <= deadlines[i], name=f\"Deadline_{i}\")")
        lines.append("")
        lines.append("    # 6. Makespan")
        lines.append("    for i in range(n_act):")
        lines.append("        d_eff_i = quicksum(mode_dur[i][md] * x[i,md] for md in range(n_modes))")
        lines.append("        m.addConstr(Cmax >= S[i] + d_eff_i, name=f\"Makespan_{i}\")")
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
        n_act = inst.get("n_activities", 0)
        n_res = inst.get("n_resources", 0)
        n_modes = inst.get("n_modes", 0)
        caps = inst.get("resource_capacities", [])
        mode_dur = inst.get("mode_durations", [])
        mode_req = inst.get("mode_resource_requirements", [])
        mode_cost = inst.get("mode_costs", [])
        budget = inst.get("budget", 0)
        time_lags = inst.get("time_lags", [])
        release_dates = inst.get("release_dates", [])
        deadlines = inst.get("deadlines", [])

        all_durs = [d for row in mode_dur for d in row]
        all_costs = [c for row in mode_cost for c in row]
        all_reqs = [r for act in mode_req for mode in act for r in mode]
        n_maxlags = sum(1 for _, _, _, lmax in time_lags if lmax is not None)
        n_deadlines = sum(1 for d in deadlines if d is not None)
        n_releases = sum(1 for r in release_dates if r > 0)

        user_content = f"""Write a professional project management memo for a multi-mode resource-constrained project scheduling problem.

**Scenario:**
A project has {n_act} activities, each executable in {n_modes} different modes (speed/cost tradeoffs).
Activities have dependency relationships and must share {n_res} resource types under capacity limits.
There is a fixed project budget of {budget} that caps total execution costs.

**Problem Scale:**
- {n_act} activities, each with {n_modes} execution modes
- Durations range from {min(all_durs)} to {max(all_durs)} time units across modes
- Costs range from {min(all_costs)} to {max(all_costs)} per mode
- {n_res} resource types with capacities {caps}
- {len(time_lags)} precedence relationships ({n_maxlags} with maximum delay limits)
- {n_releases} activities with earliest start times, {n_deadlines} with hard deadlines
- Resource requirements: 0-{max(all_reqs)} units per resource per mode

**Key Rules:**
- Each activity must use exactly one execution mode (different duration, resource usage, and cost)
- Precedence constraints with minimum delays: successor waits at least lag_min after predecessor finishes
- Some dependencies also have maximum delays: successor must start within lag_max of predecessor finishing
- Activities may have earliest allowed start times (release dates) and hard completion deadlines
- Resource capacity: concurrent resource usage must not exceed capacity
- Budget: total cost of chosen modes must not exceed the project budget
- CRITICAL SAFETY RULE: If there EXISTS even ONE pair of modes (one for each activity) and ONE resource where their combined usage exceeds capacity, those two independent activities CANNOT run at the same time. This is conservative — even if some mode pairs fit, the *possibility* of overload means they must be sequenced
- Goal: minimize total project duration (makespan) while staying within budget

**Your Task:**
Write a business memo describing this scheduling challenge.
IMPORTANT: Use ONLY these placeholders for data:
{{NUM_ACTIVITIES}}, {{NUM_RESOURCES}}, {{NUM_MODES}}, {{RESOURCE_CAPACITIES}},
{{MODE_DURATIONS}}, {{MODE_RESOURCE_REQUIREMENTS}}, {{MODE_COSTS}}, {{BUDGET}},
{{PRECEDENCE_EDGES_WITH_LAGS}}, {{RELEASE_DATES}}, {{DEADLINES}}

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
            "NUM_ACTIVITIES", "NUM_RESOURCES", "NUM_MODES", "RESOURCE_CAPACITIES",
            "MODE_DURATIONS", "MODE_RESOURCE_REQUIREMENTS", "MODE_COSTS", "BUDGET",
            "PRECEDENCE_EDGES_WITH_LAGS", "RELEASE_DATES", "DEADLINES",
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
        fmt["NUM_ACTIVITIES"] = str(inst.get("n_activities", ""))
        fmt["NUM_RESOURCES"] = str(inst.get("n_resources", ""))
        fmt["NUM_MODES"] = str(inst.get("n_modes", ""))

        caps = inst.get("resource_capacities", [])
        fmt["RESOURCE_CAPACITIES"] = "Resource Capacities:\n" + "\n".join(
            f"  - Resource {k+1}: {c} units" for k, c in enumerate(caps)
        )

        mode_dur = inst.get("mode_durations", [])
        dur_lines = ["Execution Mode Durations (time units per mode):"]
        for i, durs in enumerate(mode_dur):
            parts = ", ".join(f"Mode {m+1}: {d}" for m, d in enumerate(durs))
            dur_lines.append(f"  - Activity {i}: [{parts}]")
        fmt["MODE_DURATIONS"] = "\n".join(dur_lines)

        mode_req = inst.get("mode_resource_requirements", [])
        rr_lines = ["Resource Requirements per Mode (units consumed while activity is running):"]
        for i, modes in enumerate(mode_req):
            for m, reqs in enumerate(modes):
                parts = ", ".join(f"R{k+1}: {r}" for k, r in enumerate(reqs))
                rr_lines.append(f"  - Activity {i}, Mode {m+1}: [{parts}]")
        fmt["MODE_RESOURCE_REQUIREMENTS"] = "\n".join(rr_lines)

        mode_cost = inst.get("mode_costs", [])
        cost_lines = ["Execution Mode Costs:"]
        for i, costs in enumerate(mode_cost):
            parts = ", ".join(f"Mode {m+1}: {c}" for m, c in enumerate(costs))
            cost_lines.append(f"  - Activity {i}: [{parts}]")
        fmt["MODE_COSTS"] = "\n".join(cost_lines)

        fmt["BUDGET"] = str(inst.get("budget", ""))

        time_lags = inst.get("time_lags", [])
        lag_lines = ["Precedence Dependencies (with timing constraints):"]
        for (a, b, lag_min, lag_max) in time_lags:
            if lag_max is not None:
                lag_lines.append(
                    f"  - Activity {a} → Activity {b}: "
                    f"min delay = {lag_min}, max delay = {lag_max}"
                )
            elif lag_min > 0:
                lag_lines.append(
                    f"  - Activity {a} → Activity {b}: min delay = {lag_min}"
                )
            else:
                lag_lines.append(f"  - Activity {a} → Activity {b}")
        fmt["PRECEDENCE_EDGES_WITH_LAGS"] = "\n".join(lag_lines)

        release_dates = inst.get("release_dates", [])
        rel_lines = ["Earliest Start Times (release dates):"]
        for i, r in enumerate(release_dates):
            rel_lines.append(f"  - Activity {i}: {r}")
        fmt["RELEASE_DATES"] = "\n".join(rel_lines)

        deadlines = inst.get("deadlines", [])
        dl_lines = ["Completion Deadlines:"]
        for i, d in enumerate(deadlines):
            if d is not None:
                dl_lines.append(f"  - Activity {i}: must finish by time {d}")
            else:
                dl_lines.append(f"  - Activity {i}: no deadline")
        fmt["DEADLINES"] = "\n".join(dl_lines)

        result = llm_description
        for ph, val in fmt.items():
            result = result.replace(f"{{{ph}}}", val)

        remaining = [p for p in required if f"{{{p}}}" in result]
        if remaining:
            print(f"Placeholders not filled: {remaining}")
            return None

        return result
