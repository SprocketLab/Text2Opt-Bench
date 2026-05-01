import numpy as np
from typing import Tuple
from main.generation.base_problem import BaseProblem


class QueuingStaffingGenerator(BaseProblem):
    """
    Queuing-Based Service Staffing Problem (ILP) with INDUCED CONSTRAINTS.

    A service center has multiple stations (departments). Each station receives
    customers at a known arrival rate. Multiple staff types (skill levels) can
    be assigned to stations, each with different handling times and hourly costs.

    INDUCED CONSTRAINTS — the LLM must derive constraint coefficients from raw data:

    1. UNIT CONVERSION (service time → service rate):
       Given: average_handling_time[k][i] in MINUTES per customer
       Must compute: service_rate[k][i] = 60 / handling_time[k][i] customers/HOUR
       (0 handling time means staff type k is not qualified for station i)

    2. UTILIZATION THRESHOLD (arrival rate + policy → capacity requirement):
       Given: arrival_rate[i] customers/hour, max_utilization[i] (e.g. 0.85)
       Must compute: min_capacity[i] = arrival_rate[i] / max_utilization[i]
       (the station needs enough aggregate service rate to stay below utilization cap)

    Decision Variables:
    - Assign_K{k}_S{i}: Integer — number of staff type k assigned to station i

    Constraints:
    1. Staff availability:  sum_i Assign[k][i] <= available[k]
    2. Capacity (INDUCED):  sum_k Assign[k][i] * (60/handling_time[k][i])
                            >= arrival_rate[i] / max_utilization[i]
    3. Min staffing:        sum_k Assign[k][i] >= min_staff[i]

    Objective (MINIMIZE):
    sum_{k,i} hourly_cost[k] * Assign[k][i]
    """

    SYSTEM_PROMPT = """You are a Service Operations Manager writing a memo about a staffing decision.

**TASK:** Write a natural, business-like problem description for a service center staffing problem.

**TONE:** Professional operations memo. The reader should understand the staffing challenge intuitively.

**REQUIRED PLACEHOLDERS (include ALL exactly once):**
{NUM_STATIONS}, {NUM_STAFF_TYPES}, {ARRIVAL_RATES}, {HANDLING_TIMES}, {HOURLY_COSTS}, {AVAILABLE_STAFF}, {MAX_UTILIZATION}, {MIN_STAFF}

**CRITICAL - DATA HANDLING:**
- Use ONLY the placeholders above for data. Do NOT write out sample values.
- Each placeholder should appear EXACTLY ONCE.
- Place all data in a dedicated "ANNEX" section at the end.

**WHAT TO INCLUDE:**
- A service center with multiple service stations (help desks, counters, departments)
- Customers arrive at each station at a known rate (customers per hour)
- Multiple types of staff with different skill levels can be assigned
- Each staff type handles customers at different speeds depending on the station
- Handling speed is described as average time per customer (in minutes)
- Some staff types are not qualified for certain stations (cannot serve there)
- Management policy limits how busy staff should be (utilization cap)
- Staff should not be occupied more than a certain fraction of their time
- Each station needs a minimum number of staff for coverage
- Total staff of each type is limited
- Goal: Assign staff to stations at minimum total hourly cost while meeting all service requirements

**WHAT TO AVOID:**
- Mathematical formulas or queuing theory terminology
- Do NOT say "M/M/1", "Erlang", "utilization formula"
- Do NOT explicitly say "convert minutes to hourly rate" or give conversion formulas
- Do NOT list sample data values - only use placeholders
- Do NOT use terms like "decision variables", "objective function", "constraints"

**KEY SUBTLETY (describe naturally, don't give formulas):**
- Handling time is given in minutes per customer, but arrival rates are per hour
- The utilization policy means the station's combined service speed must exceed arrival volume with a safety margin
- Describe this as a practical business policy, not as a mathematical formula

**SUGGESTED STRUCTURE:**
1. Subject/Context: Service center staffing review
2. Situation: Current operations and service demand
3. Challenge: Assign staff efficiently across stations
4. Requirements: Service quality policies
5. Goal: Minimize staffing cost
6. Annex - Operational Data: All placeholders"""

    def __init__(
        self,
        n_stations: int = 5,
        n_staff_types: int = 4,
        arrival_rate_range: Tuple[float, float] = (10.0, 50.0),
        handling_time_range: Tuple[float, float] = (3.0, 15.0),
        hourly_cost_range: Tuple[float, float] = (15.0, 45.0),
        utilization_range: Tuple[float, float] = (0.75, 0.90),
        min_staff_range: Tuple[int, int] = (1, 3),
        unqualified_fraction: float = 0.15,
        prob_type: str = "LP",
        goal: str = "MINIMIZE",
    ):
        self.n_stations = n_stations
        self.n_staff_types = n_staff_types
        self.arrival_rate_range = arrival_rate_range
        self.handling_time_range = handling_time_range
        self.hourly_cost_range = hourly_cost_range
        self.utilization_range = utilization_range
        self.min_staff_range = min_staff_range
        self.unqualified_fraction = unqualified_fraction

        n_vars = n_staff_types * n_stations

        # Constraints:
        # Staff availability: n_staff_types
        # Capacity (induced): n_stations
        # Min staffing: n_stations
        n_constrs = n_staff_types + 2 * n_stations

        super().__init__(
            n_vars, n_constrs,
            prob_type=prob_type, goal=goal,
            problem_type="queuing staffing",
        )

        # Instance data
        self.arrival_rates = None        # [station] customers/hour
        self.handling_times = None       # [staff_type][station] minutes/customer (0=unqualified)
        self.hourly_costs = None         # [staff_type] $/hour
        self.available_staff = None      # [staff_type] count
        self.max_utilization = None      # [station] fraction
        self.min_staff = None            # [station] count

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, time_limit: float = 300.0):
        self._generate_instance_data()
        self._build_matrices()
        self.solve(time_limit=time_limit)
        return self

    def _generate_instance_data(self):
        n_s = self.n_stations
        n_k = self.n_staff_types

        # Arrival rates (customers/hour) — integers for clarity
        self.arrival_rates = np.random.randint(
            int(self.arrival_rate_range[0]),
            int(self.arrival_rate_range[1]) + 1,
            size=n_s
        ).astype(float)

        # Handling times (minutes/customer) — rounded to 1 decimal
        # Higher-skill (higher-cost) staff types tend to be faster
        self.handling_times = np.zeros((n_k, n_s))
        for k in range(n_k):
            # Skill factor: first staff type is slowest, last is fastest
            skill_factor = 1.0 - 0.15 * k  # 1.0, 0.85, 0.70, 0.55, ...
            skill_factor = max(skill_factor, 0.4)
            for i in range(n_s):
                base_time = np.random.uniform(*self.handling_time_range)
                self.handling_times[k, i] = round(base_time * skill_factor, 1)

        # Mark some (staff_type, station) pairs as unqualified (handling_time = 0)
        n_pairs = n_k * n_s
        n_unqualified = int(n_pairs * self.unqualified_fraction)
        if n_unqualified > 0:
            # Never make ALL staff types unqualified for a station
            candidates = []
            for k in range(n_k):
                for i in range(n_s):
                    candidates.append((k, i))

            np.random.shuffle(candidates)
            removed = 0
            qualified_count = {i: n_k for i in range(n_s)}  # track per station
            for k, i in candidates:
                if removed >= n_unqualified:
                    break
                if qualified_count[i] > 2:  # keep at least 2 qualified types per station
                    self.handling_times[k, i] = 0.0
                    qualified_count[i] -= 1
                    removed += 1

        # Hourly costs — higher index = higher skill = higher cost
        base_costs = np.random.uniform(
            self.hourly_cost_range[0], self.hourly_cost_range[1], size=n_k
        )
        # Sort so that faster staff costs more
        self.hourly_costs = np.sort(base_costs).round(2)

        # Max utilization per station
        self.max_utilization = np.random.uniform(
            self.utilization_range[0], self.utilization_range[1], size=n_s
        ).round(2)

        # Min staff per station
        self.min_staff = np.random.randint(
            self.min_staff_range[0], self.min_staff_range[1] + 1, size=n_s
        ).astype(float)

        # Available staff — ensure feasibility
        # Compute how many staff are needed at minimum
        # For each station, compute min servers needed for capacity:
        #   min_capacity = arrival_rate / max_utilization
        #   best_rate = max over qualified k of 60/handling_time[k][i]
        #   min_servers_i = ceil(min_capacity / best_rate)
        total_needed = np.zeros(n_k)
        for i in range(n_s):
            min_cap = self.arrival_rates[i] / self.max_utilization[i]
            # Find best qualified rate for this station
            best_rate = 0
            best_k = 0
            for k in range(n_k):
                if self.handling_times[k, i] > 0:
                    rate = 60.0 / self.handling_times[k, i]
                    if rate > best_rate:
                        best_rate = rate
                        best_k = k
            min_servers = int(np.ceil(min_cap / best_rate)) if best_rate > 0 else 1
            min_servers = max(min_servers, int(self.min_staff[i]))
            total_needed[best_k] += min_servers

        # Generate available staff with buffer
        self.available_staff = np.zeros(n_k)
        for k in range(n_k):
            base = max(total_needed[k], 2)
            self.available_staff[k] = float(int(base * np.random.uniform(1.2, 1.8)))
        self.available_staff = self.available_staff.astype(float)

    # ------------------------------------------------------------------
    # Index helper
    # ------------------------------------------------------------------

    def _idx(self, k, i):
        """Variable index for Assign_K{k}_S{i}."""
        return k * self.n_stations + i

    # ------------------------------------------------------------------
    # Build A, b, c
    # ------------------------------------------------------------------

    def _build_matrices(self):
        n_s = self.n_stations
        n_k = self.n_staff_types

        # ---- Variable names ----
        self.var_names = []
        for k in range(n_k):
            for i in range(n_s):
                self.var_names.append(f"Assign_K{k}_S{i}")

        # ---- Variable types (all Integer) ----
        self.var_types = ["Integer"] * self.n

        # ---- Variable bounds ----
        self.var_bounds = []
        for k in range(n_k):
            for i in range(n_s):
                if self.handling_times[k, i] == 0:
                    # Unqualified: force to 0
                    self.var_bounds.append((0.0, 0.0))
                else:
                    self.var_bounds.append((0.0, self.available_staff[k]))

        # ---- Objective: minimize total hourly cost ----
        self.c = np.zeros(self.n)
        for k in range(n_k):
            for i in range(n_s):
                self.c[self._idx(k, i)] = self.hourly_costs[k]

        # ---- Constraints ----
        self.A = np.zeros((self.m, self.n))
        self.b = np.zeros(self.m)
        self.senses = []
        self.constr_names = []
        row = 0

        # C1: Staff availability — sum_i Assign[k][i] <= Available[k]
        for k in range(n_k):
            for i in range(n_s):
                self.A[row, self._idx(k, i)] = 1.0
            self.b[row] = self.available_staff[k]
            self.senses.append("<=")
            self.constr_names.append(f"Avail_K{k}")
            row += 1

        # C2: Capacity (INDUCED) —
        #   sum_k Assign[k][i] * (60/handling_time[k][i]) >= arrival_rate[i] / max_utilization[i]
        for i in range(n_s):
            for k in range(n_k):
                ht = self.handling_times[k, i]
                if ht > 0:
                    service_rate = 60.0 / ht  # INDUCED: convert minutes → hourly rate
                    self.A[row, self._idx(k, i)] = service_rate
                # else: coefficient is 0 (unqualified)
            self.b[row] = self.arrival_rates[i] / self.max_utilization[i]  # INDUCED threshold
            self.senses.append(">=")
            self.constr_names.append(f"Capacity_S{i}")
            row += 1

        # C3: Min staffing — sum_k Assign[k][i] >= min_staff[i]
        for i in range(n_s):
            for k in range(n_k):
                if self.handling_times[k, i] > 0:
                    self.A[row, self._idx(k, i)] = 1.0
                # Unqualified staff don't count toward min staffing
            self.b[row] = self.min_staff[i]
            self.senses.append(">=")
            self.constr_names.append(f"MinStaff_S{i}")
            row += 1

        assert row == self.m, f"Expected {self.m} constraints, built {row}"

    # ------------------------------------------------------------------
    # Instance data
    # ------------------------------------------------------------------

    def _get_instance_data(self):
        return {
            "n_stations": self.n_stations,
            "n_staff_types": self.n_staff_types,
            "arrival_rates": self.arrival_rates.tolist(),
            "handling_times": self.handling_times.tolist(),
            "hourly_costs": self.hourly_costs.tolist(),
            "available_staff": self.available_staff.tolist(),
            "max_utilization": self.max_utilization.tolist(),
            "min_staff_per_station": self.min_staff.tolist(),
            # NOTE: service_rates and capacity_thresholds are intentionally NOT included
            # The LLM must compute them from handling_times and arrival_rates/max_utilization
        }

    # ------------------------------------------------------------------
    # Gold solution code
    # ------------------------------------------------------------------

    def generate_gurobi_code_reference(self, data: dict) -> str:
        inst = data.get("instance_data", {})
        meta = data.get("meta", {})

        n_s = inst.get("n_stations", 0)
        n_k = inst.get("n_staff_types", 0)
        arrival = inst.get("arrival_rates", [])
        handling = inst.get("handling_times", [])
        costs = inst.get("hourly_costs", [])
        avail = inst.get("available_staff", [])
        max_util = inst.get("max_utilization", [])
        min_staff = inst.get("min_staff_per_station", [])

        sense = "GRB.MINIMIZE" if meta.get("goal") == "MINIMIZE" else "GRB.MAXIMIZE"

        lines = []
        lines.append("```python")
        lines.append("from gurobipy import *")
        lines.append("")
        lines.append("def solve_problem():")
        lines.append('    m = Model("Queuing_Staffing")')
        lines.append("    m.setParam('TimeLimit', 300)")
        lines.append("")
        lines.append("    # --- Data ---")
        lines.append(f"    n_stations = {n_s}")
        lines.append(f"    n_staff_types = {n_k}")
        lines.append(f"    arrival_rates = {arrival}  # customers/hour")
        lines.append(f"    handling_times = {handling}  # minutes/customer [staff_type][station]")
        lines.append(f"    hourly_costs = {costs}  # $/hour per staff type")
        lines.append(f"    available_staff = {avail}  # max available per type")
        lines.append(f"    max_utilization = {max_util}  # utilization cap per station")
        lines.append(f"    min_staff = {min_staff}  # minimum staff per station")
        lines.append("")
        lines.append("    # COMPUTE service rates from handling times (INDUCED)")
        lines.append("    # handling_time is in minutes/customer, we need customers/hour")
        lines.append("    service_rate = {}")
        lines.append("    for k in range(n_staff_types):")
        lines.append("        for i in range(n_stations):")
        lines.append("            ht = handling_times[k][i]")
        lines.append("            service_rate[(k,i)] = 60.0 / ht if ht > 0 else 0.0")
        lines.append("")
        lines.append("    # COMPUTE capacity thresholds (INDUCED)")
        lines.append("    min_capacity = [arrival_rates[i] / max_utilization[i] for i in range(n_stations)]")
        lines.append("")
        lines.append("    # --- Variables ---")
        lines.append("    assign = {}")
        lines.append("    for k in range(n_staff_types):")
        lines.append("        for i in range(n_stations):")
        lines.append("            ub = int(available_staff[k]) if handling_times[k][i] > 0 else 0")
        lines.append('            assign[(k,i)] = m.addVar(lb=0, ub=ub, vtype=GRB.INTEGER,')
        lines.append('                                     name=f"Assign_K{k}_S{i}")')
        lines.append("")
        lines.append("    # --- Objective: minimize total hourly cost ---")
        lines.append("    m.setObjective(")
        lines.append("        quicksum(hourly_costs[k] * assign[(k,i)]")
        lines.append("                 for k in range(n_staff_types) for i in range(n_stations)),")
        lines.append(f"        {sense})")
        lines.append("")
        lines.append("    # --- Constraints ---")
        lines.append("    # Staff availability")
        lines.append("    for k in range(n_staff_types):")
        lines.append("        m.addConstr(")
        lines.append("            quicksum(assign[(k,i)] for i in range(n_stations)) <= available_staff[k],")
        lines.append('            name=f"Avail_K{k}")')
        lines.append("")
        lines.append("    # Capacity (induced): aggregate service rate >= arrival / max_util")
        lines.append("    for i in range(n_stations):")
        lines.append("        m.addConstr(")
        lines.append("            quicksum(service_rate[(k,i)] * assign[(k,i)]")
        lines.append("                     for k in range(n_staff_types) if handling_times[k][i] > 0)")
        lines.append("            >= min_capacity[i],")
        lines.append('            name=f"Capacity_S{i}")')
        lines.append("")
        lines.append("    # Minimum staffing")
        lines.append("    for i in range(n_stations):")
        lines.append("        m.addConstr(")
        lines.append("            quicksum(assign[(k,i)] for k in range(n_staff_types)")
        lines.append("                     if handling_times[k][i] > 0)")
        lines.append("            >= min_staff[i],")
        lines.append('            name=f"MinStaff_S{i}")')
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

        n_s = inst.get("n_stations", 0)
        n_k = inst.get("n_staff_types", 0)
        arrival = inst.get("arrival_rates", [])
        handling = inst.get("handling_times", [])

        # Count unqualified pairs
        n_unqualified = sum(1 for k in range(n_k) for i in range(n_s)
                           if handling[k][i] == 0)
        total_pairs = n_k * n_s

        # Compute avg handling time for context (excluding zeros)
        all_times = [handling[k][i] for k in range(n_k) for i in range(n_s)
                     if handling[k][i] > 0]
        avg_time = sum(all_times) / len(all_times) if all_times else 0

        user_content = f"""Write a professional operations memo for a service center staffing problem.

**Scenario:**
A service center has {n_s} service stations and {n_k} types of staff available.
Customers arrive at each station at a known rate (customers per hour).
Staff handle customers at different speeds depending on their skill and the station.

**Problem Scale:**
- {n_s} stations, {n_k} staff types = {total_pairs} possible assignments
- {n_unqualified} of those are restricted (staff not qualified for that station)
- Handling times range from {min(all_times):.1f} to {max(all_times):.1f} minutes per customer (avg {avg_time:.1f})
- Arrival rates range from {min(arrival):.0f} to {max(arrival):.0f} customers/hour

**Key Rules:**
- Handling speed is given as average minutes per customer (varies by staff type and station)
- Arrival volume is given as customers per hour
- Management policy: staff at each station must not be busy more than a target fraction of their time
- Each station needs a minimum number of staff for coverage
- Total staff of each type is limited
- Some staff types cannot work at certain stations
- Goal: Minimize total hourly staffing cost

**Your Task:**
Write a business memo describing this staffing challenge.
IMPORTANT: Use ONLY these placeholders for data:
{{NUM_STATIONS}}, {{NUM_STAFF_TYPES}}, {{ARRIVAL_RATES}}, {{HANDLING_TIMES}},
{{HOURLY_COSTS}}, {{AVAILABLE_STAFF}}, {{MAX_UTILIZATION}}, {{MIN_STAFF}}

Each placeholder should appear exactly ONCE in an Annex section.
Do NOT write out any sample data values.
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
            "NUM_STATIONS", "NUM_STAFF_TYPES", "ARRIVAL_RATES", "HANDLING_TIMES",
            "HOURLY_COSTS", "AVAILABLE_STAFF", "MAX_UTILIZATION", "MIN_STAFF",
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
        fmt["NUM_STATIONS"] = str(inst.get("n_stations", ""))
        fmt["NUM_STAFF_TYPES"] = str(inst.get("n_staff_types", ""))
        fmt["HOURLY_COSTS"] = str(inst.get("hourly_costs", []))
        fmt["AVAILABLE_STAFF"] = str(inst.get("available_staff", []))
        fmt["MIN_STAFF"] = str(inst.get("min_staff_per_station", []))

        # Arrival rates
        arrival = inst.get("arrival_rates", [])
        ar_lines = ["Arrival Rates (customers per hour):"]
        for i, rate in enumerate(arrival):
            ar_lines.append(f"  - Station {i}: {rate:.0f} customers/hour")
        fmt["ARRIVAL_RATES"] = "\n".join(ar_lines)

        # Max utilization
        max_util = inst.get("max_utilization", [])
        mu_lines = ["Maximum Utilization Policy (fraction of time staff may be busy):"]
        for i, u in enumerate(max_util):
            mu_lines.append(f"  - Station {i}: {u:.2f}")
        fmt["MAX_UTILIZATION"] = "\n".join(mu_lines)

        # Handling times as table
        handling = inst.get("handling_times", [])
        n_s = inst.get("n_stations", 0)
        n_k = inst.get("n_staff_types", 0)
        ht_lines = [f"Average Handling Time (minutes per customer, by staff type and station):"]
        ht_lines.append(f"  A value of 0 means the staff type is not qualified for that station.")
        header = "Staff Type \\ Station | " + " | ".join(f"S{i}" for i in range(n_s))
        ht_lines.append(header)
        ht_lines.append("-" * len(header))
        for k in range(n_k):
            row_vals = " | ".join(
                f"{handling[k][i]:5.1f}" if handling[k][i] > 0 else "  N/A"
                for i in range(n_s)
            )
            ht_lines.append(f"Type {k}              | {row_vals}")
        fmt["HANDLING_TIMES"] = "\n".join(ht_lines)

        result = llm_description
        for ph, val in fmt.items():
            result = result.replace(f"{{{ph}}}", val)

        remaining = [p for p in required if f"{{{p}}}" in result]
        if remaining:
            print(f"Placeholders not filled: {remaining}")
            return None

        return result
