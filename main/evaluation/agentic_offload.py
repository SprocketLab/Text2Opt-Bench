#!/usr/bin/env python3
"""
Agentic-offload evaluation helpers.

Two offload modes:
  - instance_data mode (template problems): serialize raw instance_data fields to a
    local JSON file; the LLM reads the file and derives ALL variables/constraints itself.
  - array mode (abstract resource_allocation problems): serialize the pre-computed
    (A, b, c, bounds) arrays to a local JSON file; the LLM loads and builds the model.

In both modes numeric data never appears in the prompt.  On failure, a repair loop
feeds back error signals (stderr, evaluation status) and asks for a fix.
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Default cache location (relative to project root)
DEFAULT_ARRAY_CACHE_DIR = Path("synthetic_dataset/_agentic_array_cache")

AGENTIC_SYSTEM_PROMPT = (
    "You are an optimization coding agent. "
    "Return only executable Python code in a single fenced ```python ... ``` block."
)

_ARRAY_SCHEMA_NOTE = textwrap.dedent(
    """
    Array file schema (JSON):
      goal              : str          — "MINIMIZE" or "MAXIMIZE"
      var_names         : list[str]    — variable names
      var_types         : list[str]    — "Continuous", "Integer", or "Binary"
      lb                : list[float]  — lower bounds
      ub                : list[float|null] — upper bounds (null = GRB.INFINITY)
      c                 : list[float]  — objective coefficients
      constraint_names  : list[str]    — constraint names
      constraint_senses : list[str]    — "<=", ">=", or "=="
      rhs               : list[float]  — right-hand side values
      A_sparse          : list of [row_idx, col_idx, value] triplets (0-indexed)
    """
).strip()

_CODE_REQUIREMENTS = textwrap.dedent(
    """
    Requirements:
      1. Return exactly one fenced ```python ... ``` block.
      2. ONLY import `json` and `from gurobipy import *`. No other imports. No try/except.
      3. gurobipy IS available — do NOT create mock or fallback implementations.
      4. Define `solve_problem()` with no required arguments.
      5. Inside `solve_problem()`, open and parse PROBLEM_ARRAY_PATH with json.load().
      6. Build Gurobi variables from var_names / var_types / lb / ub.
         Use GRB.INFINITY for null upper bounds.
      7. Build the objective from vector `c` and `goal` (GRB.MINIMIZE or GRB.MAXIMIZE).
      8. Reconstruct the constraint matrix from A_sparse (list of [row, col, val]).
         Build a dict A_dict[row][col] = val, then use quicksum for each constraint row.
      9. Add all constraints using constraint_senses and rhs.
     10. Set `m.setParam("TimeLimit", 100)`.
     11. Call `m.optimize()` and return `m`.
     12. Do NOT hardcode any numeric coefficients, bounds, or RHS values.
    """
).strip()


# ---------------------------------------------------------------------------
# Array serialization
# ---------------------------------------------------------------------------

def _parse_ub(val: Any) -> Optional[float]:
    """Convert bound value to float or None (= GRB.INFINITY)."""
    if val is None:
        return None
    if isinstance(val, str) and val.lower() in ("inf", "infinity", "+inf"):
        return None
    try:
        f = float(val)
        return None if f == float("inf") else f
    except (TypeError, ValueError):
        return None


def build_problem_array(problem_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a problem JSON dict into a flat array representation.
    Uses sparse format for the constraint matrix (A_sparse).
    """
    variables = problem_data.get("variables", {})
    constraints = problem_data.get("constraints", {})
    meta = problem_data.get("meta", {})

    var_names = list(variables.keys())
    constraint_names = list(constraints.keys())
    constraint_index = {name: i for i, name in enumerate(constraint_names)}

    n = len(var_names)

    c = [0.0] * n
    lb = [0.0] * n
    ub: list = [None] * n
    var_types = ["Continuous"] * n
    A_sparse = []  # list of [row, col, value]

    for j, var_name in enumerate(var_names):
        var_data = variables[var_name]

        c[j] = float(var_data.get("objective_linear_coefficient", 0.0))

        bounds = var_data.get("range", [0.0, None])
        lb[j] = float(bounds[0]) if bounds[0] is not None else 0.0
        ub[j] = _parse_ub(bounds[1] if len(bounds) > 1 else None)

        var_types[j] = str(var_data.get("type", "Continuous"))

        for constr_name, coeff in (var_data.get("resource_costs") or {}).items():
            if constr_name in constraint_index:
                row = constraint_index[constr_name]
                val = float(coeff)
                if val != 0.0:
                    A_sparse.append([row, j, val])

    senses = []
    rhs = []
    for constr_name in constraint_names:
        constr_data = constraints[constr_name]
        senses.append(str(constr_data.get("sense", "==")))
        rhs.append(float(constr_data.get("rhs", 0.0)))

    return {
        "goal": str(meta.get("goal", "MINIMIZE")).upper(),
        "var_names": var_names,
        "var_types": var_types,
        "lb": lb,
        "ub": ub,
        "c": c,
        "constraint_names": constraint_names,
        "constraint_senses": senses,
        "rhs": rhs,
        "A_sparse": A_sparse,
        "num_vars": n,
        "num_constraints": len(constraint_names),
    }


def persist_problem_array(
    problem_data: Dict[str, Any],
    problem_id: str,
    cache_dir: Path = DEFAULT_ARRAY_CACHE_DIR,
) -> Path:
    """
    Serialize problem arrays to disk.  Returns the absolute path to the file.
    Skips writing if the file already exists (idempotent).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(problem_id).stem
    out_path = (cache_dir / f"{stem}_arrays.json").resolve()

    if not out_path.exists():
        payload = build_problem_array(problem_data)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    return out_path


# ---------------------------------------------------------------------------
# Instance-data serialization (template problems)
# ---------------------------------------------------------------------------

def persist_instance_data(
    problem_data: Dict[str, Any],
    problem_id: str,
    cache_dir: Path = DEFAULT_ARRAY_CACHE_DIR,
) -> Path:
    """
    Serialize only instance_data to disk. Returns the absolute path to the file.
    Skips writing if the file already exists (idempotent).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(problem_id).stem
    # Include problem_type to avoid filename collisions across categories
    ptype = problem_data.get("meta", {}).get("problem_type", "")
    prefix = ptype.replace(" ", "_") + "_" if ptype else ""
    out_path = (cache_dir / f"{prefix}{stem}_instance_data.json").resolve()

    if not out_path.exists():
        inst = problem_data.get("instance_data", {})
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(inst, f)

    return out_path


def persist_agentic_data(
    problem_data: Dict[str, Any],
    problem_id: str,
    cache_dir: Path = DEFAULT_ARRAY_CACHE_DIR,
) -> Tuple[Path, str]:
    """
    Unified entry point: auto-detect offload mode and persist data.

    Returns (data_path, mode):
    - "instance_data": problem has non-empty instance_data → template problem
    - "two_phase": no instance_data → two-phase extract-then-solve
      (data_path is the target path for the extracted JSON; may not exist yet)
    """
    inst = problem_data.get("instance_data")
    if inst:
        path = persist_instance_data(problem_data, problem_id, cache_dir=cache_dir)
        return path, "instance_data"
    else:
        # Direct-solve mode: no data file needed.
        # The LLM solves from the description directly (like 1-shot)
        # but with an agentic repair loop on failure.
        return None, "direct_solve"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_CODE_TEMPLATE = textwrap.dedent(
    r"""
    ```python
    import json
    from gurobipy import *

    PROBLEM_ARRAY_PATH = r"<FILL_IN_PATH>"

    def solve_problem():
        with open(PROBLEM_ARRAY_PATH, "r") as f:
            d = json.load(f)

        m = Model("problem")
        m.setParam("TimeLimit", 100)

        type_map = {"Continuous": GRB.CONTINUOUS, "Integer": GRB.INTEGER, "Binary": GRB.BINARY}
        vars_ = []
        for j, name in enumerate(d["var_names"]):
            ub = d["ub"][j] if d["ub"][j] is not None else GRB.INFINITY
            vars_.append(m.addVar(lb=d["lb"][j], ub=ub,
                                  vtype=type_map.get(d["var_types"][j], GRB.CONTINUOUS),
                                  name=name))
        m.update()

        obj = quicksum(d["c"][j] * vars_[j] for j in range(len(vars_)))
        m.setObjective(obj, GRB.MINIMIZE if d["goal"] == "MINIMIZE" else GRB.MAXIMIZE)

        # Build sparse row dict from A_sparse
        rows = {}
        for row, col, val in d["A_sparse"]:
            rows.setdefault(row, {})[col] = val

        for i, (sense, rhs) in enumerate(zip(d["constraint_senses"], d["rhs"])):
            expr = quicksum(rows.get(i, {}).get(j, 0.0) * vars_[j]
                            for j in rows.get(i, {}).keys())
            if sense == "<=":
                m.addConstr(expr <= rhs)
            elif sense == ">=":
                m.addConstr(expr >= rhs)
            else:
                m.addConstr(expr == rhs)

        m.optimize()
        return m
    ```
    """
).strip()


# Separator keywords that mark where numeric data begins in LLM_description
_DATA_SECTION_MARKERS = ["Annex", "Appendix", "---\n", "===\n"]


def extract_problem_context(problem_data: Dict[str, Any]) -> str:
    """
    Fallback: extract the textual problem context from LLM_description by stripping
    numeric data sections (finds first known data-section marker).
    Used when LLM_description_template is not available.
    """
    desc = problem_data.get("LLM_description", "").strip()
    if not desc:
        meta = problem_data.get("meta", {})
        return f"Problem type: {meta.get('problem_type', 'optimization')}"

    best_idx = len(desc)
    for marker in _DATA_SECTION_MARKERS:
        idx = desc.find(marker)
        if 0 < idx < best_idx:
            best_idx = idx

    return desc[:best_idx].strip()


def _fill_template_with_refs(template: str, instance_data: Dict[str, Any]) -> str:
    """
    Replace {PLACEHOLDER} tokens with d["key"] references for the agentic prompt.

    Placeholder names are the UPPERCASE version of the instance_data key name.
    e.g., {SUPPLIES} → `d["supplies"]`
          {FACILITY_COORDINATES} → `d["facility_coordinates"]`

    Any placeholder not matched to an instance_data key is left unchanged.
    """
    result = template
    for key in instance_data.keys():
        placeholder = "{" + key.upper() + "}"
        result = result.replace(placeholder, f'`d["{key}"]`')
    return result


def build_context_from_template(problem_data: Dict[str, Any]) -> str:
    """
    Build the problem context string for the agentic prompt.

    Priority:
      1. If LLM_description_template exists: replace {PLACEHOLDER} tokens with
         d["key"] references — gives the LLM full semantic context with no numbers.
      2. Fallback: extract_problem_context() strips numeric sections from LLM_description.
    """
    template = problem_data.get("LLM_description_template", "").strip()
    if template:
        inst = problem_data.get("instance_data", {})
        return _fill_template_with_refs(template, inst)
    return extract_problem_context(problem_data)


def _list_shape(val: Any) -> str:
    """Recursively determine the shape of a nested list and return a shape string."""
    if not isinstance(val, list):
        return ""
    n = len(val)
    if n == 0:
        return f"[0]"
    inner = _list_shape(val[0])
    return f"[{n}]{inner}"


# Per-problem-type annotations for instance_data fields.
# For arrays: dimension labels (e.g., "demands[unit][supply][day]")
# For scalars: units or semantics (e.g., "km", "Ω·m")
_FIELD_ANNOTATIONS: Dict[str, Dict[str, str]] = {
    "disaster response logistics": {
        "demands": "demands[unit][supply][day]",
        "depot_supplies": "depot_supplies[depot][supply][day]",
        "transport_costs": "transport_costs[depot][unit]",
        "risk_costs": "risk_costs[depot][unit]",
        "penalty_costs": "penalty_costs[supply]",
        "supply_volumes": "supply_volumes[supply]",
    },
    "power transmission": {
        "node_coordinates": "node_coordinates[node][x,y] — km",
        "wire_resistivity": "Ω·m",
        "wire_cross_section": "mm²",
        "operating_voltage": "kV",
        "line_capacity": "MW",
        "build_cost_per_km": "$/km",
        "loss_cost_rate": "$/unit-loss",
        "power_demands": "MW per node",
        "generator_capacities": "MW per node",
    },
    "Transportation Problem": {
        "supplies": "supplies[source]",
        "demands": "demands[destination]",
        "cost_matrix": "cost_matrix[source][destination]",
    },
    "facility location": {
        "facility_coordinates": "facility_coordinates[facility][x,y]",
        "customer_coordinates": "customer_coordinates[customer][x,y]",
        "facility_capacities": "facility_capacities[facility]",
        "customer_demands": "customer_demands[customer]",
        "fixed_costs": "fixed_costs[facility]",
    },
    "vehicle routing": {
        "depot_coordinates": "depot_coordinates[x,y]",
        "customer_coordinates": "customer_coordinates[customer][x,y]",
        "customer_demands": "customer_demands[customer] — units of goods",
        "time_windows": "time_windows[node][earliest,latest] — index 0=depot, 1..N=customers",
        "service_times": "service_times[customer] — time spent at each customer",
    },
    "job shop scheduling": {
        "processing_times": "processing_times[job][operation] — duration in time units",
        "machine_assignments": "machine_assignments[job][operation] — machine index (0-based)",
    },
    "queuing staffing": {
        "arrival_rates": "arrival_rates[station] — customers/hour",
        "handling_times": "handling_times[staff_type][station] — minutes/customer (0=not qualified)",
        "hourly_costs": "hourly_costs[staff_type] — $/hour",
        "available_staff": "available_staff[staff_type] — max count",
        "max_utilization": "max_utilization[station] — fraction (e.g. 0.85)",
        "min_staff_per_station": "min_staff_per_station[station] — minimum headcount",
    },
    "modified facility location": {
        "facility_coordinates": "facility_coordinates[facility][x,y]",
        "customer_coordinates": "customer_coordinates[customer][x,y]",
        "facility_capacities": "facility_capacities[facility]",
        "customer_demands": "customer_demands[customer]",
        "fixed_costs": "fixed_costs[facility]",
        "modifiers": "list of active modifier names",
        "modifier_params": "modifier-specific parameters — all facility/customer indices are 0-based (e.g. linking_pair [2,3] means facility index 2 and 3)",
    },
    "stochastic transportation": {
        "supplies": "supplies[source] — base warehouse capacity",
        "recourse_capacities": "recourse_capacities[source] — emergency shipping capacity",
        "base_demands": "base_demands[destination] — mean demand (for context)",
        "base_costs": "base_costs[source][destination] — $/unit base shipping",
        "recourse_costs": "recourse_costs[source][destination] — $/unit emergency shipping",
        "demand_scenarios": "demand_scenarios[scenario][destination] — realized demand per scenario",
        "confidence_level": "fraction (e.g. 0.9 = 90% reliability)",
        "max_violations_per_dest": "max scenarios where demand may go unmet per destination",
    },
    "resource-constrained project scheduling": {
        "n_activities": "int — number of activities to schedule",
        "n_resources": "int — number of renewable resource types",
        "n_modes": "int — number of execution modes per activity",
        "resource_capacities": "resource_capacities[resource] — capacity of each resource",
        "mode_durations": "mode_durations[activity][mode] — duration of activity in each mode",
        "mode_resource_requirements": "mode_resource_requirements[activity][mode][resource] — resource usage",
        "mode_costs": "mode_costs[activity][mode] — cost of selecting each mode",
        "budget": "scalar — total budget for mode costs",
        "precedence_with_lags": "precedence_with_lags[edge][4] — each entry is [predecessor, successor, min_lag, max_lag] (0-based indices, -1 means no max lag)",
        "release_dates": "release_dates[activity] — earliest allowed start time (0 if none)",
        "deadlines": "deadlines[activity] — latest allowed finish time (large value means effectively no deadline)",
    },
    "multi-objective transportation": {
        "supplies": "supplies[source] — warehouse shipping capacity",
        "demands": "demands[destination] — store minimum demand",
        "cost_matrix": "cost_matrix[source][destination] — $/unit shipping cost",
        "emission_matrix": "emission_matrix[source][destination] — kg CO2/unit emissions",
        "fixed_costs": "fixed_costs[source][destination] — $ one-time route setup cost",
        "min_order_quantities": "min_order_quantities[source][destination] — minimum units if route is used",
        "max_suppliers_per_store": "int — max warehouses that can supply each store",
        "weight_cost": "scalar weight for cost objective (w_cost + w_emission = 1)",
        "weight_emission": "scalar weight for emission objective",
    },
}


def _describe_instance_data_schema(
    instance_data: Dict[str, Any],
    problem_type: str = "",
) -> str:
    """Auto-generate a concise schema description from instance_data structure.

    When problem_type is provided, uses _FIELD_ANNOTATIONS to add dimension
    labels and units so the LLM knows what each axis/value means.
    """
    annotations = _FIELD_ANNOTATIONS.get(problem_type, {})
    lines = []
    for key, val in instance_data.items():
        annotation = annotations.get(key, "")
        if isinstance(val, bool):
            base = f"bool = {val}"
        elif isinstance(val, (int, float)):
            base = f"number = {val}"
        elif isinstance(val, list):
            shape = _list_shape(val)
            dims = shape.replace("][", "×").replace("[", "").replace("]", "")
            base = f"list{shape}  — shape {dims}"
        else:
            base = f"value = {val}"

        if annotation:
            lines.append(f"  {key:<30s} : {base}  [{annotation}]")
        else:
            lines.append(f"  {key:<30s} : {base}")
    return "\n".join(lines)


_INSTANCE_DATA_CODE_REQUIREMENTS = textwrap.dedent(
    """
    Requirements:
      1. Return exactly one fenced ```python ... ``` block.
      2. Allowed imports: `json`, `math`, and `from gurobipy import *`. No others. No try/except.
      3. gurobipy IS available — do NOT create mock or fallback implementations.
      4. Define `solve_problem()` with no required arguments.
      5. Inside `solve_problem()`, open INSTANCE_DATA_PATH with json.load() to get `d`.
      6. Derive ALL decision variables, constraints, and objective from the fields in `d`.
         — For distances/costs not directly in `d`, compute them from coordinates.
         — Use meaningful variable names reflecting the problem structure.
      7. Do NOT hardcode any numeric coefficients, bounds, or RHS values from the data.
      8. Set `m.setParam("TimeLimit", 100)`.
      9. Call `m.optimize()` and return `m`.
    """
).strip()

_INSTANCE_DATA_CODE_TEMPLATE = textwrap.dedent(
    r"""
    ```python
    import json
    import math
    from gurobipy import *

    INSTANCE_DATA_PATH = r"<FILL_IN_PATH>"

    def solve_problem():
        with open(INSTANCE_DATA_PATH, "r") as f:
            d = json.load(f)

        m = Model("problem")
        m.setParam("TimeLimit", 100)

        # Variable types: GRB.CONTINUOUS, GRB.INTEGER, GRB.BINARY
        # Objective sense: GRB.MINIMIZE or GRB.MAXIMIZE

        # TODO: Define decision variables derived from d fields
        # e.g.: x = m.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, name="x")

        # TODO: Set objective
        # e.g.: m.setObjective(quicksum(c[i]*x[i] for i in range(n)), GRB.MINIMIZE)

        # TODO: Add constraints derived from d fields
        # e.g.: for i in range(n): m.addConstr(quicksum(...) <= rhs[i])

        m.optimize()
        return m
    ```
    """
).strip()


def build_instance_data_prompt(problem_data: Dict[str, Any], inst_path: Path) -> str:
    """
    Build the initial user prompt for instance-data offload mode.

    Includes:
      - The textual problem context (Situation, Mission, constraints description)
        extracted from LLM_description — no numeric values.
      - The path to the instance_data JSON file that holds all numeric parameters.
      - A schema of the instance_data fields so the LLM knows what keys to access.
      - Code requirements and a code template.
    """
    meta = problem_data.get("meta", {})
    problem_type = meta.get("problem_type", "optimization")
    goal = meta.get("goal", "")
    inst = problem_data.get("instance_data", {})
    schema = _describe_instance_data_schema(inst, problem_type=problem_type)
    context = build_context_from_template(problem_data)

    template_filled = _INSTANCE_DATA_CODE_TEMPLATE.replace("<FILL_IN_PATH>", str(inst_path))

    return textwrap.dedent(
        f"""
        Solve the following {problem_type} optimization problem using Gurobi.
        Goal: {goal}

        --- Problem Description ---
        {context}

        --- Numeric Data ---
        All numeric data is stored in a local JSON file — do NOT embed values in code.

        INSTANCE_DATA_PATH = r"{inst_path}"

        Instance data fields (accessible via json.load, then d["key"]):
        {schema}

        {_INSTANCE_DATA_CODE_REQUIREMENTS}

        Use this template as your starting point:
        {template_filled}
        """
    ).strip()


def build_array_prompt(problem_data: Dict[str, Any], array_path: Path) -> str:
    """
    Build the user prompt for array-offload mode (abstract/resource-allocation problems).
    Contains NO numeric data — only metadata, file path, schema, and a code template.
    """
    meta = problem_data.get("meta", {})
    problem_type = meta.get("problem_type", "optimization")
    goal = meta.get("goal", "")
    num_vars = meta.get("num_vars", "?")
    num_constraints = meta.get("num_constraints", "?")

    template_filled = _CODE_TEMPLATE.replace("<FILL_IN_PATH>", str(array_path))

    return textwrap.dedent(
        f"""
        Solve a {problem_type} optimization problem using Gurobi.
        All numeric data is pre-serialized in a local array file — do NOT embed values.

        PROBLEM_ARRAY_PATH = r"{array_path}"

        Problem metadata:
          - goal            : {goal}
          - num_vars        : {num_vars}
          - num_constraints : {num_constraints}

        {_ARRAY_SCHEMA_NOTE}

        {_CODE_REQUIREMENTS}

        Use this template as your starting point (fill in the path, adjust as needed):
        {template_filled}
        """
    ).strip()


# ---------------------------------------------------------------------------
# Chunked extract-then-solve (for problems without instance_data)
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = (
    "You are a precise data extraction assistant. "
    "Extract numeric parameters from text into structured JSON. "
    "Return ONLY a fenced ```json ... ``` block."
)


def build_variable_extraction_prompt(problem_data: Dict[str, Any]) -> str:
    """
    Phase 1a: Extract the optimization goal and ALL decision variables
    (name, type, bounds, objective coefficient) from the full description.
    Output is small even for 20 variables (~1-2KB JSON).
    """
    llm_description = problem_data.get("LLM_description", "")
    return textwrap.dedent(
        f"""
        Read the problem description below and extract:
        1. The optimization goal (MINIMIZE or MAXIMIZE)
        2. ALL decision variables with their properties

        --- Problem Description ---
        {llm_description}

        Return a JSON object:
        ```json
        {{
          "goal": "MINIMIZE" or "MAXIMIZE",
          "variables": [
            {{
              "name": "<short descriptive name>",
              "type": "Continuous" | "Integer" | "Binary",
              "lb": <lower bound (number)>,
              "ub": <upper bound (number or null)>,
              "objective_coefficient": <cost/profit per unit (number)>
            }}
          ]
        }}
        ```

        Rules:
        - Use short, descriptive variable names from the problem context.
        - Extract exact numeric values — do NOT round or approximate.
        - Include ALL variables mentioned. Do NOT include constraints.
        """
    ).strip()


def chunk_description(text: str, max_chars: int = 2000) -> List[str]:
    """
    Split text into chunks at paragraph boundaries (double newline).
    Each chunk is at most *max_chars* characters.
    If a single paragraph exceeds the limit, split at sentence boundaries.
    """
    import re as _re

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    def _flush():
        nonlocal current_parts, current_len
        if current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_len = 0

    for para in paragraphs:
        # If adding this paragraph would exceed the limit, flush first
        if current_len + len(para) + 2 > max_chars and current_parts:
            _flush()

        # Single paragraph too long → split at sentence boundaries
        if len(para) > max_chars:
            sentences = _re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if current_len + len(sent) + 1 > max_chars and current_parts:
                    _flush()
                current_parts.append(sent)
                current_len += len(sent) + 1
        else:
            current_parts.append(para)
            current_len += len(para) + 2

    _flush()
    return chunks


def build_chunk_contribution_prompt(
    chunk_text: str,
    variable_names: List[str],
) -> str:
    """
    Phase 1b: For a text chunk, extract each variable's per-unit resource
    usage / contributions AND any resource limits (constraint RHS values).

    Resource allocation descriptions typically embed coefficients in variable
    paragraphs ("uses 3.22 hours of Blending Line A per unit") while constraint
    limits appear in a separate section ("Blending Line A hours: 189.54").
    This prompt captures both.
    """
    var_list = ", ".join(variable_names)
    return textwrap.dedent(
        f"""
        Context: an optimization problem with these variables:
        {var_list}

        Read the text passage below and extract TWO things:

        1. **contributions**: For each variable mentioned, list its per-unit
           resource usage / contribution toward shared resources or constraints.
           These are the CONSTRAINT COEFFICIENTS — how much of each shared
           resource one unit of the variable consumes or contributes.
           (e.g., "contributes 2.41 units toward Field Hours" → coefficient 2.41)
           Do NOT include the variable's objective cost here — only resource usage.
        2. **limits**: Any resource capacity limits / requirements mentioned
           (e.g., "Resource_A hours: at most 189.54" or "must be at least 22.5").

        --- Text Passage ---
        {chunk_text}

        Return a JSON object:
        ```json
        {{
          "contributions": [
            {{
              "variable": "<variable_name from known list>",
              "resource": "<resource name>",
              "coefficient": <per-unit amount (number)>
            }}
          ],
          "limits": [
            {{
              "resource": "<resource name>",
              "sense": "<=" | ">=" | "==",
              "rhs": <limit value (number)>
            }}
          ]
        }}
        ```

        Rules:
        - Only use variable names from the known list above.
        - If this passage has no contributions or limits, return empty lists.
        - Extract exact numeric values — do NOT round or approximate.
        - Use consistent resource names across contributions and limits.
        """
    ).strip()


def merge_chunked_extractions(
    goal: str,
    variables: List[Dict[str, Any]],
    all_contributions: List[Dict[str, Any]],
    all_limits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Assemble the final extraction by combining:
    - variables (from Phase 1a)
    - per-variable contributions → constraint coefficients
    - resource limits → constraint RHS values

    Matches contributions to limits by resource name to build full constraints.
    Deduplicates limits and filters out variable-bound limits.
    """
    from collections import defaultdict

    var_names_lower = {v["name"].lower().replace(" ", "_") for v in variables}

    # Group contributions by normalised resource name
    def _norm(name: str) -> str:
        return name.lower().strip().replace("_", " ").replace("-", " ")

    resource_coeffs: Dict[str, Dict[str, float]] = defaultdict(dict)
    # Map normalised name → canonical name (first seen)
    canonical_name: Dict[str, str] = {}
    for c in all_contributions:
        nk = _norm(c["resource"])
        if nk not in canonical_name:
            canonical_name[nk] = c["resource"]
        resource_coeffs[nk][c["variable"]] = c["coefficient"]

    # Deduplicate limits: keep first limit per normalised resource name.
    # Filter out limits that are actually variable bounds.
    seen_limits: Dict[str, Dict[str, Any]] = {}
    for lim in all_limits:
        nk = _norm(lim["resource"])
        # Skip variable-bound limits (resource name matches a variable name)
        if nk.replace(" ", "_") in var_names_lower or nk.replace(" ", "") in var_names_lower:
            continue
        if nk not in seen_limits:
            seen_limits[nk] = lim

    # Build constraints: match limits to contributions by normalised name.
    # Only keep constraints that have at least one variable coefficient.
    constraints = []
    matched_nk = set()
    for nk, lim in seen_limits.items():
        coeffs = resource_coeffs.get(nk, {})
        if coeffs:
            name = canonical_name.get(nk, lim["resource"])
            constraints.append({
                "name": name,
                "sense": lim["sense"],
                "rhs": lim["rhs"],
                "coefficients": coeffs,
            })
            matched_nk.add(nk)

    # Resources with contributions but no matching limit → warn but skip
    # (no RHS means we can't form a valid constraint).

    return {
        "goal": goal,
        "variables": variables,
        "constraints": constraints,
    }


# Keep the old single-shot extraction prompt as a fallback / for the repair loop
def build_extraction_prompt(problem_data: Dict[str, Any]) -> str:
    """
    Single-shot extraction prompt (used in repair loop and as fallback).
    Extracts everything at once — works for small problems but may fail on large ones.
    """
    llm_description = problem_data.get("LLM_description", "")
    return textwrap.dedent(
        f"""
        Read the following optimization problem description and extract all decision
        variables, constraints, and objective function parameters into structured JSON.

        --- Problem Description ---
        {llm_description}

        Extract ALL parameters into the following JSON schema:
        ```json
        {{
          "goal": "MINIMIZE" or "MAXIMIZE",
          "variables": [
            {{
              "name": "<variable name as described>",
              "type": "Continuous" | "Integer" | "Binary",
              "lb": <lower bound (number)>,
              "ub": <upper bound (number or null)>,
              "objective_coefficient": <cost/profit per unit (number)>
            }}
          ],
          "constraints": [
            {{
              "name": "<constraint name or short description>",
              "sense": "<=" | ">=" | "==",
              "rhs": <right-hand side value (number)>,
              "coefficients": {{
                "<variable name>": <coefficient (number)>
              }}
            }}
          ]
        }}
        ```

        Rules:
        - Use short descriptive variable names from the problem context.
        - Include ALL constraints. Do NOT include variable bounds as constraints.
        - Extract exact numeric values — do NOT round or approximate.
        """
    ).strip()


_EXTRACTED_ARRAY_SCHEMA_NOTE = textwrap.dedent(
    """
    Extracted data file schema (JSON):
      goal        : str — "MINIMIZE" or "MAXIMIZE"
      variables   : list of objects, each with:
                      name, type (Continuous/Integer/Binary), lb, ub, objective_coefficient
      constraints : list of objects, each with:
                      name, sense (<=/>=/==), rhs, coefficients (dict: var_name → number)
    """
).strip()

_SOLVE_FROM_EXTRACTED_REQUIREMENTS = textwrap.dedent(
    """
    Requirements:
      1. Return exactly one fenced ```python ... ``` block.
      2. ONLY import `json` and `from gurobipy import *`. No other imports. No try/except.
      3. gurobipy IS available — do NOT create mock or fallback implementations.
      4. Define `solve_problem()` with no required arguments.
      5. Inside `solve_problem()`, open and parse EXTRACTED_DATA_PATH with json.load().
      6. Build Gurobi variables from the "variables" list (name, type, lb, ub).
         Use GRB.INFINITY when ub is null.
      7. Build the objective from each variable's objective_coefficient and the goal.
      8. Build all constraints from the "constraints" list using coefficients, sense, rhs.
      9. Set `m.setParam("TimeLimit", 100)`.
     10. Call `m.optimize()` and return `m`.
     11. Do NOT hardcode any numeric coefficients, bounds, or RHS values.
    """
).strip()

_SOLVE_FROM_EXTRACTED_TEMPLATE = textwrap.dedent(
    r"""
    ```python
    import json
    from gurobipy import *

    EXTRACTED_DATA_PATH = r"<FILL_IN_PATH>"

    def solve_problem():
        with open(EXTRACTED_DATA_PATH, "r") as f:
            d = json.load(f)

        m = Model("problem")
        m.setParam("TimeLimit", 100)

        type_map = {"Continuous": GRB.CONTINUOUS, "Integer": GRB.INTEGER, "Binary": GRB.BINARY}

        # Build variables
        vars_map = {}
        for v in d["variables"]:
            ub = v["ub"] if v["ub"] is not None else GRB.INFINITY
            vars_map[v["name"]] = m.addVar(
                lb=v["lb"], ub=ub,
                vtype=type_map.get(v["type"], GRB.CONTINUOUS),
                name=v["name"]
            )
        m.update()

        # Objective
        obj = quicksum(v["objective_coefficient"] * vars_map[v["name"]] for v in d["variables"])
        sense = GRB.MINIMIZE if d["goal"] == "MINIMIZE" else GRB.MAXIMIZE
        m.setObjective(obj, sense)

        # Constraints
        for c in d["constraints"]:
            expr = quicksum(coeff * vars_map[var] for var, coeff in c["coefficients"].items())
            if c["sense"] == "<=":
                m.addConstr(expr <= c["rhs"], name=c.get("name", ""))
            elif c["sense"] == ">=":
                m.addConstr(expr >= c["rhs"], name=c.get("name", ""))
            else:
                m.addConstr(expr == c["rhs"], name=c.get("name", ""))

        m.optimize()
        return m
    ```
    """
).strip()


def build_solve_from_extracted_prompt(extracted_path: Path) -> str:
    """
    Build the Phase-2 prompt: write solver code that loads extracted structured data.
    """
    template_filled = _SOLVE_FROM_EXTRACTED_TEMPLATE.replace(
        "<FILL_IN_PATH>", str(extracted_path)
    )
    return textwrap.dedent(
        f"""
        Write Python code that solves an optimization model by loading locally extracted data.

        EXTRACTED_DATA_PATH = r"{extracted_path}"

        {_EXTRACTED_ARRAY_SCHEMA_NOTE}

        {_SOLVE_FROM_EXTRACTED_REQUIREMENTS}

        Use this template as your starting point:
        {template_filled}
        """
    ).strip()


def build_extraction_repair_prompt(
    eval_result: Any,
    extracted_path: Path,
    original_description: str,
) -> str:
    """
    Repair prompt for the two-phase approach. When the solver fails, it may be
    because the extraction was wrong. Re-show the original description and ask
    the LLM to re-extract AND re-solve.
    """
    stderr = (eval_result.generated_solution.stderr or "").strip()
    stderr_tail = stderr[-1600:] if stderr else "(empty)"

    return textwrap.dedent(
        f"""
        The solver code failed evaluation. The extraction or the solver may be wrong.

        Evaluation signals:
          - execution_succeeded : {eval_result.execution_succeeded}
          - status_optimal      : {eval_result.status_optimal}
          - objective_matches   : {eval_result.objective_matches}
          - objective_error     : {eval_result.objective_error}

        stderr (last 1600 chars):
        {stderr_tail}

        Original problem description (for reference):
        {original_description}

        Please provide TWO fenced blocks in this exact order:

        1. A corrected ```json ... ``` block with the extracted data (same schema as before).
        2. A corrected ```python ... ``` block with solve_problem() that loads from:
           EXTRACTED_DATA_PATH = r"{extracted_path}"

        Common issues to check:
        - Did you extract ALL constraints (not just variable bounds)?
        - Are constraint coefficients correct (which variable contributes to which constraint)?
        - Is the optimization direction (MINIMIZE/MAXIMIZE) correct?
        - Are variable types (Continuous/Integer/Binary) correct?
        """
    ).strip()


def parse_extraction_json(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from the LLM response. Tries fenced blocks first, then raw JSON."""
    import re
    # Try fenced ```json ... ``` block
    match = re.search(r"```json\s*\n(.*?)```", response, re.DOTALL)
    if not match:
        # Try without the json language tag
        match = re.search(r"```\s*\n(\{.*?\})\s*\n```", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try parsing the entire response as raw JSON (for JSON mode responses)
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        return None


def parse_solver_code(response: str) -> Optional[str]:
    """Parse a ```python ... ``` block from the LLM response."""
    import re
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return "```python\n" + match.group(1).strip() + "\n```"
    return None


def build_agentic_prompt(problem_data: Dict[str, Any], data_path: Path) -> str:
    """
    Auto-routing entry point: choose instance_data or array prompt based on problem.

    - Template problems (non-empty instance_data) → instance_data offload prompt.
    - Abstract problems (no instance_data) → two-phase extract prompt (Phase 1).
    """
    if problem_data.get("instance_data"):
        return build_instance_data_prompt(problem_data, data_path)
    # For problems without instance_data, Phase 1 is the extraction prompt.
    # The caller (run_eval) will handle the two-phase orchestration.
    return build_extraction_prompt(problem_data)


def _build_structural_comparison(eval_result: Any, problem_data: dict = None) -> str:
    """Build structured comparison block for repair prompts."""
    gen_sol = eval_result.generated_solution
    hints = []

    # Objective comparison
    if gen_sol.objective_value is not None and eval_result.reference_optimum is not None:
        gen_obj = gen_sol.objective_value
        ref_obj = eval_result.reference_optimum
        if ref_obj != 0:
            pct = (gen_obj - ref_obj) / abs(ref_obj) * 100
            direction = "too high" if pct > 0 else "too low"
            hints.append(
                f"  - Your objective value : {gen_obj:.6g}\n"
                f"  - Expected objective   : {ref_obj:.6g} (yours is {abs(pct):.1f}% {direction})"
            )
        else:
            hints.append(
                f"  - Your objective value : {gen_obj:.6g}\n"
                f"  - Expected objective   : {ref_obj:.6g}"
            )
    elif eval_result.reference_optimum is not None:
        hints.append(
            f"  - Your objective value : (not available)\n"
            f"  - Expected objective   : {eval_result.reference_optimum:.6g}"
        )

    # Status comparison
    _STATUS_NAMES = {2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED"}
    gen_status = gen_sol.status
    ref_status = None
    if problem_data:
        rb = problem_data.get("gurobi_result", {})
        ref_status = rb.get("solver_status", problem_data.get("meta", {}).get("solver_status"))
    gen_s = _STATUS_NAMES.get(gen_status, str(gen_status))
    ref_s = _STATUS_NAMES.get(ref_status, str(ref_status))
    if gen_status != ref_status:
        hints.append(
            f"  - Your solver status   : {gen_s} (code {gen_status})\n"
            f"  - Expected status      : {ref_s} (code {ref_status})"
        )

    # Variable/constraint count comparison
    if problem_data and getattr(gen_sol, 'num_vars', None) is not None:
        meta = problem_data.get("meta", {})
        ref_nv = meta.get("num_vars")
        ref_nc = meta.get("num_constraints")
        gen_nv = gen_sol.num_vars
        gen_nc = gen_sol.num_constrs

        var_match = "✓" if gen_nv == ref_nv else "MISMATCH"
        con_match = "✓" if gen_nc == ref_nc else "MISMATCH"
        hints.append(
            f"  - Your model structure : {gen_nv} variables, {gen_nc} constraints\n"
            f"  - Expected structure   : {ref_nv} variables, {ref_nc} constraints "
            f"[vars: {var_match}, constrs: {con_match}]"
        )

    return "\n".join(hints)


def build_repair_prompt(eval_result: Any, data_path: Optional[Path] = None,
                        problem_data: dict = None) -> str:
    """
    Build a repair prompt from a failed EvaluationResult with structured feedback.

    Args:
        eval_result: The EvaluationResult from the failed attempt.
        data_path: Optional path to the data file (instance_data or array).
        problem_data: Optional problem JSON dict for structural comparison.
    """
    stderr = (eval_result.generated_solution.stderr or "").strip()
    stderr_tail = stderr[-1600:] if stderr else "(empty)"

    path_reminder = ""
    if data_path is not None:
        path_reminder = (
            f"\n  The data file path has not changed — use exactly this hardcoded path:\n"
            f"  INSTANCE_DATA_PATH = r\"{data_path}\"  (or PROBLEM_ARRAY_PATH for array mode)\n"
        )

    structural_block = _build_structural_comparison(eval_result, problem_data)

    return textwrap.dedent(
        f"""
        The previous code did not pass evaluation. Return a complete fixed replacement.

        Evaluation signals:
          - execution_succeeded : {eval_result.execution_succeeded}
          - status_optimal      : {eval_result.status_optimal}
          - objective_matches   : {eval_result.objective_matches}

        Detailed comparison:
        {structural_block}

        stderr (last 1600 chars):
        {stderr_tail}
        {path_reminder}
        Diagnosis hints:
          - If your objective is wrong but status is OPTIMAL, you likely have a modeling
            error: wrong coefficients, missing/extra terms, or misread problem data.
          - If variable/constraint counts differ, you may be missing decision variables
            or constraints described in the problem.
          - If status is INFEASIBLE, check constraint directions (<=, >=, ==) and bounds.
          - Re-read the original problem description carefully and compare with your code.

        Requirements:
          - Load data from the hardcoded file path at the top of the script.
          - No try/except blocks. No mock implementations.
          - Use correct gurobipy constants: GRB.CONTINUOUS, GRB.INTEGER, GRB.BINARY,
            GRB.MINIMIZE, GRB.MAXIMIZE (NOT GRB_CONTINUOUS, GRB_INTEGER, etc.)
          - Return the Gurobi model object m from solve_problem().
        """
    ).strip()


# ---------------------------------------------------------------------------
# Direct-solve + repair (for problems without instance_data)
# ---------------------------------------------------------------------------

def build_direct_solve_repair_prompt(eval_result: Any, problem_data: dict = None) -> str:
    """
    Repair prompt for direct-solve mode with structured feedback.
    The original problem description is already in conversation history (round 1).
    """
    stderr = (eval_result.generated_solution.stderr or "").strip()
    stderr_tail = stderr[-1600:] if stderr else "(empty)"

    structural_block = _build_structural_comparison(eval_result, problem_data)

    return textwrap.dedent(
        f"""
        The previous code did not pass evaluation. Return a complete fixed replacement.

        Evaluation signals:
          - execution_succeeded : {eval_result.execution_succeeded}
          - status_optimal      : {eval_result.status_optimal}
          - objective_matches   : {eval_result.objective_matches}

        Detailed comparison:
        {structural_block}

        stderr (last 1600 chars):
        {stderr_tail}

        Diagnosis hints:
          - If your objective is wrong but status is OPTIMAL, you likely have a modeling
            error: wrong coefficients, missing/extra terms, or misread problem data.
          - If variable/constraint counts differ, you may be missing decision variables
            or constraints described in the problem.
          - If status is INFEASIBLE, check constraint directions (<=, >=, ==) and bounds.
          - Re-read the original problem description carefully and compare with your code.

        Requirements:
          - Do not import any libraries other than `gurobipy`.
          - Use correct gurobipy constants: GRB.CONTINUOUS, GRB.INTEGER, GRB.BINARY,
            GRB.MINIMIZE, GRB.MAXIMIZE.
          - Return the Gurobi model object `m` from `solve_problem()`.
          - Set `m.setParam("TimeLimit", 100)` before `m.optimize()`.
          - Return exactly one fenced ```python ... ``` block.
        """
    ).strip()


# ---------------------------------------------------------------------------
# Self-review agentic approach (no evaluator feedback needed)
# ---------------------------------------------------------------------------

def build_self_review_prompt(code: str) -> str:
    """
    Self-review prompt: ask the LLM to verify its own code against
    the problem description (already in conversation history).
    No evaluator feedback needed — purely LLM self-correction.
    """
    return textwrap.dedent(
        f"""
        Review the code you just wrote against the original problem description.

        Your code:
        {code}

        Perform these checks carefully:
        1. **Variables**: Is every decision variable from the description present?
           Are types (Continuous/Integer/Binary) correct? Are bounds correct?
        2. **Objective**: Is the direction (MINIMIZE/MAXIMIZE) correct?
           Are ALL objective coefficients exact?
        3. **Constraints**: Go through each constraint in the description one-by-one.
           Is it present in the code? Are the coefficients correct? Is the sense
           (<=, >=, ==) correct? Is the RHS value exact?
        4. **Missing constraints**: Are there any constraints in the description
           that are NOT in the code?

        If you find ANY errors, return a corrected ```python ... ``` block.
        If the code is correct, return it unchanged in a ```python ... ``` block.
        Always return exactly one fenced ```python ... ``` block.
        """
    ).strip()


# ---------------------------------------------------------------------------
# Structured JSON extraction (single-shot, improved prompt)
# ---------------------------------------------------------------------------

_STRUCTURED_EXTRACTION_PROMPT = textwrap.dedent(
    """
    Read the optimization problem description below and extract its COMPLETE
    mathematical formulation into a structured JSON.

    --- Problem Description ---
    {LLM_DESCRIPTION}

    Extract into this EXACT JSON schema (return ONLY valid JSON, no markdown fences):
    {{
      "goal": "MINIMIZE" or "MAXIMIZE",
      "variables": [
        {{
          "name": "<short name from context>",
          "type": "Continuous" | "Integer" | "Binary",
          "lb": <lower bound>,
          "ub": <upper bound or null>,
          "objective_coefficient": <number>
        }}
      ],
      "constraints": [
        {{
          "name": "<short description>",
          "sense": "<=" | ">=" | "==",
          "rhs": <number>,
          "coefficients": {{
            "<variable name>": <coefficient>
          }}
        }}
      ]
    }}

    CRITICAL rules:
    - Include ALL variables. Use the EXACT names you chose consistently.
    - Include ALL constraints. Do NOT include variable bounds as constraints.
    - Extract EXACT numeric values — do NOT round.
    - Each constraint's "coefficients" dict maps variable names to their
      per-unit contribution toward that constraint.
    - If a variable does not appear in a constraint, omit it from coefficients.
    """
).strip()


def build_structured_extraction_prompt(problem_data: Dict[str, Any]) -> str:
    """Build the structured JSON extraction prompt."""
    llm_description = problem_data.get("LLM_description", "")
    return _STRUCTURED_EXTRACTION_PROMPT.format(LLM_DESCRIPTION=llm_description)
