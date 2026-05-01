#!/usr/bin/env python3
"""
Evaluators for optimization problems stored in JSON.
- Execute embedded or generated Gurobi code safely.
- Compare results to JSON reference meta (objective, variables).
- Used for validating gold code and scoring LLM-generated solutions.
"""

import os
import re
import subprocess
import tempfile
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import sys

from main.utils import load_json

@dataclass
class ExecutionResult:
    """Results from executing generated code"""
    exec_success: bool
    status: Optional[int]  # Gurobi status code (int)
    objective_value: Optional[float]
    variable_values: Dict[str, float]
    stdout: str
    stderr: str
    execution_time: float
    exit_code: int
    num_vars: Optional[int] = None
    num_constrs: Optional[int] = None
    num_bin_vars: Optional[int] = None
    num_int_vars: Optional[int] = None

@dataclass
class EvaluationResult:
    """Complete evaluation results for a problem"""
    problem_id: str
    execution_succeeded: bool
    status_optimal: bool
    objective_error: Optional[float]
    objective_matches: bool
    execution_time: float
    reference_optimum: float
    generated_solution: ExecutionResult

class CodeSolutionEvaluator:
    def __init__(self):
        # Relative tolerance for objective comparison (0.01% of the reference value)
        # This is more appropriate than absolute tolerance for problems with varying scales
        self.tolerance_rel = 1e-5
        # Absolute tolerance floor for very small objectives
        self.tolerance_abs = 1e-6
        
        self.time_limit = 510  # seconds (Gurobi gets 500s, 10s buffer for subprocess)
    
    def objectives_match(self, computed: float, reference: float) -> bool:
        """
        Compare objectives using relative tolerance with absolute floor.
        
        More robust than pure absolute tolerance for:
        - Large objectives (fixed costs + transport costs can be thousands)
        - Floating-point operations involving sqrt, multiplication chains
        """
        return math.isclose(computed, reference, rel_tol=self.tolerance_rel, abs_tol=self.tolerance_abs)
        
    def load_problem_json(self, json_path: str) -> Dict[str, Any]:
        """Load a problem from a JSON file"""
        return load_json(json_path)

    def extract_python_code(self, model_output: str) -> Optional[str]:
        """Extract Python code from model output using regex"""
        # Strip leading/trailing whitespace from the entire input
        model_output = model_output.strip()
        
        # Look for Python code blocks - more flexible patterns
        python_patterns = [
            r'```python\s*\n(.*?)\n\s*```',      # standard with optional whitespace before closing
            r'```python\s*\n(.*?)```',            # no newline before closing backticks
            r'```gurobi\s*\n(.*?)\n\s*```',
            r'```gurobi\s*\n(.*?)```',
            r'```py\s*\n(.*?)\n\s*```',
            r'```py\s*\n(.*?)```',
            r'```\s*\n(.*?)\n\s*```',             # bare code block without language tag
        ]
        
        for pattern in python_patterns:
            matches = re.findall(pattern, model_output, re.DOTALL)
            if matches:
                # Return the last (most complete) Python block
                code = matches[-1].strip()
                # Strip any nested markdown fences (e.g. double-wrapped ```python...```)
                code = re.sub(r'^```(?:python|py|gurobi)?\s*\n', '', code)
                code = re.sub(r'\n\s*```\s*$', '', code)
                code = self._ensure_function_call(code)
                return code
        
        # Handle vLLM output where opening fence is stripped but closing remains
        # e.g. "python\nimport json\n...\n```"
        cleaned = model_output.strip()
        if cleaned.startswith("python\n"):
            cleaned = cleaned[len("python\n"):]
        # Strip trailing markdown fences
        cleaned = re.sub(r'\n\s*```\s*$', '', cleaned)

        # If the string itself looks like code (no markdown blocks), try it directly
        if "from gurobipy import" in cleaned or "import gurobipy" in cleaned:
             return self._ensure_function_call(cleaned.strip())

        return None
    
    def _ensure_function_call(self, code: str) -> str:
        """
        Ensure that if code defines a function, it also calls it.
        """
        # Check if there's already a function call or main guard
        if 'if __name__ == "__main__":' in code or '__main__' in code:
            return code
        
        # Look for the specific solve_problem definition we expect
        function_match = re.search(r'def\s+(\w+)\([^)]*\):', code)
        if function_match:
            function_name = function_match.group(1)
            # Check if function is called outside of def
            # Simple check: count occurrences of "function_name("
            # If only 1 (the def), then it needs a call.
            if code.count(f"{function_name}(") <= 1:
                 return f"{code}\n\n# Auto-generated function call\nif __name__ == \"__main__\":\n    {function_name}()"
        
        return code
    
    def _prepare_code_for_execution(self, python_code: str) -> str:
        """
        Modify code to capture the returned model object and print its status/results
        in a format we can parse.
        """
        # Override any Gurobi TimeLimit the LLM sets to match our subprocess timeout
        gurobi_timelimit = int(self.time_limit - 10)
        python_code = re.sub(
            r'(\.setParam\s*\(\s*["\']TimeLimit["\']\s*,\s*)\d+',
            rf'\g<1>{gurobi_timelimit}',
            python_code
        )
        # We wrap the existing code or execution logic to ensure we print specific tags
        
        # Find where the function is called
        if 'if __name__ == "__main__":' in python_code:
            # We will append our own extraction logic instead of trusting the user's print statements
            # But we need to capture the return value of the function call.
            # This is tricky with string manipulation. 
            # Strategy: Replace the main block with our own.
            base_code = python_code.split('if __name__ == "__main__":')[0]
            
            # Find the function name again to call it
            function_match = re.search(r'def\s+(\w+)\([^)]*\):', base_code)
            if function_match:
                func_name = function_match.group(1)
                
                extraction_code = f"""
if __name__ == "__main__":
    try:
        model = {func_name}()
        
        if model:
            print(f"_MODEL_STATUS_: {{model.status}}")
            print(f"_MODEL_NUMVARS_: {{model.NumVars}}")
            print(f"_MODEL_NUMCONSTRS_: {{model.NumConstrs}}")
            print(f"_MODEL_NUMBINVARS_: {{model.NumBinVars}}")
            print(f"_MODEL_NUMINTVARS_: {{model.NumIntVars}}")

            if model.status == 2:  # GRB.OPTIMAL
                print(f"_MODEL_OBJ_: {{model.objVal}}")
                for v in model.getVars():
                    try:
                        print(f"_MODEL_VAR_ {{v.VarName}} = {{v.X}}")
                    except:
                        pass
            else:
                print("_MODEL_OBJ_: NA")
        else:
             print("ERROR: Function did not return a model object")

    except Exception as e:
        print(f"ERROR: {{e}}")
"""
                return base_code + extraction_code
            else:
                # No function found in base_code - this is a bug indicator
                print(f"WARNING: No function found in base_code. Code starts with: {base_code[:200]}")

        # No main block found - add one with extraction code
        # This handles cases where _ensure_function_call didn't add a main block
        function_match = re.search(r'def\s+(\w+)\([^)]*\):', python_code)
        if function_match:
            func_name = function_match.group(1)
            extraction_code = f"""

if __name__ == "__main__":
    try:
        model = {func_name}()

        if model:
            print(f"_MODEL_STATUS_: {{model.status}}")
            print(f"_MODEL_NUMVARS_: {{model.NumVars}}")
            print(f"_MODEL_NUMCONSTRS_: {{model.NumConstrs}}")
            print(f"_MODEL_NUMBINVARS_: {{model.NumBinVars}}")
            print(f"_MODEL_NUMINTVARS_: {{model.NumIntVars}}")

            if model.status == 2:  # GRB.OPTIMAL
                print(f"_MODEL_OBJ_: {{model.objVal}}")
                for v in model.getVars():
                    try:
                        print(f"_MODEL_VAR_ {{v.VarName}} = {{v.X}}")
                    except:
                        pass
            else:
                print("_MODEL_OBJ_: NA")
        else:
             print("ERROR: Function did not return a model object")

    except Exception as e:
        print(f"ERROR: {{e}}")
"""
            return python_code + extraction_code
        
        return python_code

    def execute_code_safely(self, python_code: str) -> ExecutionResult:
        """Execute Python code in a sandboxed environment"""
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare code
            modified_code = self._prepare_code_for_execution(python_code)
            
            code_file = os.path.join(temp_dir, "code_exec.py")
            with open(code_file, 'w') as f:
                f.write(modified_code)
            
            env = os.environ.copy()

            # Use python3 with gurobipy installed.
            # Fall back to sys.executable if not found.
            _GUROBI_PYTHON = "python3"
            _python = _GUROBI_PYTHON if os.path.isfile(_GUROBI_PYTHON) else sys.executable

            try:
                result = subprocess.run(
                    [_python, code_file],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.time_limit,
                    env=env
                )
                
                execution_time = time.time() - start_time
                status, obj_value, var_values, nv, nc, nbv, niv = self._parse_output(result.stdout)

                return ExecutionResult(
                    exec_success=result.returncode == 0,
                    status=status,
                    objective_value=obj_value,
                    variable_values=var_values,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    execution_time=execution_time,
                    exit_code=result.returncode,
                    num_vars=nv,
                    num_constrs=nc,
                    num_bin_vars=nbv,
                    num_int_vars=niv,
                )
                
            except subprocess.TimeoutExpired:
                execution_time = time.time() - start_time
                return ExecutionResult(
                    exec_success=False,
                    status="TIME_LIMIT",
                    objective_value=None,
                    variable_values={},
                    stdout="",
                    stderr="Execution timed out",
                    execution_time=execution_time,
                    exit_code=-1
                )
            except Exception as e:
                return ExecutionResult(
                    exec_success=False,
                    status="ERROR",
                    objective_value=None,
                    variable_values={},
                    stdout="",
                    stderr=str(e),
                    execution_time=time.time() - start_time,
                    exit_code=-1
                )

    async def execute_code_safely_async(self, python_code: str) -> ExecutionResult:
        """Async version of execute_code_safely — does not block the event loop."""
        import asyncio
        start_time = time.time()

        with tempfile.TemporaryDirectory() as temp_dir:
            modified_code = self._prepare_code_for_execution(python_code)
            code_file = os.path.join(temp_dir, "code_exec.py")
            with open(code_file, 'w') as f:
                f.write(modified_code)

            env = os.environ.copy()
            _GUROBI_PYTHON = "python3"
            _python = _GUROBI_PYTHON if os.path.isfile(_GUROBI_PYTHON) else sys.executable

            try:
                proc = await asyncio.create_subprocess_exec(
                    _python, code_file,
                    cwd=temp_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        proc.communicate(), timeout=self.time_limit
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    return ExecutionResult(
                        exec_success=False, status="TIME_LIMIT",
                        objective_value=None, variable_values={},
                        stdout="", stderr="Execution timed out",
                        execution_time=time.time() - start_time, exit_code=-1,
                    )

                stdout_str = stdout_bytes.decode(errors="replace")
                stderr_str = stderr_bytes.decode(errors="replace")
                execution_time = time.time() - start_time
                status, obj_value, var_values, nv, nc, nbv, niv = self._parse_output(stdout_str)

                return ExecutionResult(
                    exec_success=(proc.returncode == 0),
                    status=status, objective_value=obj_value,
                    variable_values=var_values,
                    stdout=stdout_str, stderr=stderr_str,
                    execution_time=execution_time,
                    exit_code=proc.returncode,
                    num_vars=nv, num_constrs=nc,
                    num_bin_vars=nbv, num_int_vars=niv,
                )
            except Exception as e:
                return ExecutionResult(
                    exec_success=False, status="ERROR",
                    objective_value=None, variable_values={},
                    stdout="", stderr=str(e),
                    execution_time=time.time() - start_time, exit_code=-1,
                )

    def _parse_output(self, stdout: str) -> Tuple[str, Optional[float], Dict[str, float],
                                                   Optional[int], Optional[int],
                                                   Optional[int], Optional[int]]:
        """Parse the standardized output format"""
        status = None
        obj_value = None
        var_values = {}
        num_vars = None
        num_constrs = None
        num_bin_vars = None
        num_int_vars = None

        lines = stdout.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith("_MODEL_STATUS_:"):
                status_str = line.split(":", 1)[1].strip()
                try:
                    status = int(status_str)
                except ValueError:
                    status = None
            elif line.startswith("_MODEL_NUMVARS_:"):
                try:
                    num_vars = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("_MODEL_NUMCONSTRS_:"):
                try:
                    num_constrs = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("_MODEL_NUMBINVARS_:"):
                try:
                    num_bin_vars = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("_MODEL_NUMINTVARS_:"):
                try:
                    num_int_vars = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("_MODEL_OBJ_:"):
                obj_str = line.split(":", 1)[1].strip()
                if obj_str != "NA":
                    try:
                        obj_value = float(obj_str)
                    except ValueError:
                        pass
            elif line.startswith("_MODEL_VAR_"):
                # Format: _MODEL_VAR_ x0 = 1.23
                parts = line.replace("_MODEL_VAR_", "").split("=", 1)
                if len(parts) == 2:
                    var_name = parts[0].strip()
                    try:
                        var_value = float(parts[1].strip())
                        var_values[var_name] = var_value
                    except ValueError:
                        pass
        
        return status, obj_value, var_values, num_vars, num_constrs, num_bin_vars, num_int_vars


class GeneratedSolutionEvaluator(CodeSolutionEvaluator):
    """
    Evaluates generated (LLM) solutions against reference JSON meta.
    Reuses code extraction and execution logic from CodeSolutionEvaluator.
    """

    def evaluate_generated_code(
        self, problem_data: dict, model_output: str, problem_id: str
    ) -> EvaluationResult:
        """
        Evaluate the model output against the problem definition.
        """
        # 1. Extract Code
        python_code = self.extract_python_code(model_output)

        if not python_code:
            return EvaluationResult(
                problem_id=problem_id,
                execution_succeeded=False,
                status_optimal=False,
                objective_error=None,
                objective_matches=False,
                execution_time=0.0,
                reference_optimum=0.0,
                generated_solution=ExecutionResult(
                    exec_success=False,
                    status=None,
                    objective_value=None,
                    variable_values={},
                    stdout="",
                    stderr="No code found",
                    execution_time=0.0,
                    exit_code=-1
                ),
            )

        # 2. Execute Code
        exec_result = self.execute_code_safely(python_code)

        # 3. Compare with Ground Truth (Meta/Result)
        meta = problem_data.get("meta", {})
        result_block = problem_data.get("gurobi_result", {})

        ref_optimum = result_block.get("theoretical_optimum")
        if ref_optimum is None:
            ref_optimum = meta.get("theoretical_optimum", 0.0)

        # Validation Logic: compare solver status with reference
        ref_status = result_block.get("solver_status", meta.get("solver_status"))
        status_correct = (exec_result.status == ref_status)

        objective_error = None
        if exec_result.objective_value is not None:
            objective_error = abs(exec_result.objective_value - ref_optimum)

        obj_ok = (exec_result.objective_value is not None and 
                  self.objectives_match(exec_result.objective_value, ref_optimum))

        return EvaluationResult(
            problem_id=problem_id,
            execution_succeeded=exec_result.exec_success,
            status_optimal=status_correct,
            objective_error=objective_error,
            objective_matches=obj_ok,
            execution_time=exec_result.execution_time,
            reference_optimum=ref_optimum,
            generated_solution=exec_result,
        )

    async def evaluate_generated_code_async(
        self, problem_data: dict, model_output: str, problem_id: str
    ) -> EvaluationResult:
        """Async version — runs Gurobi subprocess without blocking the event loop."""
        python_code = self.extract_python_code(model_output)

        if not python_code:
            return EvaluationResult(
                problem_id=problem_id,
                execution_succeeded=False, status_optimal=False,
                objective_error=None, objective_matches=False,
                execution_time=0.0, reference_optimum=0.0,
                generated_solution=ExecutionResult(
                    exec_success=False, status=None, objective_value=None,
                    variable_values={}, stdout="", stderr="No code found",
                    execution_time=0.0, exit_code=-1,
                ),
            )

        exec_result = await self.execute_code_safely_async(python_code)

        meta = problem_data.get("meta", {})
        result_block = problem_data.get("gurobi_result", {})
        ref_optimum = result_block.get("theoretical_optimum")
        if ref_optimum is None:
            ref_optimum = meta.get("theoretical_optimum", 0.0)

        ref_status = result_block.get("solver_status", meta.get("solver_status"))
        status_correct = (exec_result.status == ref_status)
        objective_error = None
        if exec_result.objective_value is not None:
            objective_error = abs(exec_result.objective_value - ref_optimum)
        obj_ok = (exec_result.objective_value is not None and
                  self.objectives_match(exec_result.objective_value, ref_optimum))

        return EvaluationResult(
            problem_id=problem_id,
            execution_succeeded=exec_result.exec_success,
            status_optimal=status_correct,
            objective_error=objective_error,
            objective_matches=obj_ok,
            execution_time=exec_result.execution_time,
            reference_optimum=ref_optimum,
            generated_solution=exec_result,
        )

