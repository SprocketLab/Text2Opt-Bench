import numpy as np
import time
import os
import re
from typing import List, Tuple, Optional, Union

from main.generation.base_problem import BaseProblem
from main.generation.unstructured.resource_allocation_lp_generator import ResourceAllocationGenerator
from main.utils import save_json


class BlockDiagonalLPGenerator(BaseProblem):
    """
    Block-diagonal LP problem generator.
    
    Generates a decomposable LP problem with N independent blocks:
    - Each block is generated using ResourceAllocationGenerator
    - Blocks are combined into a block-diagonal matrix structure
    - The problem can be decomposed: Z* = Z_1* + Z_2* + ... + Z_N*
    - Individual blocks are saved separately for individual prompt generation
    - Central assignment prompt coordinates the blocks together
    
    **Real-world OR considerations:**
    In real-world scenarios, different sites/departments (blocks) often have:
    - Different operational characteristics (sparsity/density of constraints)
    - Different scales of operations (value ranges for coefficients)
    - Different resource availability (upper bounds)
    
    The generator supports both uniform parameters (all blocks same) and
    per-block parameters (each block can have different characteristics).
    Use per-block parameters for more realistic, heterogeneous scenarios.
    """

    # System prompt for overview generation (metadata only, no specific details)
    OVERVIEW_SYSTEM_PROMPT = """You are a Senior Operations Research Consultant.
Your task is to create a high-level overview of a multi-site resource allocation problem.

**CONTEXT**: This is a block-diagonal optimization problem with N independent sites/units. You will receive ONLY metadata about the problem structure (number of blocks, total variables, total constraints, goal), but NO specific details about individual blocks.

**CORE PHILOSOPHY: HIGH-LEVEL OVERVIEW**
The output must be a brief overview that:
- Sets the scene for a central distributor/coordinator managing multiple disjoint sites
- Describes the overall problem structure without specific details
- Explains the coordination framework
- Does NOT include any specific variable names, constraint names, or numerical values

**STRUCTURE**:
1. **Introduction**: Set the scene (e.g., "We operate a central distribution network with N independent sites...")
2. **Problem Structure**: Describe that there are N independent sites, each making their own decisions
3. **Coordination**: Explain how the central coordinator manages these sites
4. **Objective**: State the overall goal (maximize/minimize) without specifics

**IMPORTANT**:
- Keep it general and high-level
- NO specific variable or constraint names
- NO numerical values
- Focus on the organizational structure and coordination mechanism"""

    # System prompt template for individual block descriptions (with overview context)
    # Note: {block_idx} will be replaced with actual block index
    BLOCK_SYSTEM_PROMPT_TEMPLATE = """You are a Senior Operations Research Consultant.
Your task is to translate a structured JSON resource allocation optimization dataset into a strictly natural language business problem for ONE specific site in a multi-site operation.

**CONTEXT**: This is ONE block/site in a larger block-diagonal optimization problem. You have been provided with an overview of the overall problem structure. Your job is to describe THIS SPECIFIC SITE's problem in detail.

**CORE PHILOSOPHY: TOTAL ABSTRACTION WITH SITE IDENTIFICATION**
The output must look exactly like an email or memo from a site manager.
It must NOT look like a math problem. There should be no references to 'Var_0', 'C1_Res', or 'Objective Functions'.

**CRITICAL NAMING REQUIREMENT**:
- ALL variable names MUST end with "_block{block_idx}" where {block_idx} is the block index (0-indexed)
- ALL constraint names MUST end with "_block{block_idx}" where {block_idx} is the block index (0-indexed)
- Example: If block index is 0, variables might be "Production_Volume_block0", "Inventory_Level_block0"
- Example: If block index is 1, constraints might be "Capacity_Limit_block1", "Demand_Requirement_block1"

**Naming & Creativity Rules:**
1. **Invent Your Own Names**: You have total freedom to name the decision variables and constraints to fit the scenario, BUT they must end with "_block{block_idx}".
2. **Strict Implicit Ordering**: Describe entities in the **exact order** they appear in the JSON.
3. **Include Overview Context**: Reference the overview provided to show how this site fits into the larger operation.

**Structure Strategy (Entity-Centric):**
1. **The Introduction**: Reference the overview and set the scene for THIS specific site's resource allocation decisions.
2. **The Levers (Variables)**: Dedicate one distinct paragraph (or bullet point) to each decision variable.
   Describe its costs, revenues, and how it impacts every single constraint (resource usage, pollution, quality, etc.).
   Embed the numerical coefficients naturally into sentences.
3. **The Limits (Constraints)**: Conclude with a 'Facility Specs' or 'Market Limits' section that lists the total capacities (RHS values) for the resources mentioned above.

**Non-Redundancy**: Ensure that no constraint or variable is described more than once. The description should be concise and avoid repetition.

**Completeness Check**: You must include every single coefficient, bound, and RHS value from the JSON, disguised as business data."""

    def __init__(
        self,
        block_specs: List[Tuple[int, int]],  # List of (n_vars, n_constrs) for each block
        integer_indices: Optional[List[List[int]]] = None,  # List of integer indices per block
        sparsity: Union[float, List[float]] = 0.3,  # Can be uniform or per-block
        value_low: Union[float, List[float]] = -10.0,  # Can be uniform or per-block
        value_high: Union[float, List[float]] = 10.0,  # Can be uniform or per-block
        goal: str = "MAXIMIZE",
        ub_min: Union[float, List[float]] = 5.0,  # Can be uniform or per-block
        ub_max: Union[float, List[float]] = 20.0,  # Can be uniform or per-block
    ):
        """
        Initialize block-diagonal LP generator.
        
        Args:
            block_specs: List of (n_vars, n_constrs) tuples, one per block
            integer_indices: Optional list of integer variable indices per block
            sparsity: Sparsity level(s) - if float, same for all blocks; if list, one per block
            value_low: Lower bound(s) for matrix coefficients - if float, same for all; if list, one per block
            value_high: Upper bound(s) for matrix coefficients - if float, same for all; if list, one per block
            goal: "MAXIMIZE" or "MINIMIZE"
            ub_min: Minimum upper bound(s) for variables - if float, same for all; if list, one per block
            ub_max: Maximum upper bound(s) for variables - if float, same for all; if list, one per block
        """
        # Calculate total dimensions
        total_vars = sum(n_vars for n_vars, _ in block_specs)
        total_constrs = sum(n_constrs for _, n_constrs in block_specs)
        
        super().__init__(
            n_vars=total_vars,
            n_constrs=total_constrs,
            prob_type="LP",
            goal=goal,
            problem_type="block diagonal resource allocation"
        )
        
        self.block_specs = block_specs
        self.n_blocks = len(block_specs)
        self.integer_indices = integer_indices if integer_indices else [[] for _ in range(self.n_blocks)]
        
        # Handle uniform vs per-block parameters
        # Convert single values to lists for uniform application
        if isinstance(sparsity, (int, float)):
            self.sparsity = [np.clip(float(sparsity), 0.0, 1.0) for _ in range(self.n_blocks)]
        else:
            self.sparsity = [np.clip(float(s), 0.0, 1.0) for s in sparsity[:self.n_blocks]]
            if len(self.sparsity) < self.n_blocks:
                # Pad with last value if not enough provided
                self.sparsity.extend([self.sparsity[-1]] * (self.n_blocks - len(self.sparsity)))
        
        if isinstance(value_low, (int, float)):
            self.value_low = [float(value_low) for _ in range(self.n_blocks)]
        else:
            self.value_low = [float(v) for v in value_low[:self.n_blocks]]
            if len(self.value_low) < self.n_blocks:
                self.value_low.extend([self.value_low[-1]] * (self.n_blocks - len(self.value_low)))
        
        if isinstance(value_high, (int, float)):
            self.value_high = [float(value_high) for _ in range(self.n_blocks)]
        else:
            self.value_high = [float(v) for v in value_high[:self.n_blocks]]
            if len(self.value_high) < self.n_blocks:
                self.value_high.extend([self.value_high[-1]] * (self.n_blocks - len(self.value_high)))
        
        if isinstance(ub_min, (int, float)):
            self.ub_min = [float(ub_min) for _ in range(self.n_blocks)]
        else:
            self.ub_min = [float(v) for v in ub_min[:self.n_blocks]]
            if len(self.ub_min) < self.n_blocks:
                self.ub_min.extend([self.ub_min[-1]] * (self.n_blocks - len(self.ub_min)))
        
        if isinstance(ub_max, (int, float)):
            self.ub_max = [float(ub_max) for _ in range(self.n_blocks)]
        else:
            self.ub_max = [float(v) for v in ub_max[:self.n_blocks]]
            if len(self.ub_max) < self.n_blocks:
                self.ub_max.extend([self.ub_max[-1]] * (self.n_blocks - len(self.ub_max)))
        
        # Ensure ub_max >= ub_min for each block
        for i in range(self.n_blocks):
            self.ub_max[i] = max(self.ub_min[i], self.ub_max[i])
        
        # Store individual block generators
        self.blocks: List[ResourceAllocationGenerator] = []
        
        # Store per-block parameters for reference
        self.block_parameters = [
            {
                "sparsity": self.sparsity[i],
                "value_low": self.value_low[i],
                "value_high": self.value_high[i],
                "ub_min": self.ub_min[i],
                "ub_max": self.ub_max[i],
            }
            for i in range(self.n_blocks)
        ]
        
        # Store block boundaries for variable/constraint indexing
        self.var_offsets = []  # Starting index for each block's variables
        self.constr_offsets = []  # Starting index for each block's constraints
        
        var_offset = 0
        constr_offset = 0
        for n_vars, n_constrs in block_specs:
            self.var_offsets.append(var_offset)
            self.constr_offsets.append(constr_offset)
            var_offset += n_vars
            constr_offset += n_constrs

    def generate(self, time_limit: float = 300.0):
        """
        Generate the block-diagonal problem by generating each block and combining them.
        
        Args:
            time_limit: Maximum time to spend on generation (distributed across blocks)
            
        Returns:
            self
        """
        max_retries = 20
        start_time = time.time()
        time_per_block = time_limit / self.n_blocks if self.n_blocks > 0 else time_limit
        
        for attempt in range(max_retries):
            # Check total time limit
            if time.time() - start_time > time_limit:
                print(f"Warning: Generation time limit ({time_limit}s) exceeded.")
                break
            
            # Generate each block
            self.blocks = []
            all_solved = True
            
            for block_idx, (n_vars, n_constrs) in enumerate(self.block_specs):
                block_gen = ResourceAllocationGenerator(
                    n_vars=n_vars,
                    n_constrs=n_constrs,
                    integer_indices=self.integer_indices[block_idx] if block_idx < len(self.integer_indices) else [],
                    sparsity=self.sparsity[block_idx],  # Per-block sparsity
                    value_low=self.value_low[block_idx],  # Per-block value_low
                    value_high=self.value_high[block_idx],  # Per-block value_high
                    goal=self.goal,
                    ub_min=self.ub_min[block_idx],  # Per-block ub_min
                    ub_max=self.ub_max[block_idx],  # Per-block ub_max
                )
                
                try:
                    block_gen.generate(time_limit=time_per_block)
                    if block_gen.obj_val is None or abs(block_gen.obj_val) < 1e-4:
                        all_solved = False
                        break
                    self.blocks.append(block_gen)
                except Exception as exc:
                    all_solved = False
                    if isinstance(exc, TimeoutError):
                        print(f"    [timeout] Block {block_idx} exceeded time limit, retrying...")
                    break
            
            if not all_solved or len(self.blocks) != self.n_blocks:
                continue
            
            # Combine blocks into block-diagonal structure
            self._combine_blocks()
            
            # Solve the combined problem to verify it's valid
            try:
                self.solve(time_limit=time_limit * 0.1)  # Use small fraction for verification
                
                # Verify decomposition property: total obj = sum of block objs
                total_block_obj = sum(block.obj_val for block in self.blocks)
                if abs(self.obj_val - total_block_obj) > 1e-3:
                    print(f"    [warning] Decomposition property violated: combined={self.obj_val}, sum={total_block_obj}")
                    continue
                
                # Check if non-trivial
                if abs(self.obj_val) > 1e-4:
                    return self
            except Exception as exc:
                if isinstance(exc, TimeoutError):
                    print(f"    [timeout] Combined problem exceeded time limit, retrying...")
                continue
        
        print(f"Warning: Could not generate non-trivial block-diagonal problem after {max_retries} retries.")
        return self

    def _combine_blocks(self):
        """Combine individual blocks into a block-diagonal matrix structure."""
        # Initialize combined matrices
        A_combined = np.zeros((self.m, self.n))
        b_combined = np.zeros(self.m)
        c_combined = np.zeros(self.n)
        senses_combined = []
        var_types_combined = []
        var_bounds_combined = []
        
        # Place each block in the appropriate position
        for block_idx, block in enumerate(self.blocks):
            var_offset = self.var_offsets[block_idx]
            constr_offset = self.constr_offsets[block_idx]
            
            n_vars = block.n
            n_constrs = block.m
            
            # Place block matrix in diagonal position
            A_combined[
                constr_offset : constr_offset + n_constrs,
                var_offset : var_offset + n_vars
            ] = block.A
            
            # Combine RHS
            b_combined[constr_offset : constr_offset + n_constrs] = block.b
            
            # Combine objective
            c_combined[var_offset : var_offset + n_vars] = block.c
            
            # Combine constraint senses
            senses_combined.extend(block.senses)
            
            # Combine variable types
            var_types_combined.extend(block.var_types)
            
            # Combine variable bounds
            var_bounds_combined.extend(block.var_bounds)
        
        self.A = A_combined
        self.b = b_combined
        self.c = c_combined
        self.senses = senses_combined
        self.var_types = var_types_combined
        self.var_bounds = var_bounds_combined
        
        # Combine anchor solutions (for metadata)
        anchor_combined = np.zeros(self.n)
        slack_combined = np.zeros(self.m)
        for block_idx, block in enumerate(self.blocks):
            var_offset = self.var_offsets[block_idx]
            constr_offset = self.constr_offsets[block_idx]
            anchor_combined[var_offset : var_offset + block.n] = block.anchor_x
            slack_combined[constr_offset : constr_offset + block.m] = block.slack
        
        self.anchor_x = anchor_combined
        self.slack = slack_combined

    def get_block_data(self, block_idx: int, include_gold_solution: bool = False) -> dict:
        """
        Get the JSON data for a specific block (for individual prompt generation).
        
        Args:
            block_idx: Index of the block (0-indexed)
            include_gold_solution: If True, include gold_solution code in the returned data
            
        Returns:
            Dictionary with the block's problem data
        """
        if block_idx < 0 or block_idx >= len(self.blocks):
            raise ValueError(f"Block index {block_idx} out of range [0, {len(self.blocks)})")
        
        data = self.blocks[block_idx].to_json_dict()
        
        if include_gold_solution:
            # Generate gold solution code for the block
            block = self.blocks[block_idx]
            data["gold_solution"] = block.generate_gurobi_code_reference(data)
        
        return data

    def get_all_blocks_data(self) -> List[dict]:
        """
        Get JSON data for all blocks (for individual prompt generation).
        
        Returns:
            List of dictionaries, one per block
        """
        return [block.to_json_dict() for block in self.blocks]

    def save_to_problem_folder(self, problem_folder: str, problem_name: str = "block_problem"):
        """
        Save the complete block-diagonal problem to a structured folder:
        - problem_folder/
          - metadata.json (overall problem characteristics)
          - blocks/
            - block_0.json
            - block_1.json
            - ...
            - block_n.json
        
        Args:
            problem_folder: Path to the folder where the problem should be saved
            problem_name: Name for the metadata file (default: "block_problem")
            
        Returns:
            Dictionary with paths to metadata file and block files
        """
        # Create the main problem folder
        os.makedirs(problem_folder, exist_ok=True)
        
        # Create blocks subfolder
        blocks_dir = os.path.join(problem_folder, "blocks")
        os.makedirs(blocks_dir, exist_ok=True)
        
        # Save each block to blocks/ subfolder
        block_files = []
        for block_idx in range(len(self.blocks)):
            block_file = os.path.join(blocks_dir, f"block_{block_idx}.json")
            self.save_block_to_file(block_idx, block_file)
            block_files.append(block_file)
        
        # Save metadata file with overall problem characteristics
        metadata = self.to_json_dict()
        metadata["gold_solution"] = self.generate_gurobi_code_reference(metadata)
        
        # Add paths to block files (relative to problem folder)
        metadata["block_files"] = [f"blocks/block_{i}.json" for i in range(len(self.blocks))]
        
        metadata_file = os.path.join(problem_folder, f"{problem_name}.json")
        save_json(metadata_file, metadata)
        
        return {
            "problem_folder": problem_folder,
            "metadata_file": metadata_file,
            "blocks_dir": blocks_dir,
            "block_files": block_files,
        }

    def save_block_to_folder(self, block_idx: int, folder_path: str, filename: str = "problem.json"):
        """
        Save a specific block to its own folder as a complete problem.
        (Legacy method - kept for backward compatibility)
        
        Args:
            block_idx: Index of the block to save (0-indexed)
            folder_path: Path to the folder where the block should be saved
            filename: Name of the JSON file inside the folder (default: "problem.json")
        """
        if block_idx < 0 or block_idx >= len(self.blocks):
            raise ValueError(f"Block index {block_idx} out of range [0, {len(self.blocks)})")
        
        # Create the folder
        os.makedirs(folder_path, exist_ok=True)
        
        block = self.blocks[block_idx]
        data = block.to_json_dict()
        
        # Add metadata about which block this is
        if "meta" not in data:
            data["meta"] = {}
        data["meta"]["block_index"] = block_idx
        data["meta"]["total_blocks"] = self.n_blocks
        data["meta"]["is_block_component"] = True
        
        # Save to the folder
        filepath = os.path.join(folder_path, filename)
        save_json(filepath, data)
        
        return filepath

    def save_all_blocks_to_directory(self, base_dir: str, block_folder_prefix: str = "block"):
        """
        Save all blocks to separate folders, each containing a complete problem.
        (Legacy method - kept for backward compatibility)
        
        Args:
            base_dir: Base directory where block folders should be created
            block_folder_prefix: Prefix for block folder names (will be appended with _0, _1, etc.)
            
        Returns:
            List of folder paths where blocks were saved
        """
        os.makedirs(base_dir, exist_ok=True)
        folder_paths = []
        
        for block_idx in range(len(self.blocks)):
            folder_name = f"{block_folder_prefix}_{block_idx}"
            folder_path = os.path.join(base_dir, folder_name)
            self.save_block_to_folder(block_idx, folder_path)
            folder_paths.append(folder_path)
        
        return folder_paths

    def save_block_to_file(self, block_idx: int, filepath: str):
        """
        Save a specific block to a separate JSON file with renamed variables and constraints.
        Variables and constraints are renamed to end with "_block{i}" suffix.
        
        Args:
            block_idx: Index of the block to save (0-indexed)
            filepath: Path where the block file should be saved
        """
        if block_idx < 0 or block_idx >= len(self.blocks):
            raise ValueError(f"Block index {block_idx} out of range [0, {len(self.blocks)})")
        
        block = self.blocks[block_idx]
        data = block.to_json_dict()
        
        # Add metadata about which block this is
        if "meta" not in data:
            data["meta"] = {}
        data["meta"]["block_index"] = block_idx
        data["meta"]["total_blocks"] = self.n_blocks
        data["meta"]["is_block_component"] = True
        
        # Rename variables and constraints to include _block{i} suffix
        suffix = f"_block{block_idx}"
        
        # Create mapping from old names to new names
        var_name_map = {}
        constr_name_map = {}
        
        # Map variable names
        old_var_names = list(data["variables"].keys())
        for old_name in old_var_names:
            # If name already ends with the suffix, keep it; otherwise append suffix
            if old_name.endswith(suffix):
                new_name = old_name
            else:
                # Remove any existing _block* suffix first
                base_name = re.sub(r'_block\d+$', '', old_name)
                new_name = f"{base_name}{suffix}"
            var_name_map[old_name] = new_name
        
        # Map constraint names
        old_constr_names = list(data["constraints"].keys())
        for old_name in old_constr_names:
            # If name already ends with the suffix, keep it; otherwise append suffix
            if old_name.endswith(suffix):
                new_name = old_name
            else:
                # Remove any existing _block* suffix first
                base_name = re.sub(r'_block\d+$', '', old_name)
                new_name = f"{base_name}{suffix}"
            constr_name_map[old_name] = new_name
        
        # Rename variables in the data structure
        new_variables = {}
        for old_var_name, var_data in data["variables"].items():
            new_var_name = var_name_map[old_var_name]
            # Rename constraint references in resource_costs
            new_resource_costs = {}
            for old_constr_name, coeff in var_data.get("resource_costs", {}).items():
                new_constr_name = constr_name_map[old_constr_name]
                new_resource_costs[new_constr_name] = coeff
            var_data["resource_costs"] = new_resource_costs
            new_variables[new_var_name] = var_data
        data["variables"] = new_variables
        
        # Rename constraints in the data structure
        new_constraints = {}
        for old_constr_name, constr_data in data["constraints"].items():
            new_constr_name = constr_name_map[old_constr_name]
            new_constraints[new_constr_name] = constr_data
        data["constraints"] = new_constraints
        
        # Rename variables in optimal_values
        new_optimal_values = {}
        for old_var_name, value in data["gurobi_result"]["optimal_values"].items():
            new_var_name = var_name_map[old_var_name]
            new_optimal_values[new_var_name] = value
        data["gurobi_result"]["optimal_values"] = new_optimal_values
        
        # Regenerate gold_solution code with renamed variables and constraints
        # This ensures the code uses the correct _block{i} suffix names
        # Use the block's own method to generate code (it's an instance method)
        # Note: block is already defined at the start of this method
        data["gold_solution"] = block.generate_gurobi_code_reference(data)
        
        save_json(filepath, data)

    def save_to_file(self, filepath: str):
        """
        Save the combined block-diagonal problem to a single file.
        For structured folder saving, use save_to_problem_folder() instead.
        
        Override to add block file references.
        """
        data = self.to_json_dict()
        # Attach gold solution code
        data["gold_solution"] = self.generate_gurobi_code_reference(data)
        
        # Add metadata about block files (if they were saved separately)
        if "meta" not in data:
            data["meta"] = {}
        data["meta"]["has_separate_block_files"] = True
        data["meta"]["n_blocks"] = self.n_blocks
        
        save_json(filepath, data)

    def to_json_dict(self):
        """Serialize the combined block-diagonal problem to JSON."""
        if self.x_star is None or self.obj_val is None:
            raise ValueError("Problem not solved yet. Call solve() first.")
        
        # Get the base JSON structure
        data = super().to_json_dict()
        
        # Add block structure metadata
        data["meta"]["n_blocks"] = self.n_blocks
        data["meta"]["block_specs"] = self.block_specs
        data["meta"]["var_offsets"] = self.var_offsets
        data["meta"]["constr_offsets"] = self.constr_offsets
        
        # Add per-block parameters (for reference and realism tracking)
        data["meta"]["block_parameters"] = self.block_parameters
        
        # Add individual block optimal values for verification
        data["block_optimal_values"] = [
            float(block.obj_val) for block in self.blocks
        ]
        
        # Add block summaries for the assignment prompt
        data["block_summaries"] = []
        for block_idx, block in enumerate(self.blocks):
            data["block_summaries"].append({
                "block_index": block_idx,
                "n_vars": block.n,
                "n_constrs": block.m,
                "optimal_value": float(block.obj_val),
                "sparsity": self.sparsity[block_idx],
                "value_range": [self.value_low[block_idx], self.value_high[block_idx]],
                "ub_range": [self.ub_min[block_idx], self.ub_max[block_idx]],
            })
        
        return data

    @classmethod
    def build_overview_prompt_messages(cls, problem_data: dict) -> list[dict]:
        """
        Build prompt messages for generating the overview (metadata only, no specific details).
        
        Args:
            problem_data: The problem data dictionary (will be sanitized)
            
        Returns:
            List of message dicts with 'role' and 'content' keys for overview generation
        """
        import json
        
        clean_data = cls._sanitize_for_prompt(problem_data)
        
        # Extract only metadata (no specific block details)
        meta = clean_data.get("meta", {})
        n_blocks = meta.get("n_blocks", 0)
        goal = meta.get("goal", "MAXIMIZE")
        total_vars = meta.get("num_vars", 0)
        total_constrs = meta.get("num_constraints", 0)
        
        user_content = f"""Create a high-level overview of a multi-site resource allocation problem.

**Problem Metadata:**
- Number of independent sites/blocks: {n_blocks}
- Overall optimization goal: {goal}
- Total decision variables across all sites: {total_vars}
- Total constraints across all sites: {total_constrs}

**Your Task:**
Create a brief overview that:
1. Sets the scene for a central distributor/coordinator managing {n_blocks} disjoint sites
2. Describes the overall problem structure without specific details
3. Explains the coordination framework
4. Does NOT include any specific variable names, constraint names, or numerical values

**Important**: This is metadata only. Do NOT include specific details about individual sites or their variables/constraints.

Write the high-level overview."""
        
        return [
            {"role": "system", "content": cls.OVERVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    @classmethod
    def build_block_prompt_messages(cls, block_data: dict, overview_description: str, block_idx: int) -> list[dict]:
        """
        Build prompt messages for generating an individual block description.
        
        Args:
            block_data: The block's problem data dictionary (will be sanitized)
            overview_description: The overview description generated from metadata
            block_idx: The index of this block (0-indexed)
            
        Returns:
            List of message dicts with 'role' and 'content' keys for block description generation
        """
        import json
        
        clean_data = cls._sanitize_for_prompt(block_data)
        
        user_content = f"""You are describing ONE specific site (Block {block_idx}) in a multi-site operation.

**Overview of the Overall Problem:**
{overview_description}

**Your Task:**
Describe THIS specific site's resource allocation problem in detail. 

**CRITICAL REQUIREMENT**: 
- ALL variable names MUST end with "_block{block_idx}"
- ALL constraint names MUST end with "_block{block_idx}"
- Example: "Production_Volume_block{block_idx}", "Capacity_Limit_block{block_idx}"

**Block {block_idx} Problem Data:**
{json.dumps(clean_data, indent=2)}

Write the detailed problem description for this site, making sure all variable and constraint names end with "_block{block_idx}"."""
        
        block_system_prompt = cls.BLOCK_SYSTEM_PROMPT_TEMPLATE.replace("{block_idx}", str(block_idx))
        
        return [
            {"role": "system", "content": block_system_prompt},
            {"role": "user", "content": user_content},
        ]

    @classmethod
    def build_prompt_messages(cls, problem_data: dict) -> list[dict]:
        """
        Build prompt messages for the central assignment prompt.
        (Legacy method - kept for backward compatibility)
        """
        return cls.build_overview_prompt_messages(problem_data)

    @classmethod
    def get_system_prompt(cls) -> str:
        """Return the overview system prompt (for backward compatibility)."""
        return cls.OVERVIEW_SYSTEM_PROMPT

    @staticmethod
    def concatenate_final_prompt(overview_description: str, block_descriptions: List[str]) -> str:
        """
        Concatenate the overview and all block descriptions into a final prompt.
        
        Args:
            overview_description: The overview description
            block_descriptions: List of individual block descriptions (one per block)
            
        Returns:
            The final concatenated prompt with scene-setting
        """
        lines = []
        
        # Scene-setting introduction
        lines.append("=" * 70)
        lines.append("CENTRAL DISTRIBUTION NETWORK - MULTI-SITE OPTIMIZATION PROBLEM")
        lines.append("=" * 70)
        lines.append("")
        lines.append("We operate a central distribution network with multiple disjoint sites.")
        lines.append("Each site operates independently but contributes to the overall organizational objective.")
        lines.append("")
        lines.append("-" * 70)
        lines.append("")
        
        # Overview
        lines.append("**PROBLEM OVERVIEW**")
        lines.append("")
        lines.append(overview_description)
        lines.append("")
        lines.append("-" * 70)
        lines.append("")
        
        # Individual site descriptions
        lines.append("**INDIVIDUAL SITE DESCRIPTIONS**")
        lines.append("")
        for i, block_desc in enumerate(block_descriptions):
            lines.append(f"--- Site {i+1} (Block {i}) ---")
            lines.append("")
            lines.append(block_desc)
            lines.append("")
            if i < len(block_descriptions) - 1:
                lines.append("-" * 70)
                lines.append("")
        
        # Final note
        lines.append("=" * 70)
        lines.append("")
        lines.append("**SOLUTION REQUIREMENT**:")
        lines.append("Solve the combined optimization problem that includes all sites.")
        lines.append("The overall objective is the sum of individual site objectives.")
        lines.append("Each site's variables and constraints are independent (block-diagonal structure).")
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)

    async def generate_descriptions_workflow(self, model: str = "gpt-5"):
        """
        Generate descriptions following the complete workflow:
        1. Generate overview from metadata only
        2. Generate individual block descriptions with renamed variables/constraints
        3. Concatenate into final prompt
        
        Args:
            model: Model to use for generation
            
        Returns:
            Dictionary with:
                - overview: Overview description
                - block_descriptions: List of block descriptions
                - final_prompt: Concatenated final prompt
        """
        from main.generation.language_description_creator import generate_description_async
        
        # Step 1: Generate overview from metadata only
        metadata = self.to_json_dict()
        overview_messages = self.build_overview_prompt_messages(metadata)
        
        # Use async generation
        import asyncio
        from main.utils import get_async_openai_client, resolve_chat_deployment
        
        aclient = get_async_openai_client()
        deployment_name = resolve_chat_deployment(model)
        
        # Generate overview
        extra_body = {"reasoning_effort": "high"} if model in ("gpt-5.1", "gpt-5") else None
        completion = await aclient.chat.completions.create(
            model=deployment_name,
            messages=overview_messages,
            **({"extra_body": extra_body} if extra_body is not None else {}),
        )
        overview_description = completion.choices[0].message.content
        
        # Step 2: Generate individual block descriptions
        block_descriptions = []
        for block_idx in range(len(self.blocks)):
            block_data = self.get_block_data(block_idx)
            block_messages = self.build_block_prompt_messages(block_data, overview_description, block_idx)
            
            completion = await aclient.chat.completions.create(
                model=deployment_name,
                messages=block_messages,
                **({"extra_body": extra_body} if extra_body is not None else {}),
            )
            block_desc = completion.choices[0].message.content
            block_descriptions.append(block_desc)
        
        # Step 3: Concatenate final prompt
        final_prompt = self.concatenate_final_prompt(overview_description, block_descriptions)
        
        return {
            "overview": overview_description,
            "block_descriptions": block_descriptions,
            "final_prompt": final_prompt,
        }

