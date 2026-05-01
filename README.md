# OR-LLM-Bench

A scalable benchmark for evaluating LLMs on operations research optimization problems (LP, MILP, MIQP, nonlinear) with solver-verified ground truth.

[[Paper]](https://arxiv.org/abs/XXXX.XXXXX)

## Setup

```bash
pip install -r requirements.txt
```

**API keys** — either set environment variables or add to `api_keys/keys.json`:

```bash
# Option 1: Environment variables
export OPENAI_API_KEY="sk-..."        # For GPT-5, GPT-5-Nano, o4-mini
export ANTHROPIC_API_KEY="sk-ant-..." # For Claude Opus/Sonnet
export VLLM_BASE_URL="http://localhost:8000/v1"  # For open-weight models (optional)
```

```bash
# Option 2: api_keys/keys.json (see sample_keys.json for format)
cp api_keys/sample_keys.json api_keys/keys.json
# Edit keys.json with your API keys
```

**Gurobi**: A Gurobi license is required for evaluation. The free "restricted" license works for small problems, but some categories (stochastic transportation, large-tier instances) require a full license due to solver time limits. Academic licenses are free at [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Dataset

```
synthetic_dataset/
├── Template/                    # 550 small-tier problems (50 per category)
│   ├── basic_template/
│   │   ├── transportation/      # Supply-demand LP
│   │   ├── disaster_response/   # Multi-period logistics MILP
│   │   ├── jssp/                # Job-shop scheduling MILP
│   │   ├── vrptw/               # Vehicle routing + time windows MILP
│   │   └── rcpsp/               # Multi-mode project scheduling MILP
│   ├── induce_constraint/
│   │   ├── facility_location/   # Euclidean distance derivation MILP
│   │   ├── power_transmission/  # Ohm's law derivation MIQP
│   │   └── queuing_staffing/    # Erlang-C nonlinear
│   └── industrial/
│       ├── stochastic_transportation/      # SAA chance-constrained MILP
│       ├── multiobjective_transportation/  # Bi-objective cost+emissions MILP
│       └── modified_facility_location/     # Extended constraints MILP
├── Template_large/              # Large-tier instances (7K-48K data tokens)
│   ├── transportation/          # ~90x90 cost matrices
│   ├── queuing_staffing/
│   └── multiobjective_transportation/
└── Unstructured/
    └── resource_allocation/     # 1,012 LP/MILP instances (prose-embedded data)
```

Each problem JSON contains: `LLM_description`, `instance_data`, `gold_solution`, `gurobi_result`, `variables`, `constraints`.

## Quick Start

### Single problem (sanity check)
```bash
python main/evaluation/run_eval.py \
  --dataset-path synthetic_dataset/Template/basic_template/transportation/small/trans_001.json \
  --model gpt-5-nano \
  --output-path results/test.json
```

### Pass@1 baseline on the full Template set
```bash
python main/evaluation/run_eval.py \
  --dataset-path synthetic_dataset/Template \
  --model gpt-5 \
  --output-path results/gpt5_template.json \
  --max-concurrent 200
```

### Test-time compute strategies
The three strategies below are independent — run any combination. The paper's full-TTC numbers come from running all three on the same model/dataset.

**1. BIND (agentic offload)** — data externalized to JSON, schema-only prompt, iterative repair:
```bash
python main/evaluation/run_eval.py \
  --dataset-path synthetic_dataset/Template \
  --model gpt-5 \
  --inference-option agentic-offload \
  --agentic-max-rounds 3 \
  --output-path results/gpt5_agentic.json
```

**2. Best-of-K** — reuses a Pass@1 baseline as sample 0 and draws K-1 more:
```bash
python scripts/evaluation/run_best_of_k_with_baseline.py \
  --dataset-paths synthetic_dataset/Template \
  --model gpt-5 -k 5 \
  --baseline-results results/gpt5_template.json \
  --output-path results/gpt5_best_of_5.json
```

**3. Iterative repair** — fixed-window oracle feedback loop:
```bash
python scripts/evaluation/run_repair_curve_sliding.py \
  --mode direct \
  --dataset-paths synthetic_dataset/Template \
  --model gpt-5 \
  --max-rounds 5 --window-size 1 \
  --output-path results/gpt5_repair.json
```

## Evaluation

A response is **correct** iff the generated Gurobi code:
1. Executes without error
2. Achieves optimal solver status
3. Produces an objective value matching the ground truth (relative tolerance 10^-4)

| Mode | Description |
|------|-------------|
| `default` | Full problem description in prompt (Pass@1) |
| `agentic-offload` | BIND: data externalized to JSON; schema-only prompt; iterative repair |

## RULER Long-Context Evaluation

Synthetic long-context retrieval/aggregation tasks (NIAH variants + counting) for measuring effective context length. Pre-generated samples ship in `ruler/samples/` (4 tasks × 6 lengths from 1K to 32K tokens). Requires `vllm`.

```bash
# Run a model against the pre-generated samples (recommended)
python ruler/eval.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --samples-dir ruler/samples/ \
  --output results/ruler/qwen-7B.json

# (Optional) Regenerate samples — overwrites ruler/samples/
python ruler/eval.py \
  --generate-samples \
  --samples-dir ruler/samples/ \
  --tokenizer-model Qwen/Qwen2.5-7B-Instruct \
  --tasks niah_single,niah_multikey,niah_multivalue,aggregation \
  --lengths 1024,2048,4096,8192,16384,32000 \
  --samples-per-length 200

# Re-score existing raw results without re-running inference
python ruler/eval.py --validate-only results/ruler/qwen-7B.json --output results/ruler/qwen-7B_rescored.json

# Plot accuracy vs. context length across models
# (edit the MODEL_RESULTS dict and OUTPUT_DIR at the top of ruler/analysis.py
#  to point at your result files, then run:)
python ruler/analysis.py
```

Use the same `--samples-dir` across all models so results are directly comparable.

## Repository Structure

```
main/
├── evaluation/
│   ├── run_eval.py              # Primary evaluation driver
│   ├── agentic_offload.py       # BIND: data externalization + repair
│   └── problem_evaluator.py     # Gurobi sandbox executor
├── generation/                  # Problem generators (12 categories)
├── model_registry.py            # Model configs (OpenAI / Anthropic / vLLM)
└── utils.py                     # API client routing

scripts/
├── evaluation/                  # TTC strategies (best-of-k, repair)
├── analysis/                    # Failure mode & isomorphism analysis
└── training/                    # Binding SFT data generation & eval

ruler/
├── eval.py                      # RULER binding task generation & evaluation
├── analysis.py                  # Result analysis & plotting
└── samples/                     # Pre-generated task samples (4 tasks × 6 lengths)
```

## Models Evaluated

| Category | Models |
|----------|--------|
| Frontier | GPT-5, Claude Opus 4.6, Claude Sonnet 4.6 |
| Reasoning | o4-mini, DeepSeek-R1 |
| Standard | GPT-5-Nano, DeepSeek-V3.2 |
| Open-weight | Llama-3.3-70B, Qwen2.5-7B |

<!-- ## Citation

```bibtex
@inproceedings{gao2025orllmbench,
  title={Models Can Model, But Can't Bind: Structured Grounding in Text-to-Optimization},
  author={Gao, Zhiqi and Ge, Albert and Berenbeim, Alexander and Bastian, Nathaniel D. and Sala, Frederic},
  booktitle={Conference on Language Modeling (COLM)},
  year={2025}
}
``` -->

## License

MIT
