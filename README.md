# Text2Opt-Bench

A scalable benchmark for evaluating LLMs on operations research optimization problems (LP, MILP, MIQP, nonlinear) with solver-verified ground truth.

**Paper:** [Models Can Model, But Can't Bind: Structured Grounding in Text-to-Optimization](https://arxiv.org/abs/2605.21751) (arXiv:2605.21751)

## Setup

Requires Python ≥ 3.10.

```bash
# Clone and install
git clone https://github.com/SprocketLab/Text2Opt-Bench.git
cd Text2Opt-Bench
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

`requirements.txt` covers the full evaluation and generation pipeline (`gurobipy`, `openai`, `tiktoken`, `tqdm`, `huggingface_hub`, plus `numpy`/`pandas`/`matplotlib`/`seaborn` for analysis). To run open-weight models locally, additionally install `vllm` (commented out in `requirements.txt`):

```bash
pip install vllm>=0.4
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

The **main benchmark** — `Template/` (12 categories, 550 problems) and `Unstructured/resource_allocation/` (1,441 problems) — is bundled in this repo. These are the evaluation sets behind the headline results in the paper, and no download is needed to reproduce them:

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
└── Unstructured/
    └── resource_allocation/     # 1,441 LP/MILP instances (prose-embedded data)
```

Each problem JSON contains: `LLM_description`, `instance_data`, `gold_solution`, `gurobi_result`, `variables`, `constraints`.

### Optional: supplementary assets on Hugging Face

Three supplementary assets back the auxiliary experiments in the paper — they are not needed for the main results, and live on the Hub at [`ZhiqiGao/Text2Opt-Bench`](https://huggingface.co/datasets/ZhiqiGao/Text2Opt-Bench):

- `Template_train/` — SFT training corpus for the binding-specialist study
- `Template_large/` — large-tier instances (7K–48K data tokens) for the binding stress test
- `ruler/samples/` — pre-generated long-context retrieval/aggregation tasks (RULER)

Fetch as needed:

```bash
pip install huggingface_hub
python scripts/download_data.py                  # all three bulk assets (~600 MB)
python scripts/download_data.py --only train     # Template_train (SFT data)
python scripts/download_data.py --only large     # Template_large (7K-48K-token stress instances)
python scripts/download_data.py --only ruler     # RULER long-context tasks
```

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

## Generating more problems

The benchmark is built by a forward-engineering pipeline: each generator instantiates a Gurobi model with sampled size parameters, solves it (retrying on infeasibility, timeout, or trivial-solution cases), then asks an LLM to fill a category-specific template prompt to produce the natural-language `LLM_description`. `scripts/generate.py` exposes the full pipeline through a single CLI:

```bash
python scripts/generate.py --list                                # supported categories
python scripts/generate.py jssp -n 50                            # 50 JSSP problems with sampled sizes
python scripts/generate.py resource_allocation -n 100            # generic LP/MILP
python scripts/generate.py transportation -n 30 --size large \
    --output-dir synthetic_dataset/Template_large/transportation/large
python scripts/generate.py jssp -n 5 --params n_jobs=5 n_machines=4   # fixed sizes
python scripts/generate.py jssp -n 5 --no-description            # structured data only
```

Notes:

- The description step requires `OPENAI_API_KEY` (env var or `api_keys/keys.json`) and defaults to `gpt-5`. Override with `--model`.
- Quality checks are baked in: each generator solves with Gurobi, requires `OPTIMAL` status, and (for the abstract LP/MILP categories) filters trivial-objective solutions, retrying up to 20×.
- Per-category prompts live as `SYSTEM_PROMPT` class attributes under [`main/generation/`](main/generation/) — open the relevant generator file to inspect or modify the prompt.

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
├── generate.py                  # Universal problem generator (forward-engineering pipeline)
├── download_data.py             # Fetch supplementary assets from the HF Hub
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

## Citation

```bibtex
@misc{gao2026modelsmodelcantbind,
      title={Models Can Model, But Can't Bind: Structured Grounding in Text-to-Optimization}, 
      author={Zhiqi Gao and Albert Ge and Alexander Berenbeim and Nathaniel D. Bastian and Frederic Sala},
      year={2026},
      eprint={2605.21751},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.21751}, 
}
```

## License

MIT
