#!/bin/bash
set -e
cd "${BENCH_ROOT:?Set BENCH_ROOT to your OR-LLM-Synthetic-Bench directory}"
source "${LLAMA_ROOT:?Set LLAMA_ROOT to your LLaMA-Factory directory}/.venv/bin/activate"

echo "=== Row 1: Trained 1.5B binder → Template ==="
echo "Started: $(date)"
python scripts/training/eval_two_phase.py \
  --phase2 template \
  --output-path results/two_phase_template_eval.json
echo "Finished Row 1: $(date)"

echo "=== Row 2: Trained 1.5B binder → Untrained 7B solver ==="
echo "Started: $(date)"
python scripts/training/eval_two_phase.py \
  --phase2 model \
  --phase2-model Qwen/Qwen2.5-7B-Instruct \
  --output-path results/two_phase_1.5B_binder_7B_solver.json
echo "Finished Row 2: $(date)"

echo "=== Row 5: Untrained 7B → Untrained 7B ==="
echo "Started: $(date)"
python scripts/training/eval_two_phase.py \
  --binding-model Qwen/Qwen2.5-7B-Instruct \
  --phase2 model \
  --phase2-model Qwen/Qwen2.5-7B-Instruct \
  --output-path results/two_phase_untrained_7B_7B.json
echo "Finished Row 5: $(date)"

echo "=== Binding-only eval ==="
echo "Started: $(date)"
python scripts/training/eval_binding_model.py \
  --output-path results/binding_eval_1.5B_full.json
echo "Finished binding eval: $(date)"

echo "=== ALL COMPLETE ==="
echo "Finished: $(date)"
