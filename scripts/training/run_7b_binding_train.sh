#!/bin/bash
set -e
cd "${LLAMA_ROOT:?Set LLAMA_ROOT to your LLaMA-Factory directory}"
source .venv/bin/activate

export WANDB_PROJECT="or-llm-binding-sft"
export FORCE_TORCHRUN=1

echo "=== Starting 7B binding training ==="
echo "Started: $(date)"
llamafactory-cli train examples/train_binding/binding_7B.yaml
echo "Finished training: $(date)"

echo "=== Running binding eval ==="
cd "${BENCH_ROOT:?Set BENCH_ROOT to your OR-LLM-Synthetic-Bench directory}"
python scripts/training/eval_binding_model.py \
  --model-path "${MODELS_ROOT:?Set MODELS_ROOT to your trained models directory}/binding" \
  --output-path results/binding_eval_7B.json
echo "Finished binding eval: $(date)"

echo "=== Running two-phase: 7B binder → template ==="
python scripts/training/eval_two_phase.py \
  --binding-model "$MODELS_ROOT/binding" \
  --phase2 template \
  --output-path results/two_phase_7B_binder_template.json
echo "Finished Row 3 template: $(date)"

echo "=== Running two-phase: 7B binder → 7B solver ==="
python scripts/training/eval_two_phase.py \
  --binding-model "$MODELS_ROOT/binding" \
  --phase2 model \
  --phase2-model Qwen/Qwen2.5-7B-Instruct \
  --output-path results/two_phase_7B_binder_7B_solver.json
echo "Finished Row 3 model: $(date)"

echo "=== Ground-truth binding → template (upper bound) ==="
echo "Started: $(date)"
python scripts/training/eval_two_phase.py \
  --ground-truth \
  --phase2 template \
  --output-path results/two_phase_ground_truth_template.json
echo "Finished GT template: $(date)"

echo "=== Ground-truth binding → 7B solver ==="
echo "Started: $(date)"
python scripts/training/eval_two_phase.py \
  --ground-truth \
  --phase2 model \
  --phase2-model Qwen/Qwen2.5-7B-Instruct \
  --output-path results/two_phase_ground_truth_7B_solver.json
echo "Finished GT 7B solver: $(date)"

echo "=== ALL COMPLETE ==="
echo "Finished: $(date)"
