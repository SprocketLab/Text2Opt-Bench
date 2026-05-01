#!/bin/bash
set -e

# Multi-category binding specialist training experiments
# Categories: Transportation, JSSP
# Protocol: binding specialist vs end-to-end SFT
#
# Usage:
#   nohup bash scripts/training/run_template_binding_experiments.sh > ~/template_binding.log 2>&1 &

BENCH_ROOT="${BENCH_ROOT:?Set BENCH_ROOT to your OR-LLM-Synthetic-Bench directory}"
LLAMA_ROOT="${LLAMA_ROOT:?Set LLAMA_ROOT to your LLaMA-Factory directory}"
MODELS_ROOT="${MODELS_ROOT:?Set MODELS_ROOT to your trained models directory}"
RESULTS_ROOT=$BENCH_ROOT/results

# Activate venv
source $LLAMA_ROOT/.venv/bin/activate

export WANDB_PROJECT="or-llm-binding-sft"
export FORCE_TORCHRUN=1

mkdir -p $RESULTS_ROOT

# ══════════════════════════════════════════════════════════════════════
# PHASE 0: Generate SFT data (if not already done)
# ══════════════════════════════════════════════════════════════════════
echo "=== Phase 0: Generate SFT data ==="
echo "Started: $(date)"
cd $BENCH_ROOT
python scripts/training/generate_binding_sft_data_template.py
python scripts/training/generate_e2e_sft_data_template.py
echo "Data generation complete: $(date)"

# ══════════════════════════════════════════════════════════════════════
# PHASE 1: Ground-truth upper bounds (no training needed)
# Validates that template code is correct before investing in training.
# ══════════════════════════════════════════════════════════════════════
for CATEGORY in transportation jssp; do
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  GT upper bound: $CATEGORY"
    echo "══════════════════════════════════════════════════════"

    # GT → template code
    echo "=== GT binding → template code ($CATEGORY) ==="
    echo "Started: $(date)"
    python scripts/training/eval_two_phase_template.py \
        --category $CATEGORY \
        --ground-truth \
        --phase2 template \
        --output-path $RESULTS_ROOT/two_phase_${CATEGORY}_gt_template.json
    echo "Finished: $(date)"

    # GT → untrained 7B
    echo "=== GT binding → untrained 7B ($CATEGORY) ==="
    echo "Started: $(date)"
    python scripts/training/eval_two_phase_template.py \
        --category $CATEGORY \
        --ground-truth \
        --phase2 model \
        --phase2-model Qwen/Qwen2.5-7B-Instruct \
        --output-path $RESULTS_ROOT/two_phase_${CATEGORY}_gt_model.json
    echo "Finished: $(date)"
done

echo ""
echo "=== GT upper bounds complete. Check results before proceeding to training. ==="
echo ""

# ══════════════════════════════════════════════════════════════════════
# PHASE 2: Train binding specialists (7B)
# ══════════════════════════════════════════════════════════════════════
cd $LLAMA_ROOT

echo "=== Training binding specialist: transportation ==="
echo "Started: $(date)"
llamafactory-cli train examples/train_binding/binding_7B_transportation.yaml
echo "Finished: $(date)"

echo "=== Training binding specialist: JSSP ==="
echo "Started: $(date)"
llamafactory-cli train examples/train_binding/binding_7B_jssp.yaml
echo "Finished: $(date)"

# ══════════════════════════════════════════════════════════════════════
# PHASE 3: Train end-to-end SFT baselines (7B)
# ══════════════════════════════════════════════════════════════════════
echo "=== Training e2e SFT: transportation ==="
echo "Started: $(date)"
llamafactory-cli train examples/train_binding/e2e_7B_transportation.yaml
echo "Finished: $(date)"

echo "=== Training e2e SFT: JSSP ==="
echo "Started: $(date)"
llamafactory-cli train examples/train_binding/e2e_7B_jssp.yaml
echo "Finished: $(date)"

# ══════════════════════════════════════════════════════════════════════
# PHASE 4: Evaluate binding specialists
# ══════════════════════════════════════════════════════════════════════
cd $BENCH_ROOT

for CATEGORY in transportation jssp; do
    BINDER=$MODELS_ROOT/binding_${CATEGORY}

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Evaluating binding specialist: $CATEGORY"
    echo "══════════════════════════════════════════════════════"

    # Binding accuracy
    echo "=== Binding accuracy eval ($CATEGORY) ==="
    echo "Started: $(date)"
    python scripts/training/eval_binding_model_template.py \
        --category $CATEGORY \
        --model-path $BINDER \
        --output-path $RESULTS_ROOT/binding_eval_${CATEGORY}.json
    echo "Finished: $(date)"

    # Two-phase: trained binder → template code
    echo "=== Trained binder → template ($CATEGORY) ==="
    echo "Started: $(date)"
    python scripts/training/eval_two_phase_template.py \
        --category $CATEGORY \
        --binding-model $BINDER \
        --phase2 template \
        --output-path $RESULTS_ROOT/two_phase_${CATEGORY}_binder_template.json
    echo "Finished: $(date)"

    # Two-phase: trained binder → untrained 7B
    echo "=== Trained binder → untrained 7B ($CATEGORY) ==="
    echo "Started: $(date)"
    python scripts/training/eval_two_phase_template.py \
        --category $CATEGORY \
        --binding-model $BINDER \
        --phase2 model \
        --phase2-model Qwen/Qwen2.5-7B-Instruct \
        --output-path $RESULTS_ROOT/two_phase_${CATEGORY}_binder_model.json
    echo "Finished: $(date)"
done

# ══════════════════════════════════════════════════════════════════════
# PHASE 5: Evaluate end-to-end SFT baselines
# Note: e2e models generate code directly, no two-phase pipeline.
# We evaluate them using the standard code evaluation pipeline.
# ══════════════════════════════════════════════════════════════════════
for CATEGORY in transportation jssp; do
    E2E_MODEL=$MODELS_ROOT/e2e_${CATEGORY}
    EVAL_DIR=$BENCH_ROOT/synthetic_dataset/Template/basic_template/${CATEGORY}/small

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Evaluating e2e SFT baseline: $CATEGORY"
    echo "══════════════════════════════════════════════════════"

    echo "=== E2E SFT eval ($CATEGORY) ==="
    echo "Started: $(date)"
    python scripts/training/eval_e2e_sft_template.py \
        --category $CATEGORY \
        --model-path $E2E_MODEL \
        --eval-dir $EVAL_DIR \
        --output-path $RESULTS_ROOT/e2e_eval_${CATEGORY}.json
    echo "Finished: $(date)"
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  $(date)"
echo "══════════════════════════════════════════════════════"
echo ""
echo "Results in: $RESULTS_ROOT"
echo "  two_phase_{transportation,jssp}_gt_template.json     — GT upper bounds"
echo "  two_phase_{transportation,jssp}_gt_model.json        — GT + untrained LLM"
echo "  two_phase_{transportation,jssp}_binder_template.json — Binding specialist"
echo "  two_phase_{transportation,jssp}_binder_model.json    — Binding specialist + LLM"
echo "  e2e_eval_{transportation,jssp}.json                  — End-to-end SFT baseline"
echo "  binding_eval_{transportation,jssp}.json              — Binding accuracy detail"
