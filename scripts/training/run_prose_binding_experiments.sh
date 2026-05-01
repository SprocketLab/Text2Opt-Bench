#!/bin/bash
set -e

# Prose-embedded description binding experiments
# Categories: Transportation (prose), JSSP (prose)
# Tests whether prose format (randomized order, no table) restores binding specialist advantage.
#
# Usage:
#   nohup bash scripts/training/run_prose_binding_experiments.sh > ~/prose_training.log 2>&1 &

BENCH_ROOT="${BENCH_ROOT:?Set BENCH_ROOT to your OR-LLM-Synthetic-Bench directory}"
LLAMA_ROOT="${LLAMA_ROOT:?Set LLAMA_ROOT to your LLaMA-Factory directory}"
MODELS_ROOT="${MODELS_ROOT:?Set MODELS_ROOT to your trained models directory}"
RESULTS_ROOT=$BENCH_ROOT/results

source $LLAMA_ROOT/.venv/bin/activate

export WANDB_PROJECT="or-llm-binding-sft"
export FORCE_TORCHRUN=1

mkdir -p $RESULTS_ROOT

# ══════════════════════════════════════════════════════════════════════
# PHASE 0: Convert descriptions to prose + generate SFT data
# ══════════════════════════════════════════════════════════════════════
echo "=== Phase 0: Convert to prose and generate SFT data === $(date)"
cd $BENCH_ROOT
python scripts/training/convert_to_prose_descriptions.py
python scripts/training/generate_binding_sft_data_template.py transportation_prose jssp_prose
python scripts/training/generate_e2e_sft_data_template.py transportation_prose jssp_prose
echo "Data generation complete: $(date)"

# ══════════════════════════════════════════════════════════════════════
# PHASE 1: Train binding specialists (prose)
# ══════════════════════════════════════════════════════════════════════
for CATEGORY in transportation jssp; do
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Training binding specialist: ${CATEGORY}_prose"
    echo "══════════════════════════════════════════════════════"
    echo "Started: $(date)"
    cd $LLAMA_ROOT
    torchrun --nproc_per_node=2 src/train.py \
        examples/train_binding/binding_7B_${CATEGORY}_prose.yaml
    echo "Finished binding_${CATEGORY}_prose: $(date)"
done

# ══════════════════════════════════════════════════════════════════════
# PHASE 2: Train e2e SFT baselines (prose)
# ══════════════════════════════════════════════════════════════════════
for CATEGORY in transportation jssp; do
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Training e2e SFT baseline: ${CATEGORY}_prose"
    echo "══════════════════════════════════════════════════════"
    echo "Started: $(date)"
    cd $LLAMA_ROOT
    torchrun --nproc_per_node=2 src/train.py \
        examples/train_binding/e2e_7B_${CATEGORY}_prose.yaml
    echo "Finished e2e_${CATEGORY}_prose: $(date)"
done

# ══════════════════════════════════════════════════════════════════════
# PHASE 3: Evaluate binding specialist (prose)
# ══════════════════════════════════════════════════════════════════════
cd $BENCH_ROOT
for CATEGORY in transportation jssp; do
    BINDER=$MODELS_ROOT/binding_${CATEGORY}_prose
    PROSE_EVAL_DIR=$BENCH_ROOT/synthetic_dataset/Template/basic_template/${CATEGORY}/small_prose

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Evaluating binding specialist (prose): $CATEGORY"
    echo "══════════════════════════════════════════════════════"

    # GT upper bound on prose eval set (confirms template still works)
    echo "=== GT → template (prose, $CATEGORY) === $(date)"
    python scripts/training/eval_two_phase_template.py \
        --category $CATEGORY \
        --ground-truth \
        --phase2 template \
        --eval-dir $PROSE_EVAL_DIR \
        --output-path $RESULTS_ROOT/two_phase_${CATEGORY}_prose_gt_template.json
    echo "Done: $(date)"

    # Binding accuracy
    echo "=== Binding accuracy (prose, $CATEGORY) === $(date)"
    python scripts/training/eval_binding_model_template.py \
        --category $CATEGORY \
        --model-path $BINDER \
        --eval-dir $PROSE_EVAL_DIR \
        --output-path $RESULTS_ROOT/binding_eval_${CATEGORY}_prose.json
    echo "Done: $(date)"

    # Two-phase: trained binder → template
    echo "=== Trained binder → template (prose, $CATEGORY) === $(date)"
    python scripts/training/eval_two_phase_template.py \
        --category $CATEGORY \
        --binding-model $BINDER \
        --phase2 template \
        --eval-dir $PROSE_EVAL_DIR \
        --output-path $RESULTS_ROOT/two_phase_${CATEGORY}_prose_binder_template.json
    echo "Done: $(date)"
done

# ══════════════════════════════════════════════════════════════════════
# PHASE 4: Evaluate e2e SFT baseline (prose)
# ══════════════════════════════════════════════════════════════════════
for CATEGORY in transportation jssp; do
    E2E_MODEL=$MODELS_ROOT/e2e_${CATEGORY}_prose
    PROSE_EVAL_DIR=$BENCH_ROOT/synthetic_dataset/Template/basic_template/${CATEGORY}/small_prose

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Evaluating e2e SFT baseline (prose): $CATEGORY"
    echo "══════════════════════════════════════════════════════"

    echo "=== E2E SFT eval (prose, $CATEGORY) === $(date)"
    python scripts/training/eval_e2e_sft_template.py \
        --category $CATEGORY \
        --model-path $E2E_MODEL \
        --eval-dir $PROSE_EVAL_DIR \
        --output-path $RESULTS_ROOT/e2e_eval_${CATEGORY}_prose.json
    echo "Done: $(date)"
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  ALL PROSE EXPERIMENTS COMPLETE: $(date)"
echo "══════════════════════════════════════════════════════"
echo ""
echo "Results saved to $RESULTS_ROOT/:"
echo "  two_phase_{transportation,jssp}_prose_gt_template.json  — GT upper bound"
echo "  two_phase_{transportation,jssp}_prose_binder_template.json — Binding specialist"
echo "  e2e_eval_{transportation,jssp}_prose.json               — E2E SFT baseline"
echo "  binding_eval_{transportation,jssp}_prose.json           — Binding accuracy detail"
