#!/bin/bash

# SYNAPSE Ablation Study — Robust Pipeline
# Runs all modality x method combinations, continues on failure

cd "$(dirname "$0")/synapse-src"

# Prevent memory fragmentation on 11GB GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ---- Configuration ----
GPU=0
N_EPOCHS=50
BATCH_SIZE_CONCAT=2     # concat is cheap (mean-pooling first)
BATCH_SIZE_G=1           # G/MAM_G are expensive (full sequence aggregation)
MAX_SEQ_LEN_AGG=1024     # Max tokens per modality (prevents OOM for G/MAM_G)

# ---- Combinations ----
# Multi-modal combinations (most meaningful for the ablation table)
COMBINATIONS=("CT_WSI_Clinical" "CT_WSI" "CT_Clinical" "WSI_Clinical")
# Single-modality baselines
SINGLE_COMBINATIONS=("CT" "WSI" "Clinical")
ALL_COMBINATIONS=("${COMBINATIONS[@]}" "${SINGLE_COMBINATIONS[@]}")

METHODS=("concat" "G" "MAM_G")

FAILED=()
SKIPPED=()
COMPLETED=()

echo "============================================================================"
echo "SYNAPSE Ablation Study — $(date)"
echo "GPU: $GPU | Epochs: $N_EPOCHS | Max Seq Len: $MAX_SEQ_LEN_AGG"
echo "Combinations: ${ALL_COMBINATIONS[*]}"
echo "Methods: ${METHODS[*]}"
echo "============================================================================"
echo ""

for mode in "${ALL_COMBINATIONS[@]}"; do
    for method in "${METHODS[@]}"; do
        COMBO_NAME="${mode}_${method}"
        
        # Pick batch size based on method
        if [ "$method" == "concat" ]; then
            BS=$BATCH_SIZE_CONCAT
        else
            BS=$BATCH_SIZE_G
        fi

        echo "=========================================================================="
        echo "[$COMBO_NAME] Starting | batch_size=$BS | $(date)"
        echo "=========================================================================="
        
        BEST_SAVE_DIR=""
        OUT_LOG="ablation_${COMBO_NAME}_temp.log"
    
        # ---- Training Phase ----
        if python train_ddp.py \
            --ablation_mode "$mode" \
            --integration_method "$method" \
            --max_seq_len_agg $MAX_SEQ_LEN_AGG \
            --gpu $GPU \
            --batch_size $BS \
            --n_epochs $N_EPOCHS \
            --loss_point "Last" \
            --lr 0.0001 \
            --save_best 2>&1 | tee "$OUT_LOG"; then
    
            # Extract the save directory from the log
            BEST_SAVE_DIR=$(grep "FINAL_SAVE_DIR:" "$OUT_LOG" | awk '{print $2}')
            rm -f "$OUT_LOG"
    
            if [ -n "$BEST_SAVE_DIR" ]; then
                echo ""
                echo "[$COMBO_NAME] Training complete. Starting Test Phase using: $BEST_SAVE_DIR"
                echo ""
                
                # ---- Test Phase ----
                if python test_ddp.py \
                    --ablation_mode "$mode" \
                    --integration_method "$method" \
                    --max_seq_len_agg $MAX_SEQ_LEN_AGG \
                    --gpu $GPU \
                    --batch_size 1 \
                    --test_pth "${BEST_SAVE_DIR}/checkpoint_best.pt" 2>&1; then
                    echo ""
                    echo "[$COMBO_NAME] COMPLETED SUCCESSFULLY"
                    COMPLETED+=("$COMBO_NAME")
                else
                    echo ""
                    echo "[$COMBO_NAME] WARNING: Test phase failed, but training was successful."
                    FAILED+=("${COMBO_NAME}_TEST")
                fi
            else
                echo ""
                echo "[$COMBO_NAME] WARNING: Could not find FINAL_SAVE_DIR. Skipping test phase."
                SKIPPED+=("$COMBO_NAME")
            fi
        else
            echo ""
            echo "[$COMBO_NAME] ERROR: Training failed. Continuing to next combination."
            rm -f "$OUT_LOG"
            FAILED+=("${COMBO_NAME}_TRAIN")
        fi
            
        echo ""
        echo "=========================================================================="
        echo ""
        
        # Brief pause to let GPU memory settle
        sleep 2
    done
    echo ""
done

# ---- Summary ----
echo ""
echo "============================================================================"
echo "ABLATION STUDY COMPLETE — $(date)"
echo "============================================================================"
echo ""
echo "COMPLETED (${#COMPLETED[@]}):"
for c in "${COMPLETED[@]}"; do echo "  ✓ $c"; done
echo ""
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do echo "  ✗ $f"; done
    echo ""
fi
if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "SKIPPED (${#SKIPPED[@]}):"
    for s in "${SKIPPED[@]}"; do echo "  ⊘ $s"; done
    echo ""
fi
echo "============================================================================"
