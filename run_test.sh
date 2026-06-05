#!/bin/bash
# run_test.sh — rýchla kontrola (2 foldy, 2 epochy)
# Spustenie: bash run_test.sh

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "======================================================"
echo "  TEST: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

for MODEL in "resnet50" "densenet121" "vgg19" "inceptionv3"; do
    for DATASET in "xray" "mri" "skin"; do
        echo "── $MODEL / $DATASET"
        python "$SCRIPT_DIR/train.py" \
            --dataset "$DATASET" --model "$MODEL" \
            --epochs 2 --splits 2 \
            2>&1 | tee "$LOG_DIR/test_${DATASET}_${MODEL}.log"
    done
done

echo "======================================================"
echo "  TEST HOTOVO: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"