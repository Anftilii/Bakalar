#!/bin/bash
# run_all.sh
# ─────────────────────────────────────────────────────
# Trénuje všetky modely na všetkých datasetoch postupne.
# Spustenie: bash run_all.sh
# Iba jeden model: bash run_all.sh resnet50
# ─────────────────────────────────────────────────────

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

MODELS=("resnet50" "densenet121" "vgg19" "inceptionv3")
if [ -n "$1" ]; then
    MODELS=("$1")
fi

DATASETS=("xray" "mri" "skin")
# Pridaj "histo" ak treba pretrénovať

echo "======================================================"
echo "  ŠTART: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Modely:   ${MODELS[*]}"
echo "  Datasety: ${DATASETS[*]}"
echo "======================================================"

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo ""
        echo "── $MODEL / $DATASET ── $(date '+%H:%M:%S')"
        python "$SCRIPT_DIR/train.py" \
            --dataset "$DATASET" \
            --model   "$MODEL" \
            2>&1 | tee "$LOG_DIR/${DATASET}_${MODEL}.log"
        echo "── $MODEL / $DATASET HOTOVO: $(date '+%H:%M:%S')"
    done
done

echo ""
echo "======================================================"
echo "  VŠETKO HOTOVO: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"