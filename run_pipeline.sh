#!/bin/bash
# =============================================================================
# run_pipeline.sh — Full content moderation pipeline (flat structure)
# =============================================================================
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
#
# Override defaults:
#   RAW_DATA=data/train.csv SERVE=false ./run_pipeline.sh
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

log()     { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ $1${NC}"; }
warn()    { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠ $1${NC}"; }
error()   { echo -e "${RED}[$(date '+%H:%M:%S')] ✗ $1${NC}"; exit 1; }
section() { echo -e "\n${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; \
            echo -e "${BOLD}  $1${NC}"; \
            echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"; }

PYTHON=${PYTHON:-python3}
RAW_DATA=${RAW_DATA:-data/train.csv}
PROCESSED_DIR=${PROCESSED_DIR:-data/processed}
OUTPUT_DIR=${OUTPUT_DIR:-outputs}
SERVE=${SERVE:-true}
PORT=${PORT:-8000}

# ---------------------------------------------------------------------------
# 0. Preflight
# ---------------------------------------------------------------------------
section "0 / 5  Preflight checks"

command -v $PYTHON &>/dev/null || error "Python not found."
log "Python: $($PYTHON --version)"
[ -f "$RAW_DATA" ] || error "data/train.csv not found. Download from Kaggle and place it in data/."
success "data/train.csv found"

log "Checking dependencies..."
$PYTHON -c "import transformers, datasets, onnxruntime, fastapi" 2>/dev/null || {
    warn "Missing packages — installing..."
    pip install torch transformers datasets onnx onnxruntime fastapi uvicorn \
        scikit-learn pandas numpy pydantic mlflow accelerate matplotlib seaborn -q
}
success "Dependencies OK"

# ---------------------------------------------------------------------------
# 1. Convert raw data
# ---------------------------------------------------------------------------
section "1 / 5  Convert raw data"

if [ -f "$PROCESSED_DIR/dataset.csv" ]; then
    warn "dataset.csv already exists — skipping. Delete $PROCESSED_DIR/dataset.csv to redo."
else
    log "Converting Jigsaw CSV → binary dataset..."
    $PYTHON convert_datasets.py \
        --input "$RAW_DATA" \
        --output "$PROCESSED_DIR/dataset.csv"
    success "Saved to $PROCESSED_DIR/dataset.csv"
fi

# ---------------------------------------------------------------------------
# 2. Tokenize + split
# ---------------------------------------------------------------------------
section "2 / 5  Tokenize & split"

if [ -d "$PROCESSED_DIR/dataset" ]; then
    warn "Tokenized dataset already exists — skipping."
else
    log "Tokenizing..."
    $PYTHON prepare_data.py \
        --input "$PROCESSED_DIR/dataset.csv" \
        --output "$PROCESSED_DIR"
    success "Tokenized dataset saved"
fi

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
section "3 / 5  Fine-tune DistilBERT"

if [ -d "$OUTPUT_DIR/best_model" ]; then
    warn "Trained model found — skipping. Delete $OUTPUT_DIR/best_model to retrain."
else
    log "Training (20–60 min CPU, ~5 min GPU)..."
    $PYTHON train.py --output "$OUTPUT_DIR"
    success "Training complete → $OUTPUT_DIR/best_model"
fi

# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------
section "4 / 5  Evaluate"

log "Running evaluation on test set..."
$PYTHON evaluate.py \
    --checkpoint "$OUTPUT_DIR/best_model" \
    --dataset    "$PROCESSED_DIR/dataset" \
    --output     "$OUTPUT_DIR/evaluation"
success "Evaluation saved to $OUTPUT_DIR/evaluation/"
echo ""
cat "$OUTPUT_DIR/evaluation/classification_report.txt" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 5. Export ONNX
# ---------------------------------------------------------------------------
section "5 / 5  ONNX export & serve"

if [ -f "$OUTPUT_DIR/model_quantized.onnx" ]; then
    warn "Quantized model already exists — skipping export."
else
    log "Exporting to ONNX + quantizing..."
    $PYTHON onnx_export.py \
        --checkpoint "$OUTPUT_DIR/best_model" \
        --output     "$OUTPUT_DIR"
    success "ONNX export complete"
fi

echo ""
log "Model sizes:"
for f in "$OUTPUT_DIR"/model*.onnx; do
    echo "  $(du -sh $f | cut -f1)  $f"
done

if [ "$SERVE" = "false" ]; then
    echo ""
    success "Pipeline complete. Start server with:"
    echo "  uvicorn app:app --host 0.0.0.0 --port $PORT"
    exit 0
fi

echo ""
log "Starting server on http://localhost:$PORT"
log "Docs: http://localhost:$PORT/docs"
log "Press Ctrl+C to stop."
echo ""
uvicorn app:app --host 0.0.0.0 --port "$PORT" --workers 1 --access-log
