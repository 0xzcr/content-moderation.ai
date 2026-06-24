# Real-Time Content Moderation

Fine-tuned DistilBERT for binary content moderation (safe / toxic), optimized with ONNX for real-time inference.

## Project Structure

```
content-moderation/
├── data/
│   ├── train.csv              ← place Jigsaw CSV here
│   └── processed/             ← auto-generated
├── outputs/                   ← auto-generated (model checkpoints, ONNX files)
│
├── dataset.py                 ← data cleaning, tokenization, splitting
├── trainer.py                 ← DistilBERT fine-tuning
├── evaluate.py                ← metrics, confusion matrix, error analysis
├── onnx_export.py             ← ONNX export + INT8 quantization
├── app.py                     ← FastAPI inference server
│
├── convert_datasets.py        ← step 1: raw CSV → dataset.csv
├── prepare_data.py            ← step 2: tokenize + split
├── train.py                   ← step 3: fine-tune
├── evaluate.py                ← step 4: evaluate
├── onnx_export.py             ← step 5: export + benchmark
│
├── test_data_pipeline.py      ← pytest tests for data pipeline
├── test_api.py                ← pytest tests for API
│
├── run_pipeline.sh            ← runs everything end to end
├── pyproject.toml             ← dependencies
└── .gitignore
```

## Setup

```bash
# 1. Install dependencies
pip install -e "."

# 2. Place your data
cp /path/to/jigsaw/train.csv data/train.csv

# 3. Run the full pipeline
chmod +x run_pipeline.sh
./run_pipeline.sh
```

## Step by Step

```bash
python convert_datasets.py --input data/train.csv --output data/processed/dataset.csv
python prepare_data.py     --input data/processed/dataset.csv --output data/processed/
python train.py
python evaluate.py         --checkpoint outputs/best_model/ --dataset data/processed/dataset/
python onnx_export.py      --checkpoint outputs/best_model/ --output outputs/
uvicorn app:app            --host 0.0.0.0 --port 8000
```

## API

```bash
# Classify posts
curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great post!", "I hate you!!"]}'

# Health check
curl http://localhost:8000/health
```

## Classes

| ID | Label | Description |
|---|---|---|
| 0 | safe  | Normal content |
| 1 | toxic | Hate speech, harassment, threats, insults |

## Performance Targets

| Metric | Target |
|---|---|
| Macro F1 | ≥ 0.85 |
| Toxic recall | ≥ 0.90 |
| p99 latency | < 50ms |
| ONNX model size | < 100MB |
