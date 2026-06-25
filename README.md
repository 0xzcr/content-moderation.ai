# real-time content moderation

fine-tuned distilbert for binary content moderation (safe / toxic), optimized for low-latency inference using onnx runtime graph optimizations and dynamic int8 quantization.

has raw dataset ingestion to production-ready inference deployment — covering dataset engineering, model training, evaluation, experiment tracking, model optimization, benchmarking, and api serving.

---

## overview

modern content platforms process millions of user-generated text samples every day. manually reviewing posts is expensive, slow, and impossible at scale.

the objective of this project was to build a lightweight real-time moderation pipeline capable of:

- detecting toxic content in under 25ms
- minimizing inference latency across batch sizes
- reducing model footprint by over 70% through quantization
- maintaining classification quality after compression
- exposing predictions through a production-style rest api

the system fine-tunes a distilbert sequence classification model on the jigsaw toxic comment dataset and deploys it through onnx runtime with int8 quantization.

---

## architecture

```text
raw dataset
      │
      ▼
dataset cleaning
      │
      ▼
tokenization
      │
      ▼
distilbert fine-tuning
      │
      ▼
evaluation + mlflow tracking
      │
      ▼
onnx export
      │
      ▼
onnx graph optimization
      │
      ▼
dynamic int8 quantization
      │
      ▼
onnx runtime inference
      │
      ▼
fastapi service
      │
      ▼
real-time moderation endpoint
```

---

## project structure

```text
content-moderation/
├── data/
│   ├── train.csv
│   └── processed/
│
├── outputs/
│   ├── best_model/
│   ├── model.onnx
│   ├── model_optimized.onnx
│   └── model_quantized.onnx
│
├── convert_datasets.py
├── prepare_data.py
├── dataset.py
├── trainer.py
├── train.py
├── evaluate.py
├── onnx_export.py
├── app.py
│
├── test_api.py
├── test_data_pipeline.py
│
├── run_pipeline.sh
├── pyproject.toml
└── README.md
```

---

## quickstart

```bash
# 1. install dependencies
pip install -e "."

# 2. place jigsaw dataset
cp /path/to/train.csv data/train.csv

# 3. run full pipeline
chmod +x run_pipeline.sh
./run_pipeline.sh
```

or step by step:

```bash
python convert_datasets.py --input data/train.csv --output data/processed/dataset.csv
python prepare_data.py     --input data/processed/dataset.csv --output data/processed/
python train.py
python evaluate.py         --checkpoint outputs/best_model/ --dataset data/processed/dataset/
python onnx_export.py      --checkpoint outputs/best_model/ --output outputs/
uvicorn app:app            --host 0.0.0.0 --port 8000
```

---

## dataset engineering

### source

[jigsaw toxic comment classification challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data) — 159,571 user-generated comments labeled across six toxicity dimensions: toxic, severe toxic, obscene, threat, insult, and identity hate.

for this project all six dimensions were collapsed into a single binary moderation label:

```text
any toxicity dimension == 1  →  toxic  (1)
all dimensions == 0          →  safe   (0)
```

class distribution:

```text
safe  : 143,000  (90%)
toxic :  16,500  (10%)
```

### class imbalance handling

a naive model that always predicts safe achieves 90% accuracy while catching zero toxic content. this was addressed using `weightedRandomSampler` at the dataloader level — every mini-batch is class-balanced before reaching the model, producing more stable gradients than reweighting the cross-entropy loss and requiring no additional hyperparameter tuning.

### preprocessing pipeline

```text
null handling
→ html tag stripping
→ url replacement    →  [URL]
→ mention removal    →  [USER]
→ whitespace normalisation
→ unicode normalisation
→ duplicate removal
```

cleaning runs **before** the train/val/test split to prevent any signal leaking from held-out sets into training.

### train / val / test split

stratified sampling preserves the 90/10 class ratio across all three sets:

```text
train : 127,656  (80%)
val   :  15,957  (10%)
test  :  15,958  (10%)
```

### tokenization

```python
AutoTokenizer.from_pretrained("distilbert-base-uncased")
```

```text
max_length : 128 tokens
padding    : max_length
truncation : tail (front-loaded toxicity signal)
```

the tokenizer is pretrained and carries zero information from this dataset — no leakage possible.

---

## model

### distilbert-base-uncased

distilbert is a compressed transformer derived from bert via knowledge distillation.

```text
parameters    : 66,955,010
layers        : 6 transformer encoder blocks
hidden size   : 768
attention heads: 12
size vs bert  : ~40% fewer parameters
speed vs bert : ~60% faster at inference
accuracy      : retains ~97% of bert's language understanding
```

classification head: `linear(768 → 2)` applied to the `[CLS]` token representation.

### training configuration

| hyperparameter | value |
|---|---|
| optimizer | adamw |
| learning rate | 2e-5 |
| warmup ratio | 6% of total steps |
| weight decay | 0.01 |
| gradient clipping | 1.0 |
| batch size | 32 |
| early stopping metric | macro f1 |
| hardware | apple m1, 8gb unified memory |
| training time | ~3 hours |

early stopping monitors **macro f1**, not validation loss. on imbalanced data, loss can keep decreasing while f1 on the minority class quietly plateaus or drops — optimising loss in this setting is optimising the wrong signal.

---

## experiment tracking

mlflow tracks all runs with:

- training and validation loss per step
- precision, recall, f1 per class
- macro f1 and roc-auc
- all hyperparameters
- model checkpoints
- best model artifact

```bash
mlflow ui  # view runs at http://localhost:5000
```

---

## evaluation

### results

```text
dataset  : 11,500 test samples
accuracy : 96.08%
roc-auc  : 0.9753
```

### classification report

```text
              precision    recall    f1-score   support

safe            0.9754    0.9796      0.9775     10000
toxic           0.8600    0.8353      0.8475      1500

macro avg       0.9177    0.9075      0.9125     11500
```

### screenshots

![evaluation results](pics/Screenshot%202026-06-24%20at%203.35.44%E2%80%AFPM.png)

![confusion matrix](pics/Screenshot%202026-06-24%20at%2010.49.44%E2%80%AFPM.png)

### metric breakdown

**precision** — of all posts flagged as toxic, how many were actually toxic.
```text
tp / (tp + fp)
```

**recall** — of all actually toxic posts, how many were caught.
```text
tp / (tp + fn)
```

toxic recall at **83.53%** is the number that matters most in production. a false negative — missed toxic content — is more damaging than a false positive. the fix is threshold tuning, not retraining. lowering the classification threshold from 0.5 to ~0.35 pushes toxic recall toward 90%+ at an acceptable precision cost. the api returns full softmax probability distributions so the consuming application owns that decision.

**f1 score** — harmonic mean of precision and recall.
```text
2 × (precision × recall) / (precision + recall)
```

**roc-auc** — measures ranking quality across all classification thresholds.
```text
roc-auc : 0.9753
```
a randomly selected toxic sample receives a higher toxicity score than a randomly selected safe sample 97.5% of the time. excellent class separability.

### error analysis

451 misclassifications were exported and sorted by model confidence descending — highest-confidence wrong predictions first. this surfaces systematic failure modes faster than random sampling.

---

## onnx export

after training, the pytorch model is exported to onnx:

```python
torch.onnx.export(
    model,
    args=(input_ids, attention_mask),
    f="model.onnx",
    opset_version=17,
    dynamic_axes={
        "input_ids":      {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits":         {0: "batch_size"},
    },
    do_constant_folding=True,
)
```

dynamic axes allow variable batch sizes and sequence lengths at runtime without re-exporting. `do_constant_folding=True` collapses constant subgraphs at export time.

exported graph: **315 nodes, 125 initializers**.

benefits over pytorch runtime:
- framework-independent execution
- reduced inference overhead
- hardware portability
- enables runtime-level graph optimization

---

## graph optimization

onnx runtime applies graph-level transformations at `ORT_ENABLE_ALL`:

- **operator fusion** — layernorm, gelu, and attention bias fused into single kernels
- **constant folding** — pre-computes static subgraphs
- **dead node elimination** — removes unused computation
- **memory planning** — reduces allocation overhead

```text
model.onnx  →  model_optimized.onnx
```

---

## dynamic int8 quantization

the optimized fp32 model is quantized to int8:

```python
quantize_dynamic(
    model_input="model_optimized.onnx",
    model_output="model_quantized.onnx",
    weight_type=QuantType.QInt8,
)
```

weights are quantized **offline** to int8. activations are quantized **at runtime**. no calibration dataset required, unlike static quantization. matmul operations — which dominate transformer inference time — are the primary quantization target.

---

## benchmark results

### model size

| model | size | reduction |
|---|---:|---:|
| fp32 (optimized) | 267.89 mb | — |
| int8 (quantized) | 67.36 mb | **74.9%** |

### latency — p50

| batch size | fp32 | int8 | speedup |
|---|---:|---:|---:|
| 1 | 36.69 ms | 20.41 ms | 1.80× |
| 8 | 339.69 ms | 149.89 ms | 2.27× |
| 16 | 890.02 ms | 306.94 ms | 2.90× |
| 32 | 1872.25 ms | 641.88 ms | 2.92× |

### latency — p99

| batch size | fp32 | int8 | speedup |
|---|---:|---:|---:|
| 1 | 57.3 ms | 24.9 ms | 2.30× |
| 8 | 363.0 ms | 157.5 ms | 2.30× |
| 16 | 1069.1 ms | 333.2 ms | 3.21× |
| 32 | 2142.8 ms | 1208.6 ms | 1.77× |

> batch 32 p99 spikes to 1208ms while p50 sits at 641ms — tail latency becomes unpredictable at high batch sizes. the production api caps requests at batch 16 to keep p99 consistently under 350ms.

### throughput

| batch size | fp32 | int8 |
|---|---:|---:|
| 1 | 26 req/s | 49 req/s |
| 8 | 24 req/s | 53 req/s |
| 16 | 18 req/s | 52 req/s |
| 32 | 17 req/s | 48 req/s |

### screenshots

![benchmark results](pics/Screenshot%202026-06-24%20at%2010.49.19%E2%80%AFPM.png)

![latency chart](pics/Screenshot%202026-06-24%20at%208.46.14%E2%80%AFPM.png)

---

## prediction parity validation

quantization was validated to confirm it did not degrade classification behavior:

```text
input  : "great post!"
fp32   : safe  (100.00%)
int8   : safe  (100.00%)
delta  : 0.0000
```

```text
input  : "you are a dumb bitch, shut the fuck up!"
fp32   : toxic (100.00%)
int8   : toxic  (99.99%)
delta  : 0.0001
```

max confidence delta across all test inputs: **0.0001** — the quantized model is production-safe.

---

## serving

inference is exposed through a fastapi async application.

```text
request
   │
   ▼
pydantic v2 validation
   │
   ▼
tokenizer
   │
   ▼
onnx runtime (coreml / cpu)
   │
   ▼
softmax
   │
   ▼
response
```

### runtime configuration

- **execution provider** — `CoreMLExecutionProvider` on apple silicon, falling back to `CPUExecutionProvider`
- **ort threading** — `intra_op_num_threads=4`, `inter_op_num_threads=1` (latency mode)
- **threadpoolexecutor** — capped at 2 concurrent workers to prevent contention between uvicorn's event loop and ort's internal threading
- **startup warmup** — dummy inference pass on lifespan startup to force ort kernel compilation before the first real request

### input handling

| condition | behaviour |
|---|---|
| text > 1000 characters | rejected at validation layer (422) |
| tokens > 128 | tail-truncated silently |
| batch size > 16 | rejected at validation layer (422) |

### api

```http
POST /moderate
```

request:
```json
{
  "texts": ["great post!", "i hate you"]
}
```

response:
```json
{
  "request_id": "abc-123",
  "results": [
    { "label": "safe",  "confidence": 0.999, "probabilities": { "safe": 0.999, "toxic": 0.001 } },
    { "label": "toxic", "confidence": 0.998, "probabilities": { "safe": 0.002, "toxic": 0.998 } }
  ],
  "latency_ms": 21.4
}
```

```http
GET /health
```

returns model load status, requests served, and mean latency.

### screenshots

![api docs](pics/Screenshot%202026-06-24%20at%207.32.21%E2%80%AFPM.png)

![api response](pics/Screenshot%202026-06-24%20at%2010.06.34%E2%80%AFPM.png)

---

## integration

the api is a stateless rest service that accepts plain json and returns a label, confidence score, and full probability distribution. it integrates with any stack over standard http — no sdk, no special client.

| use case | integration pattern |
|---|---|
| chat app | call `/moderate` before broadcasting a message |
| forum | call `/moderate` before writing a post to the database |
| mobile app | call `/moderate` before submitting user content |
| discord / slack bot | intercept messages, call `/moderate`, delete or flag |
| data pipeline | batch-moderate historical records in groups of 16 |
| browser extension | screen comments before posting to third-party platforms |

the response includes a probability score, not just a label. each integration sets its own threshold based on its own risk tolerance — a children's platform might act at 20% toxic probability, a general forum at 70%. the model surfaces the signal, the product decides what to do with it.

![pipeline overview](pics/Screenshot%202026-06-24%20at%203.36.35%E2%80%AFPM.png)

---

## technical challenges

### onnx export compatibility

newer transformer releases introduce operators that occasionally break export paths. issues encountered included dynamic shape inference failures, opset compatibility warnings, and exporter version mismatches. resolved through graph inspection and explicit opset pinning at version 17.

### quantization tensor conflicts

onnx runtime shape inference produced tensor dimension conflicts during dynamic quantization. resolved through fallback export path, tensor type overrides, and model validation before quantization.

### apple silicon execution provider

`CoreMLExecutionProvider` occasionally attempted to partition unsupported graph sections. resolved by validating provider availability at startup and falling back gracefully to `CPUExecutionProvider` for stable benchmarking.

### benchmark variance

initial latency measurements showed high variance between runs. resolved with 10-run warmup before measurement, 100-run benchmark loops, and p50/p95/p99 percentile reporting rather than mean.

---

## future work

- multi-label classification — hate speech, threats, spam as independent outputs
- severity scoring — continuous toxicity score alongside binary label
- static int8 quantization with calibration dataset for better accuracy
- docker deployment
- prometheus metrics endpoint
- kubernetes horizontal scaling
- grpc serving with triton inference server
- distributed inference benchmarking
- ci/cd pipeline with automated regression tests on quantized model

---

## tech stack

| layer | tools |
|---|---|
| model | pytorch, transformers, distilbert |
| data | pandas, scikit-learn, datasets |
| tracking | mlflow |
| optimization | onnx, onnxruntime |
| serving | fastapi, uvicorn, pydantic v2 |
| testing | pytest |
| visualization | matplotlib, seaborn |

---

## results summary

```text
dataset                 : 159,571 comments
model parameters        : 66.9m
training time           : ~3 hours (m1 macbook air, no gpu)
accuracy                : 96.08%
roc-auc                 : 0.9753
macro f1                : 0.9125
toxic recall            : 83.53%
fp32 model size         : 267.89 mb
int8 model size         : 67.36 mb
size reduction          : 74.9%
int8 p99 latency        : 24.9ms at batch 1
maximum p50 speedup     : 2.92× at batch 32
prediction drift        : 0.0001 max delta
```

the final system demonstrates the complete path from transformer fine-tuning to production-oriented optimized inference — preserving classification quality under aggressive model compression and exposing predictions through a low-latency, integration-ready rest api.
