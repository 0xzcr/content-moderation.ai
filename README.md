# real-time content moderation

fine-tuned distilbert for binary content moderation (safe / toxic), optimized for low-latency inference using onnx runtime graph optimizations and dynamic int8 quantization.

this project demonstrates a complete ml lifecycle from raw dataset ingestion to production-ready inference deployment, including dataset engineering, model training, evaluation, experiment tracking, model optimization, benchmarking, and api serving.

---

## overview

modern content platforms process millions of user-generated text samples every day. manually reviewing posts is expensive, slow, and impossible at scale.

the objective of this project was to build a lightweight real-time moderation pipeline capable of:

- detecting toxic content
- minimizing inference latency
- reducing model footprint
- maintaining classification quality after quantization
- exposing predictions through a production-style api

the system uses a fine-tuned distilbert sequence classification model trained on the jigsaw toxic comment dataset and deploys the resulting model through onnx runtime.

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

## dataset engineering

### source dataset

jigsaw toxic comment classification dataset

the original dataset contains user-generated comments labeled across multiple toxicity dimensions.

examples include:

- toxic
- severe toxic
- obscene
- threat
- insult
- identity hate

for this project the labels were consolidated into a binary moderation task:

```text
safe  -> 0
toxic -> 1
```

---

### preprocessing pipeline

before training, the dataset passes through several normalization stages.

#### text cleaning

- null handling
- whitespace normalization
- unicode normalization
- duplicate removal

#### train validation split

stratified sampling is used to preserve class distributions across datasets.

#### tokenization

tokenization is performed using the distilbert tokenizer.

```python
AutoTokenizer.from_pretrained(...)
```

sequence length:

```text
128 tokens
```

longer comments are truncated while shorter comments are padded.

---

## model architecture

### distilbert

distilbert is a compressed transformer architecture derived from bert using knowledge distillation.

advantages:

- ~40% fewer parameters than bert-base
- significantly lower memory consumption
- faster inference
- minimal accuracy degradation

model statistics:

```text
parameters: 66,955,010
architecture: transformer encoder
hidden size: 768
layers: 6
attention heads: 12
```

---

## training pipeline

training is performed using the huggingface trainer api.

core components:

```python
Trainer
TrainingArguments
AutoModelForSequenceClassification
```

training stages:

```text
dataset
→ tokenizer
→ dataloader
→ forward pass
→ cross entropy loss
→ backpropagation
→ optimizer update
→ evaluation
→ checkpointing
```

---

## experiment tracking

mlflow is used for experiment management.

tracked artifacts include:

- training loss
- validation loss
- precision
- recall
- f1 score
- checkpoints
- hyperparameters

this allows reproducible experimentation and model version comparison.

---

## evaluation

multiple evaluation metrics were used instead of relying solely on accuracy.

### confusion matrix

used to analyze:

- false positives
- false negatives
- class imbalance behavior

---

### precision

```text
tp / (tp + fp)
```

measures prediction correctness.

---

### recall

```text
tp / (tp + fn)
```

measures toxic content capture rate.

---

### f1 score

```text
2 × (precision × recall)
-------------------------
 precision + recall
```

balances precision and recall.

---

### roc curve

receiver operating characteristic analysis was used to measure ranking quality across all classification thresholds.

result:

```text
roc auc = 0.975
```

interpretation:

a randomly selected toxic sample receives a higher toxicity score than a randomly selected safe sample approximately 97.5% of the time.

this indicates excellent class separability.

---

## onnx export pipeline

after training, the pytorch model is exported to onnx.

```python
torch.onnx.export(...)
```

benefits:

- framework-independent execution
- reduced inference overhead
- hardware portability
- runtime optimization opportunities

---

## graph optimization

onnx runtime applies graph-level transformations.

examples:

- operator fusion
- dead node elimination
- constant folding
- memory planning improvements

output:

```text
model.onnx
↓
model_optimized.onnx
```

---

## dynamic int8 quantization

the optimized fp32 model is converted to int8.

```python
quantize_dynamic(...)
```

quantized layers:

- linear layers
- matmul operations
- transformer feed-forward projections

objective:

```text
reduce memory bandwidth
reduce model size
improve cpu throughput
```

---

## benchmark results

### model size comparison

| model | size |
|---------|---------:|
| fp32 optimized | 267.89 mb |
| int8 quantized | 67.36 mb |

reduction:

```text
74.9%
```

---

### latency benchmark

| batch size | fp32 p50 | int8 p50 | speedup |
|------------|-----------:|-----------:|---------:|
| 1 | 36.69 ms | 20.41 ms | 1.80x |
| 8 | 339.69 ms | 149.89 ms | 2.27x |
| 16 | 890.02 ms | 306.94 ms | 2.90x |
| 32 | 1872.25 ms | 641.88 ms | 2.92x |

---

### throughput comparison

| batch size | fp32 | int8 |
|------------|-------:|-------:|
| 1 | 26/sec | 49/sec |
| 8 | 24/sec | 53/sec |
| 16 | 18/sec | 52/sec |
| 32 | 17/sec | 48/sec |

---

## prediction parity validation

verification was performed to ensure quantization did not significantly impact classification quality.

example outputs:

```text
input:
"great post!"

fp32:
safe (100%)

int8:
safe (100%)

delta:
0.0000
```

```text
input:
"you are a dumb bitch, shut the fuck up!"

fp32:
toxic (100%)

int8:
toxic (99.99%)

delta:
0.0001
```

the quantized model preserved prediction behavior while achieving significantly higher throughput.

---

## serving layer

inference is exposed through a fastapi application.

pipeline:

```text
request
   │
   ▼
fastapi
   │
   ▼
tokenizer
   │
   ▼
onnx runtime
   │
   ▼
softmax
   │
   ▼
response
```

endpoint:

```http
POST /moderate
```

example request:

```json
{
  "texts": [
    "great post!",
    "i hate you"
  ]
}
```

example response:

```json
[
  {
    "label": "safe",
    "confidence": 0.999
  },
  {
    "label": "toxic",
    "confidence": 0.998
  }
]
```

---

## technical challenges

### onnx export incompatibilities

newer transformer releases introduce operators that occasionally break legacy export paths.

issues encountered:

- dynamic shape inference failures
- exporter version mismatches
- opset compatibility warnings

---

### quantization failures

onnx runtime shape inference produced tensor dimension conflicts during dynamic quantization.

resolution:

- fallback export path
- tensor type overrides
- graph inspection
- model validation before quantization

---

### apple silicon runtime issues

coreml execution provider occasionally attempted unsupported graph partitions.

resolution:

```python
providers=["CPUExecutionProvider"]
```

for stable benchmarking.

---

### benchmark reproducibility

initial measurements showed large variance.

resolution:

- warmup iterations
- repeated benchmark runs
- percentile-based latency reporting

---

## key learnings

through this project i gained hands-on experience with:

- transformer fine-tuning
- dataset engineering
- mlflow experiment tracking
- model evaluation
- onnx export internals
- graph optimization
- quantization techniques
- inference benchmarking
- fastapi deployment
- production inference pipelines

---

## future work

planned improvements:

- multi-label moderation
- hate speech categorization
- threat detection
- profanity scoring
- docker deployment
- ci/cd integration
- prometheus monitoring
- kubernetes deployment
- distributed inference benchmarking
- grpc serving

---

## tech stack

### machine learning

- pytorch
- transformers
- scikit-learn
- numpy

### optimization

- onnx
- onnx runtime
- dynamic int8 quantization

### experiment tracking

- mlflow

### backend

- fastapi
- uvicorn

### testing

- pytest

### visualization

- matplotlib
- seaborn

---

## results summary

```text
roc auc                 : 0.975
model parameters        : 66.9m
fp32 model size         : 267.89 mb
int8 model size         : 67.36 mb
size reduction          : 74.9%

batch-32 fp32 latency   : 1872.25 ms
batch-32 int8 latency   : 641.88 ms

maximum speedup         : 2.92x
```

the final system successfully demonstrates the complete path from transformer training to production-oriented optimized inference while preserving classification quality under aggressive model compression.


## pictures

![pic1](pics/Screenshot%202026-06-24%20at%203.35.44%E2%80%AFPM.png)

![pic2](pics/Screenshot%202026-06-24%20at%2010.49.44%E2%80%AFPM.png)

![pic3](pics/Screenshot%202026-06-24%20at%2010.49.19%E2%80%AFPM.png)

![pic4](pics/Screenshot%202026-06-24%20at%208.46.14%E2%80%AFPM.png)

![pic5](pics/Screenshot%202026-06-24%20at%207.32.21%E2%80%AFPM.png)

![pic6](pics/Screenshot%202026-06-24%20at%2010.06.34%E2%80%AFPM.png)

![pic7](pics/Screenshot%202026-06-24%20at%203.36.35%E2%80%AFPM.png)