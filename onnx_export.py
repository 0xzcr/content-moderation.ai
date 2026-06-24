"""
Script to convert to ONNX format
Steps:
  1. Export PyTorch model to ONNX
  2. Apply ORT graph optimizations
  3. Apply dynamic INT8 quantization
  4. Benchmark latency / throughput
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from onnx import TensorProto
logger = logging.getLogger(__name__)


def export_to_onnx(
    model_path: str | Path,
    onnx_path: str | Path,
    opset_version: int = 18,
    max_length: int = 128,
) -> Path:
    model_path = Path(model_path)
    onnx_path  = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    #tokenizer = AutoTokenizer.from_pretrained(model_path) #<--- not being used rn
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    print(model.__class__)
    print(sum(p.numel() for p in model.parameters()))
    model.eval()

    dummy_input_ids      = torch.ones(1, max_length, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long)

    logger.info(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        model,
        args=(dummy_input_ids, dummy_attention_mask),
        f=str(onnx_path),
        opset_version=opset_version,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits":         {0: "batch_size"},
        },
        do_constant_folding=True,
        dynamo=False,
    )

    # onnx.checker.check_model(onnx.load(str(onnx_path)))
    # logger.info(f"ONNX model saved → {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")
    
    
    # print("PyTorch params:", sum(p.numel() for p in model.parameters()))

    # onnx_model = onnx.load(str(onnx_path))

    # print("ONNX nodes:", len(onnx_model.graph.node))

    # print("ONNX initializers:", len(onnx_model.graph.initializer))

    # print("ONNX size:", onnx_path.stat().st_size / 1e6, "MB")

def optimize_onnx(onnx_path: str | Path, optimized_path: str | Path) -> Path:
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = str(optimized_path)
    ort.InferenceSession(str(onnx_path), sess_options=sess_options)
    logger.info(f"Optimized model saved → {optimized_path}")
    return Path(optimized_path)


def quantize_onnx(onnx_path: str | Path, quantized_path: str | Path) -> Path:
    logger.info("Applying INT8 dynamic quantization...")
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(quantized_path),
        weight_type=QuantType.QUInt8,
        extra_options={
            "DefaultTensorType": TensorProto.FLOAT
        }
    )
    size_before = Path(onnx_path).stat().st_size / 1e6
    size_after  = Path(quantized_path).stat().st_size / 1e6
    logger.info(f"Quantization: {size_before:.1f} MB → {size_after:.1f} MB ({(1-size_after/size_before)*100:.0f}% reduction)")
    return Path(quantized_path)


class ONNXInferenceSession:
    def __init__(self, model_path: str | Path, tokenizer_path: str | Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1
        # self.session = ort.InferenceSession(
        #     str(model_path),
        #     sess_options=sess_options,
        #     providers=["CUDAExecutionProvider", "CPUExecutionProvider"],<---hardcode CUDA
        # )
        #providers=ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options = sess_options,
            providers=providers
        )
        self._label_names = ["safe", "toxic"]

    def predict(self, texts: list[str], max_length: int = 128) -> list[dict]:
        encoded = self.tokenizer(
            texts, max_length=max_length, padding="max_length",
            truncation=True, return_tensors="np",
        )
        outputs = self.session.run(
            ["logits"],
            {
                "input_ids":      encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            },
        )
        logits   = outputs[0]
        probs    = _softmax(logits)
        pred_ids = probs.argmax(axis=1)
        return [
            {
                "label":         self._label_names[int(pid)],
                "label_id":      int(pid),
                "confidence":    float(probs[i, pid]),
                "probabilities": {n: float(probs[i, j]) for j, n in enumerate(self._label_names)},
            }
            for i, pid in enumerate(pred_ids)
        ]


def benchmark(
    session: ONNXInferenceSession,
    sample_texts: list[str],
    batch_sizes: list[int] | None = None,
    n_runs: int = 100,
    warmup_runs: int = 10,
) -> dict:
    if batch_sizes is None:
        batch_sizes = [1, 8, 16, 32]
    results = {}
    text_pool = (sample_texts * 100)[:max(batch_sizes)]

    for bs in batch_sizes:
        batch = text_pool[:bs]
        for _ in range(warmup_runs):
            session.predict(batch)
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            session.predict(batch)
            latencies.append((time.perf_counter() - t0) * 1000)
        lat = np.array(latencies)
        results[f"batch_{bs}"] = {
            "batch_size": bs,
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99)),
            "throughput_per_sec": round(bs / (np.mean(lat) / 1000)),
        }
        logger.info(f"Batch {bs:>2}: p50={results[f'batch_{bs}']['p50_ms']:.1f}ms | p99={results[f'batch_{bs}']['p99_ms']:.1f}ms | {results[f'batch_{bs}']['throughput_per_sec']}/s")
    return results


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def run_export_pipeline(model_path: str | Path, output_dir: str | Path, opset_version: int = 17, max_length: int = 128) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths = {
        "onnx":      output_dir / "model.onnx",
        "optimized": output_dir / "model_optimized.onnx",
        "quantized": output_dir / "model_quantized.onnx",
    }
    export_to_onnx(model_path, paths["onnx"], opset_version, max_length)
    optimize_onnx(paths["onnx"], paths["optimized"])
    quantize_onnx(paths["onnx"], paths["quantized"])
    return paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/best_model/")
    parser.add_argument("--output",     default="outputs/")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    paths = run_export_pipeline(args.checkpoint, args.output, max_length=args.max_length)

    session_fp32 = ONNXInferenceSession(paths["optimized"], args.checkpoint)

    fp32 = benchmark(session_fp32, [
        "This is a completely normal post.",
        "I hate you so much, you are disgusting!!!",
        "You are a stupid fuck for real, get lost bitch."
    ])

    # adding an INT8 session for comparison
    session_int8 = ONNXInferenceSession(paths["quantized"], args.checkpoint)

    int8 = benchmark(session_int8, [
        "This is a completely normal post.",
        "I hate you so much, you are disgusting!!!",
        "You are a stupid fuck for real, get lost bitch."
    ])

    test_texts = [
        "Great post!",
        "You are a dumb bitch, shut the fuck up!"
    ]

    preds_fp32 = session_fp32.predict(test_texts)
    preds_int8 = session_int8.predict(test_texts)

    for p in preds_fp32:
        print(f"  [FP32 {p['label'].upper():<6} {p['confidence']:.0%}]")

    for p in preds_int8:
        print(f"  [INT8 {p['label'].upper():<6} {p['confidence']:.0%}]")

    print("\n" + "=" * 60)
    print("MODEL SIZE COMPARISON")
    print("=" * 60)

    fp32_size = paths["optimized"].stat().st_size / 1e6
    int8_size = paths["quantized"].stat().st_size / 1e6

    print(f"FP32 Model : {fp32_size:.2f} MB")
    print(f"INT8 Model : {int8_size:.2f} MB")
    print(f"Reduction  : {(1 - int8_size / fp32_size) * 100:.1f}%")

    print("\n" + "=" * 60)
    print("LATENCY COMPARISON")
    print("=" * 60)

    for batch in fp32:
        fp32_p50 = fp32[batch]["p50_ms"]
        int8_p50 = int8[batch]["p50_ms"]

        print(
            f"{batch}: "
            f"FP32={fp32_p50:.2f}ms | "
            f"INT8={int8_p50:.2f}ms | "
            f"{fp32_p50 / int8_p50:.2f}x speedup"
        )

    print("\n" + "=" * 60)
    print("PREDICTION COMPARISON")
    print("=" * 60)

    for i, text in enumerate(test_texts):
        print(f"\nText: {text}")

        print(
            f"FP32 -> {preds_fp32[i]['label'].upper()} "
            f"({preds_fp32[i]['confidence']:.2%})"
        )

        print(
            f"INT8 -> {preds_int8[i]['label'].upper()} "
            f"({preds_int8[i]['confidence']:.2%})"
        )

        print(
            f"Confidence Delta: "
            f"{abs(preds_fp32[i]['confidence'] - preds_int8[i]['confidence']):.4f}"
        )