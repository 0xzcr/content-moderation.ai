"""
FastAPI inference server — binary content moderation (safe / toxic).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}',
)
logger = logging.getLogger(__name__)

LABEL_NAMES    = ["safe", "toxic"]
MAX_TEXT_LEN   = 1000
MAX_BATCH_SIZE = 32
MODEL_PATH     = "outputs/model_quantized.onnx"
TOKENIZER_PATH = "outputs/best_model"
MAX_SEQ_LEN    = 128


class ModelState:
    session: ort.InferenceSession
    tokenizer: Any
    request_count: int = 0
    total_latency_ms: float = 0.0


state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ONNX model...")
    t0 = time.perf_counter()
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    state.session   = ort.InferenceSession(MODEL_PATH, sess_options=sess_options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    state.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    logger.info(f"Model loaded in {(time.perf_counter()-t0)*1000:.0f}ms")
    yield
    del state.session


app = FastAPI(title="Content Moderation API", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ModerationRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)
    request_id: str  = Field(default_factory=lambda: str(uuid.uuid4()))

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts: list[str]) -> list[str]:
        for i, t in enumerate(texts):
            if not isinstance(t, str) or not t.strip():
                raise ValueError(f"texts[{i}] is empty")
            if len(t) > MAX_TEXT_LEN:
                raise ValueError(f"texts[{i}] exceeds {MAX_TEXT_LEN} characters")
        return texts


class PredictionResult(BaseModel):
    label:         str
    label_id:      int
    confidence:    float
    probabilities: dict[str, float]


class ModerationResponse(BaseModel):
    request_id: str
    results:    list[PredictionResult]
    latency_ms: float


class HealthResponse(BaseModel):
    status:           str
    model_loaded:     bool
    requests_served:  int
    mean_latency_ms:  float


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _run_inference(texts: list[str]) -> list[PredictionResult]:
    encoded = state.tokenizer(texts, max_length=MAX_SEQ_LEN, padding="max_length", truncation=True, return_tensors="np")
    outputs = state.session.run(["logits"], {
        "input_ids":      encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    })
    probs    = _softmax(outputs[0])
    pred_ids = probs.argmax(axis=1)
    return [
        PredictionResult(
            label=LABEL_NAMES[int(pid)],
            label_id=int(pid),
            confidence=round(float(probs[i, pid]), 4),
            probabilities={n: round(float(probs[i, j]), 4) for j, n in enumerate(LABEL_NAMES)},
        )
        for i, pid in enumerate(pred_ids)
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/moderate", response_model=ModerationResponse)
async def moderate(request: ModerationRequest) -> ModerationResponse:
    t0   = time.perf_counter()
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(None, _run_inference, request.texts)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")

    latency_ms = (time.perf_counter() - t0) * 1000
    state.request_count    += 1
    state.total_latency_ms += latency_ms
    logger.info(f"request_id={request.request_id} batch={len(request.texts)} latency={latency_ms:.1f}ms")
    return ModerationResponse(request_id=request.request_id, results=results, latency_ms=round(latency_ms, 2))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    mean = state.total_latency_ms / state.request_count if state.request_count > 0 else 0.0
    return HealthResponse(status="ok", model_loaded=hasattr(state, "session"), requests_served=state.request_count, mean_latency_ms=round(mean, 2))


@app.get("/")
async def root() -> dict:
    return {"service": "content-moderation", "version": "1.0.0", "docs": "/docs", "health": "/health"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
