"""
API tests — mocked inference, no real model needed.
Run: pytest test_api.py -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

MOCK_LOGITS = np.array([[2.5, 0.3]])  # "safe" wins


@pytest.fixture
def client():
    with patch("app.ort.InferenceSession") as mock_sess_cls, \
         patch("app.AutoTokenizer") as mock_tok_cls:

        mock_sess = MagicMock()
        mock_sess.run.return_value = [MOCK_LOGITS]
        mock_sess.get_providers.return_value = ["CPUExecutionProvider"]
        mock_sess_cls.return_value = mock_sess

        mock_tok = MagicMock()
        mock_tok.return_value = {
            "input_ids":      np.ones((1, 128), dtype=np.int64),
            "attention_mask": np.ones((1, 128), dtype=np.int64),
        }
        mock_tok_cls.from_pretrained.return_value = mock_tok

        from fastapi.testclient import TestClient
        from app import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    assert client.get("/health").status_code == 200

def test_moderate_single(client):
    resp = client.post("/moderate", json={"texts": ["hello world"]})
    assert resp.status_code == 200
    r = resp.json()["results"][0]
    assert r["label"] in ["safe", "toxic"]
    assert 0 <= r["confidence"] <= 1
    assert set(r["probabilities"]) == {"safe", "toxic"}

def test_moderate_batch(client):
    resp = client.post("/moderate", json={"texts": ["a", "b", "c"]})
    assert len(resp.json()["results"]) == 3

def test_has_latency(client):
    resp = client.post("/moderate", json={"texts": ["test"]})
    assert resp.json()["latency_ms"] >= 0

def test_rejects_empty_list(client):
    assert client.post("/moderate", json={"texts": []}).status_code == 422

def test_rejects_empty_string(client):
    assert client.post("/moderate", json={"texts": [""]}).status_code == 422

def test_rejects_too_long(client):
    assert client.post("/moderate", json={"texts": ["x" * 1001]}).status_code == 422

def test_rejects_oversized_batch(client):
    assert client.post("/moderate", json={"texts": ["t"] * 33}).status_code == 422

def test_custom_request_id(client):
    resp = client.post("/moderate", json={"texts": ["hi"], "request_id": "abc-123"})
    assert resp.json()["request_id"] == "abc-123"
