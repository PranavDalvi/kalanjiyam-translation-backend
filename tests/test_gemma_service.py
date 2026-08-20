from fastapi.testclient import TestClient
import pytest

from gemma_service.app import app, TranslateRequest


def test_gemma_service_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["engine"] == "gemma"
    assert "google/gemma-4-12b-it" in data["model"]


def test_gemma_service_translate_mocked(monkeypatch):
    import gemma_service.app as gemma_app

    class DummyBatch(dict):
        def __init__(self, input_ids):
            super().__init__({"input_ids": input_ids})
            self.input_ids = input_ids
        def to(self, device):
            return self

    class DummyTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.chat_template = None

        def __call__(self, texts, truncation=True, padding=True, return_tensors="pt"):
            import torch
            return DummyBatch(torch.zeros((len(texts), 10), dtype=torch.long))

        def batch_decode(self, tokens, skip_special_tokens=True):
            return ["வணக்கம் உலகம்"] * len(tokens)

        def decode(self, token_ids, skip_special_tokens=True):
            return "வணக்கம் உலகம்"

    class DummyModel:
        def __init__(self):
            import torch
            self.device = torch.device("cpu")

        def generate(self, **kwargs):
            import torch
            input_ids = kwargs.get("input_ids", torch.zeros((1, 10), dtype=torch.long))
            return torch.cat([input_ids, torch.ones((input_ids.shape[0], 5), dtype=torch.long)], dim=1)

    monkeypatch.setattr(gemma_app, "_model", DummyModel())
    monkeypatch.setattr(gemma_app, "_tokenizer", DummyTokenizer())
    monkeypatch.setattr(gemma_app, "_device", "cpu")

    client = TestClient(app)
    response = client.post(
        "/translate",
        json={
            "text": "Hello world",
            "source_language": "English",
            "target_language": "Tamil",
            "gpu_id": 0,
            "batch_size": 4,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert data["text"] == "வணக்கம் உலகம்"
