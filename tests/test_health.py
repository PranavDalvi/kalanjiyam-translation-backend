from fastapi.testclient import TestClient

import app.main as main


client = TestClient(main.app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "available_gpus" in body


def test_languages_endpoint() -> None:
    response = client.get("/languages")
    assert response.status_code == 200
    body = response.json()
    assert "English" in body
    assert "Hindi" in body


def test_models_endpoint() -> None:
    response = client.get("/models")
    assert response.status_code == 200
    body = response.json()
    assert any(model["model_name"] == "ai4bharat/indictrans2-en-indic-1B" for model in body)
    assert any(model["model_name"] == "ai4bharat/indictrans2-indic-en-1B" for model in body)
    assert any(model["model_name"] == "ai4bharat/indictrans2-indic-indic-1B" for model in body)
    assert any(model["model_name"] == "google/gemma-4-12b-it" for model in body)
    gemma_entry = next(m for m in body if m["model_name"] == "google/gemma-4-12b-it")
    assert gemma_entry["key"] == "gemma-4-12b-it"
    assert "English" in gemma_entry["source_languages"]
    assert "Hindi" in gemma_entry["target_languages"]


def test_translate_text_endpoint_with_mocked_service(monkeypatch) -> None:
    def fake_get_translation_model(model_name: str, src_lang_name: str, tgt_lang_name: str, gpu_id: int):
        assert model_name == "ai4bharat/indictrans2-en-indic-1B"
        assert src_lang_name == "English"
        assert tgt_lang_name == "Hindi"
        assert gpu_id == 0
        return object(), object(), object()

    def fake_translate_batch_memory_safe(
        sentences,
        model,
        tokenizer,
        ip,
        src_lang,
        tgt_lang,
        batch_size,
    ):
        assert src_lang == "eng_Latn"
        assert tgt_lang == "hin_Deva"
        assert batch_size == 8
        return [f"MOCK_TRANSLATED: {sentences[0]}"]

    monkeypatch.setattr(main.service, "get_translation_model", fake_get_translation_model)
    monkeypatch.setattr(main.service, "translate_batch_memory_safe", fake_translate_batch_memory_safe)

    response = client.post(
        "/translate/text",
        json={
            "text": "Hello world",
            "model_name": "ai4bharat/indictrans2-en-indic-1B",
            "source_language": "English",
            "target_language": "Hindi",
            "gpu_id": 0,
            "batch_size": 8,
        },
    )

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["engine"] == "indictrans2"
    assert res_data["model"]["name"] == "ai4bharat/indictrans2-en-indic-1B"
    assert res_data["source_language"] == "en"
    assert res_data["target_language"] == "hi"
    assert res_data["text"] == "MOCK_TRANSLATED: Hello world"
    assert res_data["translated_text"] == "MOCK_TRANSLATED: Hello world"
    assert res_data["confidence"] == 0.965
    assert res_data["input_chars"] == 11
    assert res_data["output_chars"] == len("MOCK_TRANSLATED: Hello world")
    assert isinstance(res_data["engine_latency_ms"], float)


def test_translate_response_spec_v1_sanskrit_to_english(monkeypatch) -> None:
    def fake_get_translation_model(model_name: str, src_lang_name: str, tgt_lang_name: str, gpu_id: int):
        assert model_name == "ai4bharat/indictrans2-indic-en-1B"
        assert src_lang_name == "Sanskrit"
        assert tgt_lang_name == "English"
        return object(), object(), object()

    def fake_translate_batch_memory_safe(
        sentences, model, tokenizer, ip, src_lang, tgt_lang, batch_size, **kwargs
    ):
        assert src_lang == "san_Deva"
        assert tgt_lang == "eng_Latn"
        return ["Gathered together on the sacred field of Kurukshetra, eager for battle..."]

    monkeypatch.setattr(main.service, "get_translation_model", fake_get_translation_model)
    monkeypatch.setattr(main.service, "translate_batch_memory_safe", fake_translate_batch_memory_safe)

    request_payload = {
        "text": "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।",
        "source_language": "Sanskrit",
        "target_language": "English",
        "model_name": "ai4bharat/indictrans2-indic-en-1B",
        "batch_size": 8
    }

    # Test /translate/text
    res = client.post("/translate/text", json=request_payload)
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "success"
    assert data["engine"] == "indictrans2"
    assert data["model"]["name"] == "ai4bharat/indictrans2-indic-en-1B"
    assert data["model"]["version"] == "1.0"
    assert data["source_language"] == "sa"
    assert data["target_language"] == "en"
    assert data["text"] == "Gathered together on the sacred field of Kurukshetra, eager for battle..."
    assert data["translated_text"] == "Gathered together on the sacred field of Kurukshetra, eager for battle..."
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["input_chars"] == len(request_payload["text"])
    assert data["output_chars"] == len(data["translated_text"])
    assert data["engine_latency_ms"] >= 0.0

    # Test /v1/translate alias
    res_v1 = client.post("/v1/translate", json=request_payload)
    assert res_v1.status_code == 200
    data_v1 = res_v1.json()
    assert data_v1["status"] == "success"
    assert data_v1["source_language"] == "sa"
    assert data_v1["target_language"] == "en"
    assert data_v1["text"] == data["text"]


def test_translate_error_payload_format() -> None:
    # Test invalid target language
    response = client.post(
        "/translate/text",
        json={
            "text": "धर्मक्षेत्रे",
            "source_language": "Sanskrit",
            "target_language": "French",
            "model_name": "ai4bharat/indictrans2-indic-en-1B"
        }
    )
    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "error"
    assert "Unsupported language pair" in body["detail"]
    assert "Sanskrit (sa)" in body["detail"]
    assert "French (fr)" in body["detail"]


def test_translate_text_endpoint_gemma_4_12b(monkeypatch) -> None:
    def fake_get_translation_model(model_name: str, src_lang_name: str, tgt_lang_name: str, gpu_id: int):
        assert model_name == "google/gemma-4-12b-it"
        assert src_lang_name == "English"
        assert tgt_lang_name == "Tamil"
        return object(), object(), None

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "application/json"}
        def json(self):
            return {"text": "வணக்கம் உலகம்"}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, json=None, **kwargs):
            assert "/translate" in url
            assert json["source_language"] == "English"
            assert json["target_language"] == "Tamil"
            return FakeResponse()

    monkeypatch.setattr("httpx.Client", FakeClient)

    response = client.post(
        "/translate/text",
        json={
            "text": "Hello world",
            "model_name": "google/gemma-4-12b-it",
            "source_language": "English",
            "target_language": "Tamil",
            "gpu_id": 0,
            "batch_size": 8,
        },
    )

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["engine"] == "gemma"
    assert res_data["model"]["name"] == "google/gemma-4-12b-it"
    assert res_data["source_language"] == "en"
    assert res_data["target_language"] == "ta"
    assert res_data["translated_text"] == "வணக்கம் உலகம்"


def test_gemma_model_alias_resolution(monkeypatch) -> None:
    called_urls = []

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "application/json"}
        def json(self):
            return {"text": "மொழிபெயர்ப்பு"}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, json=None, **kwargs):
            called_urls.append(url)
            return FakeResponse()

    monkeypatch.setattr("httpx.Client", FakeClient)

    for alias in ["gemma-4-12b", "gemma 4 12b", "google/gemma-4-12B-it", "gemma4-12b-it"]:
        response = client.post(
            "/translate/text",
            json={
                "text": "Testing alias",
                "model_name": alias,
                "source_language": "English",
                "target_language": "Tamil",
            },
        )
        assert response.status_code == 200
        assert response.json()["engine"] == "gemma"

    assert len(called_urls) == 4


def test_translate_batch_memory_safe_gemma_logic() -> None:
    import torch

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
            # Return dummy tensor with shape [len(texts), 10]
            return DummyBatch(torch.zeros((len(texts), 10), dtype=torch.long))

        def batch_decode(self, tokens, skip_special_tokens=True):
            return ["வணக்கம்"] * len(tokens)

    class DummyModel:
        def __init__(self):
            self.device = torch.device("cpu")
            self.config = type("Config", (), {"is_encoder_decoder": False})()

        def generate(self, **kwargs):
            input_ids = kwargs.get("input_ids", torch.zeros((1, 10), dtype=torch.long))
            # Append 5 new tokens
            return torch.cat([input_ids, torch.ones((input_ids.shape[0], 5), dtype=torch.long)], dim=1)

    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()

    results = main.service.translate_batch_memory_safe(
        sentences=["Hello world"],
        model=dummy_model,
        tokenizer=dummy_tokenizer,
        ip=None,
        src_lang="English",
        tgt_lang="Tamil",
        batch_size=8,
    )

    assert results == ["வணக்கம்"]



