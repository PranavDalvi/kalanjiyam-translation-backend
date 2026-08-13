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

