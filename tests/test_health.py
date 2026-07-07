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
    assert response.json()["text"] == "MOCK_TRANSLATED: Hello world"
