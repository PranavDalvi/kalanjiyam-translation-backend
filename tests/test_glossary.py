import csv
import os
import tempfile
import time
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.glossary import (
    GlossaryService,
    pre_translate_replace,
    post_translate_replace,
    LANGUAGE_NAME_TO_ISO
)

client = TestClient(main.app)

def test_language_mappings() -> None:
    assert LANGUAGE_NAME_TO_ISO["English"] == "en"
    assert LANGUAGE_NAME_TO_ISO["Marathi"] == "mr"
    assert LANGUAGE_NAME_TO_ISO["Hindi"] == "hi"

def test_pre_translate_replace() -> None:
    glossary = {
        "written warning": "लेखी इशारा",
        "wrongdoer": "अन्याय करणारा",
        "yield": "उत्पन्न",
    }
    
    text = "He received a Written Warning because the wrongdoer did not yield."
    processed, mapping = pre_translate_replace(text, glossary)
    
    assert "<dnt>99000</dnt>" in processed
    assert "<dnt>99001</dnt>" in processed
    assert "<dnt>99002</dnt>" in processed
    
    # Check mapping content
    assert mapping["99000"] == "लेखी इशारा"
    assert mapping["99001"] == "अन्याय करणारा"
    assert mapping["99002"] == "उत्पन्न"

def test_pre_translate_replace_longest_match_first() -> None:
    # "wrongful dismissal" is longer than "wrongful"
    # we want "wrongful dismissal" to be matched first, not leaving "wrongful" replaced and "dismissal" left over.
    glossary = {
        "wrongful": "चुकीचे",
        "wrongful dismissal": "चुकीच्या पद्धतीने बाद करणे",
    }
    
    text = "It was a wrongful dismissal."
    processed, mapping = pre_translate_replace(text, glossary)
    
    assert "<dnt>99000</dnt>" in processed
    assert "dismissal" not in processed # it should be completely replaced
    assert mapping["99000"] == "चुकीच्या पद्धतीने बाद करणे"

def test_post_translate_replace() -> None:
    mapping = {
        "99000": "लेखी इशारा",
        "99001": "अन्याय करणारा",
    }
    
    # Test strict tag matching
    text1 = "त्याला <dnt>99000</dnt> देण्यात आला."
    res1 = post_translate_replace(text1, mapping)
    assert res1 == "त्याला लेखी इशारा देण्यात आला."
    
    # Test spaces inside tag matching
    text2 = "त्याला <dnt> 99000 </dnt> देण्यात आला."
    res2 = post_translate_replace(text2, mapping)
    assert res2 == "त्याला लेखी इशारा देण्यात आला."
    
    # Test uppercase/lowercase and spaces inside tag
    text3 = "त्याला < dnt >99000< / dnt > देण्यात आला."
    res3 = post_translate_replace(text3, mapping)
    assert res3 == "त्याला लेखी इशारा देण्यात आला."
    
    # Test fallback if tags are completely stripped but number remains
    text4 = "त्याला 99000 देण्यात आला."
    res4 = post_translate_replace(text4, mapping)
    assert res4 == "त्याला लेखी इशारा देण्यात आला."

def test_glossary_service_caching_and_invalidation() -> None:
    # Use temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        service = GlossaryService()
        # Override glossaries directory
        service.get_glossaries_dir = lambda: temp_dir
        
        glossary_file = os.path.join(temp_dir, "administrative_en_mr.csv")
        
        # Write first version of glossary
        with open(glossary_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["written warning", "लेखी इशारा"])
            
        # First load
        glossary_dict1 = service.get_glossary_dict("administrative", "English", "Marathi")
        assert glossary_dict1 is not None
        assert glossary_dict1["written warning"] == "लेखी इशारा"
        
        # Verify it cached it
        assert glossary_file in service._cache
        
        # Update glossary file (force different mtime)
        time.sleep(0.01) # ensure file mtime will be different
        with open(glossary_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["written warning", "लेखी चेतावणी"])
            
        # Load again
        glossary_dict2 = service.get_glossary_dict("administrative", "English", "Marathi")
        assert glossary_dict2 is not None
        assert glossary_dict2["written warning"] == "लेखी चेतावणी"

def test_list_glossaries_endpoint(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some dummy glossary files
        open(os.path.join(temp_dir, "administrative_en_mr.csv"), "w").close()
        open(os.path.join(temp_dir, "agri_en_hi.csv"), "w").close()
        
        # Patch the directory
        monkeypatch.setattr(main.glossary_service, "get_glossaries_dir", lambda: temp_dir)
        
        response = client.get("/glossaries")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        
        names = [item["name"] for item in data]
        assert "administrative" in names
        assert "agri" in names

def test_translate_text_endpoint_with_glossary(monkeypatch) -> None:
    # Mock get_translation_model
    def fake_get_translation_model(model_name: str, src_lang: str, tgt_lang: str, gpu_id: int):
        return object(), object(), object()
        
    # Mock translate_batch_memory_safe to return translated text with placeholders preserved
    def fake_translate_batch_memory_safe(
        sentences, model, tokenizer, ip, src_lang, tgt_lang, batch_size, glossary_dict
    ):
        assert glossary_dict is not None
        assert glossary_dict["wrongdoer"] == "अन्याय करणारा"
        # Return text with placeholder preserved
        return ["त्याला <dnt> 99000 </dnt> शिक्षा झाली."]
        
    # Mock GlossaryService to return dummy terms
    def fake_get_glossary_dict(glossary, src_lang, tgt_lang):
        assert glossary == "administrative"
        return {"wrongdoer": "अन्याय करणारा"}
        
    monkeypatch.setattr(main.service, "get_translation_model", fake_get_translation_model)
    monkeypatch.setattr(main.service, "translate_batch_memory_safe", fake_translate_batch_memory_safe)
    monkeypatch.setattr(main.glossary_service, "get_glossary_dict", fake_get_glossary_dict)
    
    response = client.post(
        "/translate/text",
        json={
            "text": "The wrongdoer was punished.",
            "model_name": "ai4bharat/indictrans2-en-indic-1B",
            "source_language": "English",
            "target_language": "Marathi",
            "gpu_id": 0,
            "batch_size": 8,
            "glossary": "administrative"
        }
    )
    
    assert response.status_code == 200
    assert response.json()["text"] == "त्याला अन्याय करणारा शिक्षा झाली."


def test_get_merged_glossary_dict_multiple() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        service = GlossaryService()
        service.get_glossaries_dir = lambda: temp_dir
        
        # Create 2 glossary files
        with open(os.path.join(temp_dir, "administrative_en_mr.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["wrongdoer", "अन्याय करणारा"])
            writer.writerow(["warning", "इशारा"])
            
        with open(os.path.join(temp_dir, "agri_en_mr.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["yield", "उत्पन्न"])
            writer.writerow(["warning", "चेतावणी"])  # Conflict: warning is in both
            
        # Test merging list
        merged = service.get_merged_glossary_dict(["administrative", "agri"], "English", "Marathi")
        assert merged is not None
        assert merged["wrongdoer"] == "अन्याय करणारा"
        assert merged["yield"] == "उत्पन्न"
        # Conflict resolution: later ('agri') overrides earlier ('administrative')
        assert merged["warning"] == "चेतावणी"


def test_get_merged_glossary_dict_all() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        service = GlossaryService()
        service.get_glossaries_dir = lambda: temp_dir
        
        # Create 2 glossary files for en->mr, and 1 for en->hi (should not be loaded)
        with open(os.path.join(temp_dir, "administrative_en_mr.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["wrongdoer", "अन्याय करणारा"])
            
        with open(os.path.join(temp_dir, "agri_en_mr.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["yield", "उत्पन्न"])
            
        with open(os.path.join(temp_dir, "history_en_hi.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["king", "राजा"])
            
        # Test 'all' keyword
        merged = service.get_merged_glossary_dict("all", "English", "Marathi")
        assert merged is not None
        assert "wrongdoer" in merged
        assert "yield" in merged
        assert "king" not in merged  # different target language


def test_get_merged_glossary_dict_comma_separated() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        service = GlossaryService()
        service.get_glossaries_dir = lambda: temp_dir
        
        # Create 2 glossary files
        with open(os.path.join(temp_dir, "administrative_en_mr.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["wrongdoer", "अन्याय करणारा"])
            
        with open(os.path.join(temp_dir, "agri_en_mr.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["yield", "उत्पन्न"])
            
        # Test comma-separated string parameter
        merged = service.get_merged_glossary_dict("administrative, agri", "English", "Marathi")
        assert merged is not None
        assert "wrongdoer" in merged
        assert "yield" in merged

