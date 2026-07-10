import io
import os
import sys
import tempfile
import threading
import types
from typing import Dict, List, Optional, Tuple, Literal, Set

# Force progress bars (like tqdm) to render in non-TTY environments (e.g., Docker logs)
sys.stderr.isatty = lambda: True
sys.stdout.isatty = lambda: True


import pdfplumber
import torch
from docx import Document
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from app.glossary import GlossaryService, pre_translate_replace, post_translate_replace

glossary_service = GlossaryService()

# Keep behavior consistent with your original script.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(os.getcwd(), "gradio_temp"))
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

LANGUAGES: Dict[str, str] = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Bengali": "ben_Beng",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Punjabi": "pan_Guru",
    "Urdu": "urd_Arab",
    "Odia": "ory_Orya",
    "Assamese": "asm_Beng",
    "Sanskrit": "san_Deva",
    "Kashmiri": "kas_Arab",
    "Sindhi": "snd_Arab",
    "Manipuri": "mni_Mtei",
    "Santali": "sat_Olch",
    "Nepali": "npi_Deva",
    "Konkani": "gom_Deva",
    "Dogri": "doi_Deva",
    "Bodo": "brx_Deva",
    "Maithili": "mai_Deva",
}

MODEL_EN_INDIC = "ai4bharat/indictrans2-en-indic-1B"
MODEL_INDIC_EN = "ai4bharat/indictrans2-indic-en-1B"
MODEL_INDIC_INDIC = "ai4bharat/indictrans2-indic-indic-1B"

MODEL_CATALOG: Dict[str, Dict[str, object]] = {
    MODEL_EN_INDIC: {
        "key": "en-indic",
        "description": "English to Indic translation model",
        "source_languages": ["English"],
        "target_languages": [lang for lang in LANGUAGES.keys() if lang != "English"],
    },
    MODEL_INDIC_EN: {
        "key": "indic-en",
        "description": "Indic to English translation model",
        "source_languages": [lang for lang in LANGUAGES.keys() if lang != "English"],
        "target_languages": ["English"],
    },
    MODEL_INDIC_INDIC: {
        "key": "indic-indic",
        "description": "Indic to Indic translation model",
        "source_languages": [lang for lang in LANGUAGES.keys() if lang != "English"],
        "target_languages": [lang for lang in LANGUAGES.keys() if lang != "English"],
    },
}

ModelName = Literal[
    "ai4bharat/indictrans2-en-indic-1B",
    "ai4bharat/indictrans2-indic-en-1B",
    "ai4bharat/indictrans2-indic-indic-1B",
]


# Compatibility patches are no longer needed as the environment uses a pinned transformers==4.39.3 release.



class TranslateTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model_name: ModelName
    source_language: str
    target_language: str
    gpu_id: int = 0
    batch_size: int = Field(default=8, ge=1, le=64)
    glossary: Optional[str] = None


class TranslationService:
    def __init__(self) -> None:
        try:
            self.available_gpus = list(range(torch.cuda.device_count()))
        except Exception as e:
            print(f"Failed to query CUDA devices: {e}. Disabling GPU support.")
            self.available_gpus = []
        self.loaded_models: Dict[Tuple[str, str], Dict[str, object]] = {}
        self.active_downloads: Set[str] = set()
        self._lock = threading.Lock()
        self._inference_locks: Dict[int, threading.Lock] = {}
        self._inference_locks_lock = threading.Lock()

    def _get_inference_lock(self, model: object) -> threading.Lock:
        model_id = id(model)
        with self._inference_locks_lock:
            if model_id not in self._inference_locks:
                self._inference_locks[model_id] = threading.Lock()
            return self._inference_locks[model_id]

    def _resolve_model(self, model_name: str, src_lang_name: str, tgt_lang_name: str) -> Tuple[str, str]:
        if model_name not in MODEL_CATALOG:
            raise HTTPException(status_code=400, detail="Invalid model_name. Use one of the models returned by /models.")

        model_meta = MODEL_CATALOG[model_name]
        if src_lang_name not in model_meta["source_languages"]:
            raise HTTPException(status_code=400, detail=f"Model {model_name} does not support source language {src_lang_name}.")
        if tgt_lang_name not in model_meta["target_languages"]:
            raise HTTPException(status_code=400, detail=f"Model {model_name} does not support target language {tgt_lang_name}.")

        return model_meta["key"], model_name

    def get_translation_model(
        self, model_name: str, src_lang_name: str, tgt_lang_name: str, gpu_id: int
    ) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer, object]:
        model_key, model_name = self._resolve_model(model_name, src_lang_name, tgt_lang_name)

        use_cuda = False
        device = "cpu"

        if self.available_gpus and gpu_id in self.available_gpus:
            try:
                # Test CUDA initialization
                test_tensor = torch.zeros(1).to(f"cuda:{gpu_id}")
                del test_tensor
                device = f"cuda:{gpu_id}"
                use_cuda = True
            except Exception as e:
                print(f"CUDA initialization failed for GPU {gpu_id} ({e}). Falling back to CPU.")
        else:
            print(f"GPU {gpu_id} not available or no CUDA GPUs detected. Falling back to CPU.")

        cache_key = (device, model_key)

        if cache_key in self.loaded_models:
            bundle = self.loaded_models[cache_key]
            return bundle["model"], bundle["tokenizer"], bundle["ip"]

        with self._lock:
            if cache_key in self.loaded_models:
                bundle = self.loaded_models[cache_key]
                return bundle["model"], bundle["tokenizer"], bundle["ip"]

            # Import lazily so non-translation endpoints can run even if model deps are not ready.
            from IndicTransToolkit.processor import IndicProcessor

            self.active_downloads.add(model_name)
            try:
                offline_mode = os.environ.get("TRANSFORMERS_OFFLINE", "1") == "1"

                ip = IndicProcessor(inference=True)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only=offline_mode,
                )

                dtype = torch.float16 if use_cuda else torch.float32
                try:
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        torch_dtype=dtype,
                        local_files_only=offline_mode,
                    ).to(device)
                    model.eval()
                except Exception as e:
                    if use_cuda:
                        print(f"Failed to load model on GPU: {e}. Retrying CPU fallback.")
                        device = "cpu"
                        use_cuda = False
                        cache_key = (device, model_key)
                        if cache_key in self.loaded_models:
                            bundle = self.loaded_models[cache_key]
                            return bundle["model"], bundle["tokenizer"], bundle["ip"]

                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            torch_dtype=torch.float32,
                            local_files_only=offline_mode,
                        ).to(device)
                        model.eval()
                    else:
                        raise e
            finally:
                self.active_downloads.discard(model_name)

            self.loaded_models[cache_key] = {"model": model, "tokenizer": tokenizer, "ip": ip}
            return model, tokenizer, ip

    def _is_model_cached(self, model_name: str) -> bool:
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        cache_dir = os.path.join(hf_home, "hub", f"models--{model_name.replace('/', '--')}")
        if not os.path.exists(cache_dir):
            return False
        import glob
        pattern1 = os.path.join(cache_dir, "**", "model.safetensors")
        pattern2 = os.path.join(cache_dir, "**", "pytorch_model.bin")
        files = glob.glob(pattern1, recursive=True) + glob.glob(pattern2, recursive=True)
        complete_files = [f for f in files if not f.endswith(".incomplete")]
        return len(complete_files) > 0

    def get_model_status(self, model_name: str, key: str) -> str:
        # Check if loaded in memory
        is_loaded = any(k[1] == key for k in self.loaded_models.keys())
        if is_loaded:
            return "loaded"

        # Check if active downloading/loading
        if model_name in self.active_downloads:
            return "downloading"

        # Check if cached on disk
        if self._is_model_cached(model_name):
            return "cached"

        return "not_downloaded"

    def available_models(self) -> List[Dict[str, object]]:
        return [
            {
                "model_name": model_name,
                "key": meta["key"],
                "description": meta["description"],
                "source_languages": meta["source_languages"],
                "target_languages": meta["target_languages"],
                "status": self.get_model_status(model_name, meta["key"]),
            }
            for model_name, meta in MODEL_CATALOG.items()
        ]

    def translate_batch_memory_safe(
        self,
        sentences: List[str],
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        ip: object,
        src_lang: str,
        tgt_lang: str,
        batch_size: int = 8,
        glossary_dict: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        if not sentences:
            return []

        preprocessed_sentences = []
        mappings = []
        if glossary_dict:
            for s in sentences:
                proc_s, mapping = pre_translate_replace(s, glossary_dict)
                preprocessed_sentences.append(proc_s)
                mappings.append(mapping)
        else:
            preprocessed_sentences = sentences
            mappings = [None] * len(sentences)

        all_translations: List[str] = []
        total_sentences = len(preprocessed_sentences)

        for i in range(0, total_sentences, batch_size):
            batch = preprocessed_sentences[i : i + batch_size]
            batch_mappings = mappings[i : i + batch_size]
            valid_indices = [idx for idx, s in enumerate(batch) if s.strip()]
            valid_sentences = [batch[idx] for idx in valid_indices]

            if valid_sentences:
                preprocessed = ip.preprocess_batch(valid_sentences, src_lang=src_lang, tgt_lang=tgt_lang)
                inputs = tokenizer(
                    preprocessed,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                )

                lock = self._get_inference_lock(model)
                with lock:
                    inputs = inputs.to(model.device)
                    with torch.no_grad():
                        generated_tokens = model.generate(
                            **inputs,
                            use_cache=True,
                            min_length=0,
                            max_length=512,
                            num_beams=4,
                            early_stopping=True,
                        )

                    translations = tokenizer.batch_decode(
                        generated_tokens.detach().cpu().tolist(),
                        skip_special_tokens=True,
                    )
                translations = ip.postprocess_batch(translations, lang=tgt_lang)

                for idx, trans in zip(valid_indices, translations):
                    mapping = batch_mappings[idx]
                    if mapping:
                        trans = post_translate_replace(trans, mapping)
                    batch[idx] = trans

            all_translations.extend(batch)

        if model.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return all_translations

    def process_docx(
        self,
        file_path: str,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        ip: object,
        src_lang: str,
        tgt_lang: str,
        batch_size: int,
        glossary_dict: Optional[Dict[str, str]] = None,
    ) -> Document:
        doc = Document(file_path)

        paras_text = [p.text for p in doc.paragraphs]
        translated_paras = self.translate_batch_memory_safe(
            paras_text,
            model,
            tokenizer,
            ip,
            src_lang,
            tgt_lang,
            batch_size=batch_size,
            glossary_dict=glossary_dict,
        )

        for i, paragraph in enumerate(doc.paragraphs):
            if paragraph.text.strip():
                for run in paragraph.runs:
                    run.text = ""
                if paragraph.runs:
                    paragraph.runs[0].text = translated_paras[i]
                else:
                    paragraph.add_run(translated_paras[i])

        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        translated_cell = self.translate_batch_memory_safe(
                            [cell.text],
                            model,
                            tokenizer,
                            ip,
                            src_lang,
                            tgt_lang,
                            batch_size=1,
                            glossary_dict=glossary_dict,
                        )[0]
                        cell.text = ""
                        for paragraph in cell.paragraphs:
                            if paragraph.text == "":
                                paragraph.add_run(translated_cell)

        return doc


service = TranslationService()
app = FastAPI(title="Kalanjiyam Translation API", version="1.0.0")


@app.on_event("startup")
def preload_models():
    offline_mode = os.environ.get("TRANSFORMERS_OFFLINE", "1") == "1"
    if not offline_mode:
        print("Preloading default translation model (ai4bharat/indictrans2-en-indic-1B) on startup...")
        try:
            service.get_translation_model(
                MODEL_EN_INDIC,
                "English",
                "Hindi",
                0
            )
            print("Default model preloaded successfully.")
        except Exception as e:
            print(f"Error preloading model during startup: {e}")


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "status": "ok",
        "available_gpus": service.available_gpus,
        "offline_mode": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "0"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "0"),
        },
    }


@app.get("/languages")
def languages() -> Dict[str, str]:
    return LANGUAGES


@app.get("/models")
def models() -> List[Dict[str, object]]:
    return service.available_models()


@app.get("/glossaries")
def list_glossaries() -> List[Dict[str, str]]:
    glossaries_dir = glossary_service.get_glossaries_dir()
    if not os.path.exists(glossaries_dir):
        return []

    available = []
    try:
        for filename in os.listdir(glossaries_dir):
            if filename.endswith(".csv"):
                # Pattern is: name_src_tgt.csv
                parts = filename[:-4].split("_")
                if len(parts) >= 3:
                    tgt = parts[-1]
                    src = parts[-2]
                    name = "_".join(parts[:-2])
                    available.append({
                        "name": name,
                        "source_language_code": src,
                        "target_language_code": tgt,
                        "filename": filename
                    })
    except Exception as e:
        print(f"Error listing glossaries: {e}")

    return available


@app.post("/translate/text")
def translate_text(payload: TranslateTextRequest) -> Dict[str, str]:
    src_lang = LANGUAGES.get(payload.source_language)
    tgt_lang = LANGUAGES.get(payload.target_language)

    if not src_lang or not tgt_lang:
        raise HTTPException(status_code=400, detail="Invalid source or target language.")

    try:
        model, tokenizer, ip = service.get_translation_model(
            payload.model_name,
            payload.source_language,
            payload.target_language,
            payload.gpu_id,
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        err_msg = str(e)
        if "offline" in err_msg.lower() or "local_files" in err_msg.lower() or "does not appear to have a file named" in err_msg.lower():
            raise HTTPException(
                status_code=503,
                detail=f"Translation model failed to load. The local cache is likely incomplete or corrupted. Try running './setup_and_run.sh' to download/repair the cache. Error: {err_msg}"
            )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load translation model: {err_msg}"
        )

    # Load glossary mapping if requested
    glossary_dict = None
    if payload.glossary:
        glossary_dict = glossary_service.get_glossary_dict(
            payload.glossary,
            payload.source_language,
            payload.target_language
        )

    # Split text by newlines to prevent silent truncation on long texts
    lines = payload.text.split("\n")
    translated_lines = service.translate_batch_memory_safe(
        lines,
        model,
        tokenizer,
        ip,
        src_lang,
        tgt_lang,
        batch_size=payload.batch_size,
        glossary_dict=glossary_dict,
    )

    return {"text": "\n".join(translated_lines)}


@app.post("/translate/document")
def translate_document(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    source_language: str = Form(...),
    target_language: str = Form(...),
    gpu_id: int = Form(0),
    batch_size: int = Form(8),
    glossary: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
) -> FileResponse:
    src_lang = LANGUAGES.get(source_language)
    tgt_lang = LANGUAGES.get(target_language)

    if not src_lang or not tgt_lang:
        raise HTTPException(status_code=400, detail="Invalid source or target language.")

    try:
        model, tokenizer, ip = service.get_translation_model(model_name, source_language, target_language, gpu_id)
    except HTTPException as he:
        raise he
    except Exception as e:
        err_msg = str(e)
        if "offline" in err_msg.lower() or "local_files" in err_msg.lower() or "does not appear to have a file named" in err_msg.lower():
            raise HTTPException(
                status_code=503,
                detail=f"Translation model failed to load. The local cache is likely incomplete or corrupted. Try running './setup_and_run.sh' to download/repair the cache. Error: {err_msg}"
            )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load translation model: {err_msg}"
        )

    # Load glossary mapping if requested
    glossary_dict = None
    if glossary:
        glossary_dict = glossary_service.get_glossary_dict(
            glossary,
            source_language,
            target_language
        )

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".docx", ".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .docx, .pdf, or .txt")

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, f"input{ext}")
        output_path = os.path.join(temp_dir, "translated_output.docx")

        content = file.file.read()
        with open(input_path, "wb") as handle:
            handle.write(content)

        if ext == ".docx":
            doc = service.process_docx(input_path, model, tokenizer, ip, src_lang, tgt_lang, batch_size, glossary_dict)
            doc.save(output_path)

        elif ext == ".pdf":
            text_list: List[str] = []
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_list.append(page_text)
            full_text = "\n".join(text_list)
            translated = service.translate_batch_memory_safe(
                full_text.split("\n"),
                model,
                tokenizer,
                ip,
                src_lang,
                tgt_lang,
                batch_size=batch_size,
                glossary_dict=glossary_dict,
            )
            doc = Document()
            for line in translated:
                doc.add_paragraph(line)
            doc.save(output_path)

        else:  # .txt
            with open(input_path, "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle.readlines() if line.strip()]
            translated = service.translate_batch_memory_safe(
                lines,
                model,
                tokenizer,
                ip,
                src_lang,
                tgt_lang,
                batch_size=batch_size,
                glossary_dict=glossary_dict,
            )
            doc = Document()
            for line in translated:
                doc.add_paragraph(line)
            doc.save(output_path)

        final_name = os.path.splitext(file.filename or "document")[0] + f"_translated_{target_language}.docx"
        persisted_path = os.path.join(os.getcwd(), final_name)
        with open(output_path, "rb") as src_file, open(persisted_path, "wb") as dst_file:
            dst_file.write(src_file.read())

    if background_tasks:
        background_tasks.add_task(os.remove, persisted_path)

    return FileResponse(
        path=persisted_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=final_name,
    )
