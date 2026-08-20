"""
Gemma 4 12B Translation Sidecar Service.

A lightweight FastAPI service dedicated to running google/gemma-4-12b-it
translations with modern transformers (>=5.0).

This service runs in its own container with its own transformers version,
completely isolated from the IndicTrans2 main service.
"""

import os
import sys
import logging
import threading
import time
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("gemma_service")

# Environment
os.environ.setdefault("HF_HUB_OFFLINE", os.environ.get("TRANSFORMERS_OFFLINE", "1"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", os.environ.get("TRANSFORMERS_OFFLINE", "1"))

MODEL_NAME = "google/gemma-4-12b-it"

app = FastAPI(title="Gemma 4 12B Translation Service", version="1.0.0")

# ---------- Model Management ----------

_model = None
_tokenizer = None
_device = None
_load_lock = threading.Lock()


def _get_hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN")
    if token:
        token = token.strip().strip("'\"\\")
        if token.startswith("token="):
            token = token[6:]
    return token or None


def _select_gpu(requested_gpu: int) -> int:
    """Auto-select the GPU with the most free VRAM."""
    auto_select = os.environ.get("AUTO_SELECT_GPU", "1").lower() not in ("0", "false")
    if not auto_select or not torch.cuda.is_available():
        return requested_gpu

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        return requested_gpu

    best_gpu = requested_gpu
    max_free = 0
    for gid in range(num_gpus):
        try:
            free, total = torch.cuda.mem_get_info(gid)
            if free > max_free:
                max_free = free
                best_gpu = gid
        except Exception as e:
            logger.warning(f"Failed to query GPU {gid}: {e}")

    if best_gpu != requested_gpu:
        logger.info(
            f"Auto-selected GPU {best_gpu} (free VRAM: {max_free / (1024**2):.1f} MiB) "
            f"over requested GPU {requested_gpu}"
        )
    return best_gpu


def _load_model(gpu_id: int = 0):
    """Load the Gemma 4 model and tokenizer (lazy, thread-safe)."""
    global _model, _tokenizer, _device

    if _model is not None:
        return

    with _load_lock:
        if _model is not None:
            return

        logger.info(f"Loading {MODEL_NAME}...")
        start = time.perf_counter()

        hf_token = _get_hf_token()
        offline_mode = os.environ.get("TRANSFORMERS_OFFLINE", "1") == "1"

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            local_files_only=offline_mode,
            token=hf_token,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        gpu_id = _select_gpu(gpu_id)
        use_cuda = torch.cuda.is_available() and gpu_id < torch.cuda.device_count()

        if use_cuda:
            dtype = (
                torch.bfloat16
                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            device = f"cuda:{gpu_id}"
            try:
                torch.cuda.set_device(gpu_id)
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    device_map={"": device},
                    local_files_only=offline_mode,
                    token=hf_token,
                )
            except Exception:
                logger.warning("device_map loading failed, falling back to .to(device)")
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    local_files_only=offline_mode,
                    token=hf_token,
                ).to(device)
        else:
            device = "cpu"
            dtype = torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=dtype,
                local_files_only=offline_mode,
                token=hf_token,
            ).to(device)

        model.eval()
        elapsed = time.perf_counter() - start
        logger.info(f"{MODEL_NAME} loaded on {device} in {elapsed:.1f}s")

        _model = model
        _tokenizer = tokenizer
        _device = device


# ---------- Request / Response ----------

class TranslateRequest(BaseModel):
    text: str
    source_language: str
    target_language: str
    gpu_id: int = Field(default=0)
    batch_size: int = Field(default=4)


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok", "engine": "gemma", "model": MODEL_NAME}


@app.post("/translate")
def translate(req: TranslateRequest):
    try:
        _load_model(req.gpu_id)
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")
        err_msg = str(e)
        if "offline" in err_msg.lower() or "local_files" in err_msg.lower():
            raise HTTPException(
                status_code=503,
                detail=f"Model not available offline. Run setup_and_run.sh to download. Error: {err_msg}",
            )
        raise HTTPException(status_code=500, detail=f"Failed to load model: {err_msg}")

    try:
        lines = req.text.split("\n")
        translated_lines = []

        for i in range(0, len(lines), req.batch_size):
            batch = lines[i : i + req.batch_size]
            valid = [(idx, line) for idx, line in enumerate(batch) if line.strip()]

            if not valid:
                translated_lines.extend(batch)
                continue

            results = [""] * len(batch)
            valid_indices, valid_texts = zip(*valid)

            prompts = []
            for text_item in valid_texts:
                user_msg = (
                    f"Translate the following text from {req.source_language} to {req.target_language}. "
                    f"Provide only the direct translation without any explanation, notes, or additional commentary:\n\n{text_item}"
                )
                if hasattr(_tokenizer, "apply_chat_template") and getattr(_tokenizer, "chat_template", None):
                    try:
                        p = _tokenizer.apply_chat_template(
                            [{"role": "user", "content": user_msg}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        prompts.append(p)
                        continue
                    except Exception:
                        pass
                prompts.append(f"<start_of_turn>user\n{user_msg}<end_of_turn>\n<start_of_turn>model\n")

            inputs = _tokenizer(
                prompts,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            if _device:
                inputs = inputs.to(_device)

            with torch.no_grad():
                input_len = inputs.input_ids.shape[1]
                pad_id = getattr(_tokenizer, "pad_token_id", None) or getattr(_tokenizer, "eos_token_id", None)
                generated = _model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
                new_tokens = generated[:, input_len:] if generated.shape[1] >= input_len else generated
                translations = _tokenizer.batch_decode(
                    new_tokens.detach().cpu().tolist(),
                    skip_special_tokens=True,
                )

            # Clean up markdown fences
            cleaned = []
            for t in translations:
                t = t.strip()
                if t.startswith("```") and t.endswith("```"):
                    lines_t = t.split("\n")
                    if len(lines_t) > 2:
                        t = "\n".join(lines_t[1:-1]).strip()
                cleaned.append(t)

            for idx, trans in zip(valid_indices, cleaned):
                results[idx] = trans

            translated_lines.extend(results)

        result_text = "\n".join(translated_lines)
        return {"text": result_text}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Translation failed: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
