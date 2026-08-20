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
                detail=(
                    f"Model not available offline. "
                    f"Run setup_and_run.sh to download. Error: {err_msg}"
                ),
            )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {err_msg}",
        )

    try:
        # ---------------------------------------------------------
        # 1. Preserve the original page line structure
        # ---------------------------------------------------------
        line_break_marker = "<LINE_BREAK>"

        source_text = req.text.replace("\r\n", "\n")
        source_text = source_text.replace("\n", line_break_marker)

        # ---------------------------------------------------------
        # 2. Build the translation instruction
        # ---------------------------------------------------------
        user_msg = (
            f"Translate the following text from "
            f"{req.source_language} to {req.target_language}.\n\n"
            "Translate the text faithfully and preserve its meaning.\n"
            "Do not summarize, explain, interpret, or add information.\n"
            "Preserve names, places, dates, numbers, and proper nouns "
            "as accurately as possible.\n"
            "If the source contains grammatical errors, unusual wording, "
            "or OCR errors, translate it as faithfully as possible "
            "without inventing missing information.\n\n"

            "IMPORTANT: Any text enclosed within <dnt> and </dnt> tags "
            "must NOT be translated or modified in any way. "
            "Preserve the <dnt> tags and all text inside them exactly as "
            "they appear in the source.\n\n"

            f"The text contains the special marker {line_break_marker} "
            "to represent an original line break.\n"
            f"Preserve every {line_break_marker} marker in the output.\n"
            f"Do not add, remove, or reorder {line_break_marker} markers.\n\n"

            "Output only the translation. "
            "Do not include explanations, notes, or commentary.\n\n"
            f"{source_text}"
        )

        # ---------------------------------------------------------
        # 3. Build chat prompt
        # ---------------------------------------------------------
        if (
            hasattr(_tokenizer, "apply_chat_template")
            and getattr(_tokenizer, "chat_template", None)
        ):
            try:
                prompt = _tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = (
                    f"<start_of_turn>user\n"
                    f"{user_msg}"
                    f"<end_of_turn>\n"
                    f"<start_of_turn>model\n"
                )
        else:
            prompt = (
                f"<start_of_turn>user\n"
                f"{user_msg}"
                f"<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )

        # ---------------------------------------------------------
        # 4. Tokenize the complete page
        # ---------------------------------------------------------
        inputs = _tokenizer(
            prompt,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        if _device:
            inputs = inputs.to(_device)

        # ---------------------------------------------------------
        # 5. Generate translation
        # ---------------------------------------------------------
        with torch.no_grad():
            input_len = inputs.input_ids.shape[1]

            pad_id = (
                getattr(_tokenizer, "pad_token_id", None)
                or getattr(_tokenizer, "eos_token_id", None)
            )

            generated = _model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=pad_id,
            )

            new_tokens = (
                generated[:, input_len:]
                if generated.shape[1] >= input_len
                else generated
            )

            translation = _tokenizer.decode(
                new_tokens[0].detach().cpu().tolist(),
                skip_special_tokens=True,
            )

        # ---------------------------------------------------------
        # 6. Clean generated output
        # ---------------------------------------------------------
        translation = translation.strip()

        # Remove markdown code fences if the model produces them
        if translation.startswith("```") and translation.endswith("```"):
            translation_lines = translation.split("\n")

            if len(translation_lines) > 2:
                translation = "\n".join(
                    translation_lines[1:-1]
                ).strip()

        # ---------------------------------------------------------
        # 7. Restore original line breaks
        # ---------------------------------------------------------
        translation = translation.replace(line_break_marker, "\n")

        return {
            "text": translation,
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Translation failed: {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}",
        )
