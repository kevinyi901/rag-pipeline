# app/common/hf_utils.py
from __future__ import annotations

import os
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import settings

_tok = None
_model = None
_device = None


def _pick_device() -> str:
    # Prefer CUDA if available, else Apple MPS, else CPU
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model() -> tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    """
    Lazy-load HF tokenizer and model based on .env (HF_MODEL_ID/HF_TOKEN).
    - On CUDA: will try to use 4-bit only if bitsandbytes is available, otherwise fp16.
    - On MPS/CPU: loads in bf16/fp16 if supported, otherwise fp32.
    """
    global _tok, _model, _device
    if _tok is not None and _model is not None and _device is not None:
        return _tok, _model, _device

    model_id = settings.hf_model_id
    hf_token = settings.hf_token or None
    _device = _pick_device()

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() or _device == "mps" else torch.float32,
    }

    if _device == "cuda":
        # Try 4-bit if bitsandbytes exists; otherwise half precision
        try:
            import bitsandbytes as _  # noqa: F401
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
        except Exception:
            load_kwargs["device_map"] = "auto"
            load_kwargs["torch_dtype"] = torch.float16
    elif _device == "mps":
        # MPS prefers full/bfloat16 without quantization
        load_kwargs["device_map"] = {"": 0}  # single device mapping still ok
    else:
        # CPU
        load_kwargs["device_map"] = {"": "cpu"}

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    _tok = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=hf_token, **load_kwargs)
    _model.eval()

    return _tok, _model, _device


def generate(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
    system: Optional[str] = None,
) -> str:
    """
    Minimal text generation helper. Works with instruct models that accept plain prompts.
    If your model expects chat format, wrap it before calling (or adapt here).
    """
    tok, model, device = get_model()

    # Simple prompt; adapt if your model expects chat templates
    full_prompt = f"{system.strip()}\n\n{prompt}" if system else prompt

    inputs = tok(full_prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    elif device == "mps":
        inputs = {k: v.to("mps") for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    text = tok.decode(out[0], skip_special_tokens=True)
    # Return only the completion after the prompt
    return text[len(full_prompt):].strip() if text.startswith(full_prompt) else text.strip()
