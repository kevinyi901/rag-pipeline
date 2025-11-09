import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def _clean_text(text: str) -> str:
    text = re.sub(r"[\u0000-\u001F\u007F-\u009F]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def generate(
    prompt: str,
    system: str = "You are a helpful legal research assistant.",
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
):
    from app.common.config import settings
    model_id = settings.hf_model_id.strip()  # remove trailing spaces
    tok = AutoTokenizer.from_pretrained(model_id, use_auth_token=settings.hf_token)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        use_auth_token=settings.hf_token,
    )

    # Apply chat template
    if hasattr(tok, "apply_chat_template"):
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
        full_prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        full_prompt = f"{system}\nUser: {prompt}\nAssistant:"

    inputs = tok(full_prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.05,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    # Decode only generated tokens
    gen_tokens = output[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(gen_tokens, skip_special_tokens=True)
    return _clean_text(text)
