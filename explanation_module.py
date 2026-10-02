"""Concept explanation using the local LaMini-Flan-T5-783M model (falls back to Gemini)."""
import logging
import os
import threading

from gemini_client import generate_text

log = logging.getLogger("edugenie.explain")

MODEL_NAME = "MBZUAI/LaMini-Flan-T5-783M"

_tokenizer = None
_model = None
_load_failed = False
_lock = threading.Lock()


def _backend() -> str:
    return os.getenv("EXPLAIN_BACKEND", "local").strip().lower()


def _load_local_model() -> bool:
    """Load the model once (thread-safe). Returns False if it can't be loaded."""
    global _tokenizer, _model, _load_failed
    with _lock:
        if _model is not None:
            return True
        if _load_failed:
            return False
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            log.info("Loading local model %s (first run downloads ~3 GB)...", MODEL_NAME)
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            _model.eval()
            log.info("Local explanation model ready.")
            return True
        except Exception as exc:
            _load_failed = True
            log.warning("Local model unavailable (%s). Falling back to Gemini.", exc)
            return False


def preload_model() -> None:
    if _backend() == "local":
        _load_local_model()


def _explain_with_gemini(topic: str) -> str:
    return generate_text(
        f"Explain the concept of '{topic}' in a simple and clear way for a school student. "
        "Keep it under 150 words."
    )


def explain_topic(topic: str) -> str:
    if _backend() != "local" or not _load_local_model():
        return _explain_with_gemini(topic)

    import torch

    prompt = f"Explain the concept of '{topic}' in a simple and clear way for a school student."
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )
    explanation = _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return explanation or _explain_with_gemini(topic)