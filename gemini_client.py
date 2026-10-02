"""Shared Gemini client used by every cloud-powered module."""
import logging
import os
import threading
import time

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("edugenie.gemini")

FALLBACK_MODELS = [
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
]
RETRIES_PER_MODEL = 3      # attempts per model before switching to the next one
RETRY_DELAY_SECONDS = 1.5  # doubles after each failed attempt (1.5s, 3s, ...)

_client = None
_working_model = None
_lock = threading.Lock()


class AIServiceError(Exception):
    """Raised when the AI backend is unavailable or returns unusable output."""


def _get_client():
    global _client
    with _lock:
        if _client is None:
            key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key or key.startswith("your_"):
                raise AIServiceError(
                    "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
                )
            from google import genai

            _client = genai.Client(api_key=key)
        return _client


def _candidate_models() -> list:
    ordered = [_working_model, os.getenv("GEMINI_MODEL", "").strip(), *FALLBACK_MODELS]
    seen, result = set(), []
    for m in ordered:
        if m and m not in seen:
            seen.add(m)
            result.append(m)
    return result


def _is_model_unavailable(message: str) -> bool:
    m = message.lower()
    return "404" in m or "not_found" in m or "not found" in m or "no longer available" in m


def _is_transient(message: str) -> bool:
    """Temporary Google-side problems worth retrying (overload, rate limit, timeouts)."""
    m = message.lower()
    return any(
        token in m
        for token in ("503", "unavailable", "overloaded", "high demand", "429",
                      "resource_exhausted", "500", "internal", "504", "deadline", "timed out", "timeout")
    )


def generate_text(prompt: str, *, json_output: bool = False, temperature=None) -> str:
    """Send a prompt to Gemini and return the response text."""
    global _working_model
    from google.genai import types

    client = _get_client()
    cfg = {}
    if json_output:
        cfg["response_mime_type"] = "application/json"
    if temperature is not None:
        cfg["temperature"] = temperature
    config = types.GenerateContentConfig(**cfg) if cfg else None

    last_error = None
    for model in _candidate_models():
        for attempt in range(RETRIES_PER_MODEL):
            try:
                response = client.models.generate_content(model=model, contents=prompt, config=config)
            except Exception as exc:  # network, quota, bad model name, etc.
                message = str(exc)
                last_error = exc
                if _is_model_unavailable(message):
                    log.warning("Model %s not found, trying next model. (%s)", model, exc)
                    break  # go to next model
                if _is_transient(message):
                    log.warning("Model %s busy (attempt %d/%d): %s", model, attempt + 1, RETRIES_PER_MODEL, exc)
                    if attempt < RETRIES_PER_MODEL - 1:
                        time.sleep(RETRY_DELAY_SECONDS * (2 ** attempt))
                    continue  # retry same model, then fall through to next model
                raise AIServiceError(f"Gemini request failed: {exc}") from exc

            text = (getattr(response, "text", None) or "").strip()
            if not text:
                raise AIServiceError("Gemini returned an empty response (it may have been blocked by safety filters).")
            _working_model = model
            return text

    if last_error is not None and _is_transient(str(last_error)):
        raise AIServiceError(
            "Gemini is temporarily overloaded on all available models. Please wait a minute and try again."
        )
    raise AIServiceError(f"No Gemini model available. Set GEMINI_MODEL in .env. Last error: {last_error}")