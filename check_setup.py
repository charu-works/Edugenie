"""Quick check that your API key and model work:  python check_setup.py"""
from gemini_client import AIServiceError, generate_text

try:
    print("Gemini says:", generate_text("Reply with exactly: EduGenie is ready!"))
except AIServiceError as exc:
    print("❌", exc)
    raise SystemExit(1)