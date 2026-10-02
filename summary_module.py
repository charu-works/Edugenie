"""Summarization with Gemini."""
from gemini_client import generate_text


def summarize_text(text: str) -> str:
    prompt = (
        "Summarize the following text in simple language. Keep the key points, "
        "remove redundancy, and use short sentences or bullet points.\n\n"
        f"{text}"
    )
    return generate_text(prompt)