"""Question answering with Gemini."""
from gemini_client import generate_text


def answer_question_with_gemini(question: str) -> str:
    prompt = (
        "You are EduGenie, a friendly tutor. Answer the student's question accurately "
        "and concisely (a short paragraph unless more detail is clearly needed).\n\n"
        f"Question: {question}"
    )
    return generate_text(prompt)