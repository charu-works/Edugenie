"""Quiz generation (3 MCQs, 4 options each) with Gemini, returned as parsed JSON."""
import json
import re

from gemini_client import AIServiceError, generate_text

QUIZ_PROMPT = """You are a quiz generator.

From the following passage or topic, create 3 multiple-choice questions. Each question should include:
- A "question"
- A list of exactly 4 "options"
- A correct "answer" that must exactly match one of the options.

Format your output as **valid JSON only** (no extra text), like this:
[
  {
    "question": "What is ...?",
    "options": ["A", "B", "C", "D"],
    "answer": "A"
  }
]

Passage or topic:
__TEXT__
"""


def clean_json_block(text: str) -> str:
    """Remove Markdown json code fences if present."""
    return re.sub(r"```(?:json)?\s*\n(.*?)```", r"\1", text, flags=re.DOTALL).strip()


def _parse(raw: str):
    cleaned = clean_json_block(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _normalize(item) -> dict:
    if not isinstance(item, dict):
        raise ValueError("question is not an object")
    question = str(item.get("question", "")).strip()
    options = [str(o).strip() for o in item.get("options", [])]
    answer = str(item.get("answer", "")).strip()
    if not question or len(options) != 4 or len(set(options)) != 4:
        raise ValueError("question must have text and 4 distinct options")

    if answer not in options:
        lowered = [o.lower() for o in options]
        if answer.lower() in lowered:
            answer = options[lowered.index(answer.lower())]
        elif re.fullmatch(r"[A-Da-d][.)]?", answer):  # e.g. "B" or "b)"
            answer = options[ord(answer[0].upper()) - ord("A")]
        else:
            raise ValueError("answer does not match any option")
    return {"question": question, "options": options, "answer": answer}


def generate_quiz(text: str) -> list:
    raw = generate_text(QUIZ_PROMPT.replace("__TEXT__", text), json_output=True)
    try:
        data = _parse(raw)
        if isinstance(data, dict):
            data = data.get("questions") or data.get("quiz") or []
        quiz = [_normalize(q) for q in data][:3]
        if not quiz:
            raise ValueError("no questions returned")
        return quiz
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        raise AIServiceError(f"Could not parse quiz from model output: {exc}. Raw output: {raw[:300]}") from exc