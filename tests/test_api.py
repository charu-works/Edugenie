"""Offline tests: Gemini is mocked, so no API key or internet is needed."""
import json

import pytest
from fastapi.testclient import TestClient

import explanation_module
import main
import quiz_module
from gemini_client import AIServiceError

client = TestClient(main.app)


# ---------- pages ----------
def test_home_page_renders():
    r = client.get("/")
    assert r.status_code == 200 and "EduGenie" in r.text


def test_static_css_served():
    assert client.get("/static/style.css").status_code == 200


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


# ---------- endpoints ----------
def test_qa(monkeypatch):
    monkeypatch.setattr(main, "answer_question_with_gemini", lambda q: "The Pacific Ocean.")
    r = client.get("/qa", params={"question": "Which is the largest ocean?"})
    assert r.status_code == 200 and r.json() == {"answer": "The Pacific Ocean."}


def test_qa_requires_question():
    assert client.get("/qa").status_code == 422


@pytest.mark.parametrize("path", ["/explain", "/explain/"])
def test_explain(monkeypatch, path):
    monkeypatch.setattr(main, "explain_topic", lambda t: f"About {t}")
    r = client.post(path, json={"topic": "gravity"})
    assert r.status_code == 200 and r.json()["explanation"] == "About gravity"


def test_explain_missing_topic():
    r = client.post("/explain/", json={})
    assert r.status_code == 400 and "topic" in r.json()["error"].lower()


def test_invalid_json_body():
    r = client.post("/summarize/", content="not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_summarize(monkeypatch):
    monkeypatch.setattr(main, "summarize_text", lambda t: "short")
    assert client.post("/summarize/", json={"text": "long text"}).json() == {"summary": "short"}


def test_input_too_long():
    r = client.post("/summarize/", json={"text": "x" * (main.MAX_INPUT_CHARS + 1)})
    assert r.status_code == 400


def test_quiz_endpoint(monkeypatch):
    sample = [{"question": "Q?", "options": ["a", "b", "c", "d"], "answer": "a"}]
    monkeypatch.setattr(main, "generate_quiz", lambda t: sample)
    assert client.post("/quiz", json={"text": "Solar system"}).json() == {"quiz": sample}


def test_recommendations(monkeypatch):
    monkeypatch.setattr(main, "get_learning_recommendations", lambda t: "## Plan")
    r = client.get("/learn/recommendations", params={"topic": "SQL"})
    assert r.json() == {"topic": "SQL", "recommendation": "## Plan"}


def test_ai_failure_returns_503(monkeypatch):
    def boom(q):
        raise AIServiceError("GEMINI_API_KEY is not set.")

    monkeypatch.setattr(main, "answer_question_with_gemini", boom)
    r = client.get("/qa", params={"question": "hi"})
    assert r.status_code == 503 and "GEMINI_API_KEY" in r.json()["error"]


# ---------- quiz parsing ----------
def _quiz_json(answer="B"):
    return json.dumps([
        {"question": f"Q{i}?", "options": ["w", "x", "y", "z"], "answer": answer} for i in range(3)
    ])


def test_quiz_parses_fenced_json_and_letter_answers(monkeypatch):
    fence = "`" * 3
    monkeypatch.setattr(quiz_module, "generate_text", lambda *a, **k: fence + "json\n" + _quiz_json("B") + "\n" + fence)
    quiz = quiz_module.generate_quiz("topic")
    assert len(quiz) == 3 and all(q["answer"] == "x" for q in quiz)


def test_quiz_rejects_bad_output(monkeypatch):
    monkeypatch.setattr(quiz_module, "generate_text", lambda *a, **k: "Sorry, I can't do that.")
    with pytest.raises(AIServiceError):
        quiz_module.generate_quiz("topic")


# ---------- explanation fallback ----------
def test_explain_uses_gemini_backend(monkeypatch):
    monkeypatch.setattr(explanation_module, "generate_text", lambda p, **k: "gemini explanation")
    assert explanation_module.explain_topic("atoms") == "gemini explanation"
# ---------- diagrams ----------
import diagram_module  # noqa: E402


def test_diagram_endpoint(monkeypatch):
    monkeypatch.setattr(main, "generate_diagram", lambda t, k: {"type": k, "mermaid": "flowchart TD\n A-->B"})
    r = client.post("/diagram", json={"topic": "login flow", "type": "flowchart"})
    assert r.status_code == 200
    assert r.json() == {"topic": "login flow", "type": "flowchart", "mermaid": "flowchart TD\n A-->B"}


def test_diagram_requires_topic():
    assert client.post("/diagram", json={"type": "flowchart"}).status_code == 400


def test_diagram_type_defaults_to_auto(monkeypatch):
    seen = {}
    monkeypatch.setattr(main, "generate_diagram", lambda t, k: seen.update(kind=k) or {"type": k, "mermaid": "mindmap"})
    client.post("/diagram", json={"topic": "x"})
    assert seen["kind"] == "auto"


def test_clean_mermaid_strips_fences_and_chatter():
    fence = "`" * 3
    raw = "Sure! Here you go:\n" + fence + "mermaid\nflowchart TD\n  A[\"Start\"] --> B[\"End\"]\n" + fence
    assert diagram_module.clean_mermaid(raw) == 'flowchart TD\n  A["Start"] --> B["End"]'


def test_clean_mermaid_rejects_non_diagram():
    with pytest.raises(ValueError):
        diagram_module.clean_mermaid("I cannot draw that.")


def test_generate_diagram_unknown_type_falls_back_to_auto(monkeypatch):
    monkeypatch.setattr(diagram_module, "generate_text", lambda p, **k: "sequenceDiagram\n A->>B: hi")
    out = diagram_module.generate_diagram("chat", "banana")
    assert out == {"type": "auto", "mermaid": "sequenceDiagram\n A->>B: hi"}


def test_generate_diagram_bad_output_raises(monkeypatch):
    monkeypatch.setattr(diagram_module, "generate_text", lambda p, **k: "no diagram here")
    with pytest.raises(AIServiceError):
        diagram_module.generate_diagram("x", "flowchart")