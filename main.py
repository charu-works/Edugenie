"""EduGenie – FastAPI application (routes + wiring)."""
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException
from diagram_module import generate_diagram
from explanation_module import explain_topic, preload_model
from gemini_client import AIServiceError
from learning_path import get_learning_recommendations
from qna import answer_question_with_gemini
from quiz_module import generate_quiz
from summary_module import summarize_text

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
MAX_INPUT_CHARS = 20000


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("PRELOAD_LOCAL_MODEL", "1") == "1" and os.getenv("EXPLAIN_BACKEND", "local").lower() == "local":
        threading.Thread(target=preload_model, daemon=True).start()
    yield


app = FastAPI(title="EduGenie", description="Gemini-powered learning assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ---------- Error handling: always return {"error": "..."} ----------
@app.exception_handler(AIServiceError)
async def ai_error_handler(request: Request, exc: AIServiceError):
    return JSONResponse({"error": str(exc)}, status_code=503)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


async def _read_field(request: Request, field: str, message: str) -> str:
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.")
    value = data.get(field) if isinstance(data, dict) else None
    value = value.strip() if isinstance(value, str) else ""
    if not value:
        raise HTTPException(status_code=400, detail=message)
    if len(value) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=400, detail=f"Input too long (max {MAX_INPUT_CHARS} characters).")
    return value


# ---------- Pages ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------- API ----------
@app.get("/qa")
async def answer_question(question: str = Query(..., min_length=1, max_length=2000)):
    answer = await run_in_threadpool(answer_question_with_gemini, question.strip())
    return {"answer": answer}


@app.post("/explain/")
@app.post("/explain", include_in_schema=False)
async def explain_api(request: Request):
    topic = await _read_field(request, "topic", "Please provide a topic.")
    explanation = await run_in_threadpool(explain_topic, topic)
    return {"topic": topic, "explanation": explanation}


@app.post("/summarize/")
@app.post("/summarize", include_in_schema=False)
async def summarize_api(request: Request):
    text = await _read_field(request, "text", "Please provide text to summarize.")
    summary = await run_in_threadpool(summarize_text, text)
    return {"summary": summary}


@app.post("/quiz")
@app.post("/quiz/", include_in_schema=False)
async def quiz_api(request: Request):
    text = await _read_field(request, "text", "Please provide text for quiz.")
    quiz = await run_in_threadpool(generate_quiz, text)
    return {"quiz": quiz}

@app.post("/diagram")
@app.post("/diagram/", include_in_schema=False)
async def diagram_api(request: Request):
    topic = await _read_field(request, "topic", "Please describe the diagram you want.")
    data = await request.json()
    diagram_type = data.get("type", "auto") if isinstance(data, dict) else "auto"
    result = await run_in_threadpool(generate_diagram, topic, diagram_type)
    return {"topic": topic, **result}
@app.get("/learn/recommendations")
async def learning_recommendation_api(topic: str = Query(..., min_length=1, max_length=500)):
    recommendation = await run_in_threadpool(get_learning_recommendations, topic.strip())
    return {"topic": topic, "recommendation": recommendation}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)