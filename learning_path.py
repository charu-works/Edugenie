"""Personalized learning path generation with Gemini."""
from gemini_client import generate_text


def get_learning_recommendations(topic: str) -> str:
    prompt = (
        f"You are an AI tutor. The student wants to learn about: {topic}.\n"
        "Suggest a structured and adaptive learning path including key topics, order of learning, "
        "estimated time for each stage, and resources (videos, articles, books, courses).\n"
        "Include beginner, intermediate, and advanced levels. Use Markdown headings for each level "
        "and bullet points for topics and resources. End with a few adaptive learning tips."
    )
    return generate_text(prompt)
    