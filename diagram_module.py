"""Diagram generation: Gemini writes Mermaid code, the browser renders it."""
import re

from gemini_client import AIServiceError, generate_text

# What the user can pick (or "auto" to let the model decide).
DIAGRAM_TYPES = {
    "auto": "Choose the diagram type that best fits the request.",
    "flowchart": (
        "A process flowchart. Start with `flowchart TD`. Include a Start node, an End node, "
        "and decision nodes written like B{\"Question?\"} with labelled branches (-->|Yes| and -->|No|)."
    ),
    "architecture": (
        "A system architecture diagram. Start with `flowchart LR`. Group components into layers using "
        "`subgraph ID[\"Layer title\"]` ... `end` blocks (for example Client, Backend, AI Services, Data). "
        "Connect components with arrows showing data flow."
    ),
    "sequence": "A sequence diagram. Start with `sequenceDiagram` and use participants and ->> messages.",
    "mindmap": "A mind map. Start with `mindmap` and use a root node written like root((Topic)) with indented branches.",
    "er": "An entity relationship diagram. Start with `erDiagram`.",
    "class": "A UML class diagram. Start with `classDiagram`.",
    "state": "A state diagram. Start with `stateDiagram-v2`.",
}

# A valid Mermaid diagram must begin with one of these keywords.
_START_KEYWORDS = (
    "flowchart", "graph", "sequencediagram", "classdiagram", "statediagram",
    "erdiagram", "mindmap", "journey", "gantt", "pie", "timeline",
)

DIAGRAM_PROMPT = """You are a diagram generator that outputs Mermaid.js code.

User request: __TOPIC__

Diagram type: __TYPE_RULE__

Strict rules:
- Output ONLY the Mermaid code. No Markdown fences, no explanations, no comments.
- The first line must be the Mermaid diagram keyword.
- Use short node IDs (A, B, C1, ...). Put every node label in double quotes, e.g. A["Start"].
- Do NOT use parentheses, quotes, colons, semicolons, ampersands or HTML inside labels. Keep labels short.
- Use between 6 and 14 nodes so the diagram stays readable.
"""


def normalize_type(diagram_type) -> str:
    key = str(diagram_type or "auto").strip().lower()
    return key if key in DIAGRAM_TYPES else "auto"


def clean_mermaid(text: str) -> str:
    """Strip Markdown fences / chatter and return only the Mermaid code."""
    text = re.sub(r"```(?:mermaid)?", "", text, flags=re.IGNORECASE).strip()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower().startswith(_START_KEYWORDS):
            return "\n".join(lines[i:]).strip()
    raise ValueError("output does not start with a Mermaid diagram keyword")


def generate_diagram(topic: str, diagram_type: str = "auto") -> dict:
    kind = normalize_type(diagram_type)
    prompt = DIAGRAM_PROMPT.replace("__TOPIC__", topic).replace("__TYPE_RULE__", DIAGRAM_TYPES[kind])
    raw = generate_text(prompt, temperature=0.3)
    try:
        code = clean_mermaid(raw)
    except ValueError as exc:
        raise AIServiceError(f"Could not create a diagram: {exc}. Raw output: {raw[:200]}") from exc
    return {"type": kind, "mermaid": code}