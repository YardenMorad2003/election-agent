"""
FastAPI API for the Agentic Electoral Analyst.

This keeps the LangGraph/data tooling in Python while allowing a Next.js
frontend to provide the user experience.
"""
import json
import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
CHART_DIR = ROOT_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/election-agent-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/election-agent-cache")

from agent import get_llm, run_all_configs, run_chat

app = FastAPI(title="Agentic Electoral Analyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/charts", StaticFiles(directory=str(CHART_DIR)), name="charts")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    model: str = "gpt-4o-mini"
    compare: bool = False


class AskResponse(BaseModel):
    answer: str
    tools_used: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    chart_urls: list[str] = Field(default_factory=list)
    time: float
    comparison: dict | None = None
    suggestions: list[str] = Field(default_factory=list)


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


SUGGEST_PROMPT = """Based on this conversation about elections, suggest 3 short follow-up questions the user might want to ask next.

Rules:
- Each question should be a natural follow-up to what was just discussed
- Keep questions under 60 characters each
- Vary the types: one could dig deeper, one could compare, one could ask for a chart
- If the conversation was about U.S. data, suggest U.S. follow-ups (and vice versa for Israeli)
- Return ONLY a JSON array of 3 strings, nothing else

Example: ["What about 2024?", "Show this as a chart", "Compare with rural counties"]"""


def _chart_urls(paths: list[str] | None) -> list[str]:
    urls = []
    for raw_path in paths or []:
        path = Path(raw_path)
        try:
            resolved = path.resolve()
            if resolved.is_relative_to(CHART_DIR.resolve()) and resolved.exists():
                urls.append(f"/charts/{resolved.name}")
        except OSError:
            continue
    return urls


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _generate_suggestions(messages: list[ChatMessage]) -> list[str]:
    if not messages:
        return []
    try:
        llm = get_llm(model="gpt-4o-mini", temperature=0.7)
        recent = messages[-4:]
        convo = "\n".join(
            f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content[:200]}"
            for msg in recent
        )
        resp = llm.invoke([
            SystemMessage(content=SUGGEST_PROMPT),
            HumanMessage(content=convo),
        ])
        suggestions = json.loads(_strip_json_fences(resp.content))
        if isinstance(suggestions, list):
            return [str(item) for item in suggestions[:3]]
    except Exception:
        return []
    return []


def _normalize_comparison(results: dict) -> dict:
    normalized = {}
    for name, result in results.items():
        answer = str(result.get("answer", ""))
        tools_used = _string_list(result.get("tools_used", []))
        trace = _string_list(result.get("trace", []))
        normalized[name] = {
            "answer": answer,
            "config": str(result.get("config", name)),
            "tools_used": tools_used,
            "trace": trace,
            "chart_urls": _chart_urls(result.get("chart_paths", [])),
        }
    return normalized


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    start = time.time()
    messages = [msg.model_dump() for msg in payload.messages]
    question = payload.messages[-1].content if payload.messages else ""

    if payload.compare:
        results = run_all_configs(question, model=payload.model)
        elapsed = time.time() - start
        answer = f"Here are the results from all 4 routing configs ({elapsed:.1f}s):"
        augmented_messages = [
            *payload.messages,
            ChatMessage(role="assistant", content=answer),
        ]
        return AskResponse(
            answer=answer,
            time=elapsed,
            comparison=_normalize_comparison(results),
            suggestions=_generate_suggestions(augmented_messages),
        )

    try:
        result = run_chat(messages, model=payload.model)
    except Exception as exc:
        elapsed = time.time() - start
        return AskResponse(
            answer=f"The agent hit an error while answering: {exc}",
            trace=[f"{type(exc).__name__}: {exc}"],
            time=elapsed,
        )
    elapsed = time.time() - start
    augmented_messages = [
        *payload.messages,
        ChatMessage(role="assistant", content=result["answer"]),
    ]
    return AskResponse(
        answer=str(result.get("answer", "")),
        tools_used=_string_list(result.get("tools_used", [])),
        trace=_string_list(result.get("trace", [])),
        chart_urls=_chart_urls(result.get("chart_paths", [])),
        time=elapsed,
        suggestions=_generate_suggestions(augmented_messages),
    )
