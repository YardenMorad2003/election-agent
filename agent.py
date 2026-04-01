"""
Agentic Electoral Analyst — 4 routing configurations.

Config 1: Single-Pass LLM (no tools)
Config 2: RAG-Only (retrieve chunks → LLM)
Config 3: Agent + Fixed Routing (keyword rules → tool)
Config 4: Agent + Dynamic Routing (LLM picks tools via ReAct)
"""
import os, json, sqlite3, re
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

# Support Streamlit Cloud secrets
try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from tools.data_query import make_data_query_tool, SCHEMA
from tools.coalition import coalition_calculator
from tools.israel_politics_rss import israel_politics_rss
from tools.web_search import web_search

# ── Shared LLM ──
def get_llm(model: str = "gpt-4o-mini", temperature: float = 0):
    return ChatOpenAI(model=model, temperature=temperature)


SYSTEM_PROMPT = """You are an expert analyst of Israeli Knesset elections (1996–2022).
You have access to a structured database covering 12 elections, 1,384 localities,
party-level results, and socioeconomic data for 201 municipalities.

When answering questions:
- Be precise with numbers — use the tools to look up exact figures.
- Cite which Knesset/year you're referencing.
- For coalition questions, use the coalition calculator.
- For data lookups about the election database, use the data query tool.
- For correlations, comparisons, turnout analysis, vote-share trends, locality patterns, and socioeconomic analysis, prefer the data query tool even if the user casually mentions "web" or "search".
- For current Israeli political developments, breaking updates, and recent events, use the RSS news tool first.
- For external background or facts outside the database that are not time-sensitive, use web search.
- The database only covers election data through 2022. Do not use the data query tool for current office holders, biographies, general background, or news.
- If a tool has already returned useful information, trust that tool output instead of falling back to training knowledge.
- Never say you cannot browse the web or mention a knowledge cutoff after using a tool.
- Do not call the same tool repeatedly with the exact same query unless the previous call returned an error or no results.
- Give concise answers.
"""


WEB_SYNTHESIS_PROMPT = """You are writing the final answer from a web-search tool output.

Rules:
- Use ONLY the tool output provided.
- If the tool output starts with STATUS: OK, answer directly from those results.
- If the tool output starts with STATUS: NO_RESULTS or STATUS: ERROR, say the web search tool did not return usable results.
- Do NOT mention training data, knowledge cutoffs, or lack of internet access.
- Summarize the snippets into 2-4 concrete takeaways when possible.
- Do not tell the user to go read the sources unless the tool output is too thin to support even a short summary.
- Keep the answer concise and cite the source names or URLs when available.
"""


RSS_SYNTHESIS_PROMPT = """You are writing the final answer from an RSS news tool output.

Rules:
- Use ONLY the tool output provided.
- If the tool output starts with STATUS: OK, summarize the freshest relevant developments directly.
- If the tool output starts with STATUS: NO_RESULTS or STATUS: ERROR, say the RSS news tool did not return usable recent updates.
- Prefer the newest items and mention concrete timing when available.
- For each item you mention, include the publisher name and published timestamp if available.
- When linking, use direct markdown links in the form `[Read more](URL)` or `[Title](URL)`.
- Never write vague placeholders like `Read more here`.
- Keep the answer concise and cite the source names or URLs when available.
"""


# ═══════════════════════════════════════════════════════
# CONFIG 1: Single-Pass LLM (no tools, pure baseline)
# ═══════════════════════════════════════════════════════
def run_single_pass(question: str, llm: ChatOpenAI | None = None) -> dict:
    llm = llm or get_llm()
    resp = llm.invoke([
        SystemMessage(content="You are an expert on Israeli Knesset elections (1996-2022). "
                      "Answer the question using only your training knowledge. "
                      "If you're not sure about exact numbers, say so."),
        HumanMessage(content=question),
    ])
    return {
        "answer": resp.content,
        "config": "single_pass",
        "tools_used": [],
        "trace": ["LLM answered directly (no tools)"],
    }


# ═══════════════════════════════════════════════════════
# CONFIG 2: RAG-Only (retrieve data chunks → LLM)
# ═══════════════════════════════════════════════════════
_rag_chunks = None

def _build_rag_chunks():
    """Build text chunks from the database for RAG retrieval."""
    global _rag_chunks
    if _rag_chunks is not None:
        return _rag_chunks

    db_path = os.path.join(os.path.dirname(__file__), "elections.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    chunks = []

    # Election summaries
    for row in conn.execute("SELECT * FROM elections ORDER BY knesset"):
        chunks.append(
            f"Knesset {row['knesset']} ({row['year']}): "
            f"Eligible voters: {row['total_eligible']:,}. Turnout: {row['turnout_pct']}%. "
            f"Right: {row['right_pct']}%, Haredi: {row['haredi_pct']}%, "
            f"Center: {row['center_pct']}%, Left: {row['left_pct']}%, "
            f"Arab: {row['arab_pct']}%. "
            f"Right+Haredi bloc: {row['right_haredi_pct']}%, "
            f"Center+Left+Arab bloc: {row['center_left_arab_pct']}%."
        )

    # Party results per election
    for row in conn.execute(
        "SELECT knesset, name, bloc, vote_pct, seats FROM parties WHERE seats>0 ORDER BY knesset, seats DESC"
    ):
        chunks.append(
            f"K{row['knesset']}: {row['name']} ({row['bloc']}) — "
            f"{row['vote_pct']}% of votes, {row['seats']} seats."
        )

    # Socioeconomic summaries
    for row in conn.execute("SELECT * FROM socioeconomic LIMIT 201"):
        chunks.append(
            f"Socioeconomic — {row['name']}: pop {row['population']:.0f}, "
            f"median age {row['median_age']}, "
            f"academic degree {row['pct_academic_degree']:.1f}%, "
            f"income/capita {row['avg_monthly_income_per_capita']:.0f} NIS."
        )

    conn.close()
    _rag_chunks = chunks
    return chunks


def _simple_retrieve(question: str, top_k: int = 15) -> list[str]:
    """Keyword-based retrieval (no embeddings needed for demo — fast and deterministic)."""
    chunks = _build_rag_chunks()
    q_lower = question.lower()

    # extract keywords
    keywords = set(re.findall(r'[a-zA-Z\u0590-\u05FF]{2,}', q_lower))
    # also match knesset numbers
    knums = re.findall(r'[kK](\d{2})', question)
    keywords.update(knums)

    scored = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for kw in keywords if kw in chunk_lower)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:top_k]]


def run_rag_only(question: str, llm: ChatOpenAI | None = None) -> dict:
    llm = llm or get_llm()
    retrieved = _simple_retrieve(question)
    context = "\n".join(retrieved)

    resp = llm.invoke([
        SystemMessage(content="You are an expert on Israeli Knesset elections. "
                      "Answer the question using ONLY the retrieved context below. "
                      "If the context doesn't contain enough info, say so.\n\n"
                      f"RETRIEVED CONTEXT:\n{context}"),
        HumanMessage(content=question),
    ])
    return {
        "answer": resp.content,
        "config": "rag_only",
        "tools_used": ["rag_retrieval"],
        "trace": [f"Retrieved {len(retrieved)} chunks", "LLM synthesized answer from context"],
        "retrieved_chunks": retrieved,
    }


# ═══════════════════════════════════════════════════════
# CONFIG 3: Agent + Fixed Routing (keyword rules)
# ═══════════════════════════════════════════════════════
def _classify_question(question: str) -> str:
    q = question.lower()
    coalition_kw = ["coalition", "government", "majority", "61 seats", "form a",
                    "combine", "party combination", "קואליציה"]
    data_kw = [
        "correlation", "correlat", "academic degree", "education", "left-bloc",
        "left bloc", "turnout", "vote share", "voting pattern", "compare",
        "comparison", "average", "median", "trend", "locality", "municipality",
        "socioeconomic", "regression", "seats", "votes",
    ]
    web_kw = [
        "current", "currently", "latest", "recent", "today", "news", "search",
        "web", "wikipedia", "who is", "prime minister", "background", "history",
        "outside the database", "not in the database",
    ]
    rss_kw = [
        "rss", "headline", "headlines", "breaking", "update", "updates",
        "real-time", "realtime", "live", "developments",
    ]
    current_role_kw = [
        "who is the current", "who is currently", "current prime minister",
        "current president", "current knesset speaker", "knesset speaker",
        "current finance minister", "current defense minister",
        "current foreign minister", "current minister", "office holder",
    ]
    person_news_kw = [
        "news about", "recent news about", "latest news about",
        "most recent news about", "what happened to", "what's new with",
    ]
    recency_kw = ["latest", "recent", "most recent", "today", "news", "update", "updates"]
    if any(kw in q for kw in coalition_kw):
        return "coalition"
    if any(kw in q for kw in data_kw):
        return "data_query"
    if any(kw in q for kw in current_role_kw) and any(kw in q for kw in person_news_kw):
        return "web_search"
    if any(kw in q for kw in current_role_kw) and any(kw in q for kw in recency_kw) and "about" in q:
        return "web_search"
    if any(kw in q for kw in current_role_kw):
        return "web_search"
    if any(kw in q for kw in rss_kw):
        return "rss_news"
    if any(kw in q for kw in web_kw):
        if "who is" in q:
            return "web_search"
        if any(token in q for token in ["latest", "recent", "today", "news"]):
            return "rss_news"
        return "web_search"
    return "data_query"


def run_fixed_routing(question: str, llm: ChatOpenAI | None = None) -> dict:
    llm = llm or get_llm()
    route = _classify_question(question)
    data_query_tool = make_data_query_tool(llm)

    if route == "coalition":
        tool_result = coalition_calculator.invoke(question)
        tool_name = "coalition_calculator"
    elif route == "rss_news":
        tool_result = israel_politics_rss.invoke(question)
        tool_name = "israel_politics_rss"
        if str(tool_result).startswith(("STATUS: NO_RESULTS", "STATUS: ERROR")):
            tool_result = web_search.invoke(question)
            tool_name = "web_search"
    elif route == "web_search":
        tool_result = web_search.invoke(question)
        tool_name = "web_search"
    else:
        tool_result = data_query_tool.invoke(question)
        tool_name = "data_query"

    # synthesize final answer
    synthesis_prompt = SYSTEM_PROMPT
    if tool_name == "web_search":
        synthesis_prompt = WEB_SYNTHESIS_PROMPT
    elif tool_name == "israel_politics_rss":
        synthesis_prompt = RSS_SYNTHESIS_PROMPT
    resp = llm.invoke([
        SystemMessage(content=synthesis_prompt),
        HumanMessage(content=f"Question: {question}\n\nTool ({tool_name}) returned:\n{tool_result}\n\n"
                     "Provide a clear, well-formatted answer based on the tool output."),
    ])
    return {
        "answer": resp.content,
        "config": "fixed_routing",
        "tools_used": [tool_name],
        "trace": [f"Keyword routing → {tool_name}", f"Tool returned {len(str(tool_result))} chars",
                  "LLM synthesized final answer"],
        "tool_output": tool_result,
    }


# ═══════════════════════════════════════════════════════
# CONFIG 4: Agent + Dynamic Routing (LLM picks tools)
# ═══════════════════════════════════════════════════════
def run_dynamic_routing(question: str, llm: ChatOpenAI | None = None) -> dict:
    llm = llm or get_llm()
    route_hint = _classify_question(question)
    data_query_tool = make_data_query_tool(llm)
    tools = [data_query_tool, coalition_calculator, israel_politics_rss, web_search]

    if route_hint in {"data_query", "coalition"}:
        if route_hint == "coalition":
            tool_result = coalition_calculator.invoke(question)
            tool_name = "coalition_calculator"
        else:
            tool_result = data_query_tool.invoke(question)
            tool_name = "data_query"

        resp = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nTool ({tool_name}) returned:\n{tool_result}\n\n"
                         "Provide a clear, well-formatted answer based on the tool output."),
        ])
        return {
            "answer": resp.content,
            "config": "dynamic_routing",
            "tools_used": [tool_name],
            "trace": [
                f"Dynamic guardrail routed obvious question → {tool_name}",
                f"Tool returned {len(str(tool_result))} chars",
                "LLM synthesized final answer",
            ],
            "tool_output": tool_result,
        }

    if route_hint == "rss_news":
        tool_result = israel_politics_rss.invoke(question)
        tool_name = "israel_politics_rss"
        if str(tool_result).startswith(("STATUS: NO_RESULTS", "STATUS: ERROR")):
            tool_result = web_search.invoke(question)
            tool_name = "web_search"

        resp = llm.invoke([
            SystemMessage(content=RSS_SYNTHESIS_PROMPT if tool_name == "israel_politics_rss" else WEB_SYNTHESIS_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nTool ({tool_name}) returned:\n{tool_result}\n\n"
                         "Provide a clear, well-formatted answer based on the tool output."),
        ])
        return {
            "answer": resp.content,
            "config": "dynamic_routing",
            "tools_used": [tool_name],
            "trace": [
                f"Dynamic guardrail routed fresh-news question → {tool_name}",
                f"Tool returned {len(str(tool_result))} chars",
                "LLM synthesized final answer",
            ],
            "tool_output": tool_result,
        }

    if route_hint == "web_search":
        tool_result = web_search.invoke(question)
        tool_name = "web_search"

        resp = llm.invoke([
            SystemMessage(content=WEB_SYNTHESIS_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nTool ({tool_name}) returned:\n{tool_result}\n\n"
                         "Provide a clear, well-formatted answer based on the tool output."),
        ])
        return {
            "answer": resp.content,
            "config": "dynamic_routing",
            "tools_used": [tool_name],
            "trace": [
                f"Dynamic guardrail routed web question → {tool_name}",
                f"Tool returned {len(str(tool_result))} chars",
                "LLM synthesized final answer",
            ],
            "tool_output": tool_result,
        }

    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    # extract trace
    messages = result["messages"]
    trace = []
    tools_used = []
    final_answer = ""
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                trace.append(f"LLM decided to call: {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)[:200]})")
                tools_used.append(tc["name"])
        if msg.type == "tool":
            trace.append(f"Tool '{msg.name}' returned {len(msg.content)} chars")
        if msg.type == "ai" and not getattr(msg, "tool_calls", None):
            final_answer = msg.content

    return {
        "answer": final_answer,
        "config": "dynamic_routing",
        "tools_used": list(dict.fromkeys(tools_used)),
        "trace": trace,
    }


# ═══════════════════════════════════════════════════════
# Unified runner
# ═══════════════════════════════════════════════════════
CONFIGS = {
    "single_pass": run_single_pass,
    "rag_only": run_rag_only,
    "fixed_routing": run_fixed_routing,
    "dynamic_routing": run_dynamic_routing,
}

def run_question(question: str, config: str = "dynamic_routing",
                 model: str = "gpt-4o-mini") -> dict:
    llm = get_llm(model=model)
    return CONFIGS[config](question, llm)


def run_all_configs(question: str, model: str = "gpt-4o-mini") -> dict:
    llm = get_llm(model=model)
    results = {}
    for name, fn in CONFIGS.items():
        try:
            results[name] = fn(question, llm)
        except Exception as e:
            results[name] = {
                "answer": f"Error: {e}",
                "config": name,
                "tools_used": [],
                "trace": [f"Error: {e}"],
            }
    return results
