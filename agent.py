"""
Agentic Electoral Analyst — 4 routing configurations.

Config 1: Single-Pass LLM (no tools)
Config 2: RAG-Only (retrieve chunks → LLM)
Config 3: Agent + Fixed Routing (keyword rules → tool)
Config 4: Agent + Dynamic Routing (LLM picks tools via ReAct)
"""
import os, json, re
import numpy as np
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
from langchain_openai import ChatOpenAI
from embeddings import LocalEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from tools.data_query import make_data_query_tool, SCHEMA
from tools.coalition import coalition_calculator
from tools.operational_web_search import web_search
from tools.chart import make_chart_tool, HEBREW_TO_ENGLISH
from classifiers import classify_question_zeroshot

# ── Party name alias expansion for cross-lingual retrieval ──
# Romanized Hebrew → English canonical name
_ROMANIZED_TO_ENGLISH = {
    "haavoda": "Labor", "avoda": "Labor", "ha'avoda": "Labor", "avodah": "Labor",
    "likud": "Likud", "haikud": "Likud",
    "meretz": "Meretz", "marets": "Meretz",
    "shas": "Shas",
    "kadima": "Kadima",
    "yesh atid": "Yesh Atid", "lapid": "Yesh Atid",
    "kahol lavan": "Blue & White", "kachol lavan": "Blue & White", "gantz": "Blue & White",
    "yamina": "Yamina",
    "habayit hayehudi": "Jewish Home", "bayit yehudi": "Jewish Home",
    "yahadut hatorah": "United Torah Judaism", "utj": "United Torah Judaism",
    "hatzionut hadatit": "Religious Zionism", "smotrich": "Religious Zionism",
    "kulanu": "Kulanu", "kahlon": "Kulanu",
    "yisrael beiteinu": "Yisrael Beiteinu", "lieberman": "Yisrael Beiteinu",
    "hamachane hamamlachti": "National Unity", "mamlachti": "National Unity",
    "hamachane hatzioni": "Zionist Camp", "zionist camp": "Zionist Camp",
    "hareshima hameshutefet": "Joint List", "meshutfet": "Joint List",
    "raam": "Ra'am", "ra'am": "Ra'am",
    "hadash": "Hadash",
    "balad": "Balad",
    "tikva hadasha": "New Hope", "saar": "New Hope",
    "otzma yehudit": "Jewish Power (Otzma Yehudit)", "ben gvir": "Jewish Power (Otzma Yehudit)",
    "shinui": "Shinui",
    "hatnuah": "HaTnuah", "livni": "HaTnuah",
    "hamachane hademokrati": "Democrats (Meretz)",
    "gesher": "Gesher",
    "mafdal": "NRP (Mafdal)",
}
# Reverse: English → Hebrew (from HEBREW_TO_ENGLISH)
_ENGLISH_TO_HEBREW = {}
for _heb, _eng in HEBREW_TO_ENGLISH.items():
    _ENGLISH_TO_HEBREW.setdefault(_eng, _heb)


def _expand_party_names(question: str) -> str:
    """Expand party name aliases in the query for better retrieval.

    If the query mentions a romanized or English party name, appends the
    canonical English + Hebrew names so the embedding search can match
    bilingual chunks like 'Labor (עבודה)'.
    """
    q_lower = question.lower()
    expansions = set()

    # Check romanized aliases
    for alias, eng in _ROMANIZED_TO_ENGLISH.items():
        if alias in q_lower:
            expansions.add(eng)
            heb = _ENGLISH_TO_HEBREW.get(eng)
            if heb:
                expansions.add(heb)

    # Check English names directly (e.g., user types "Labor")
    for eng, heb in _ENGLISH_TO_HEBREW.items():
        if eng.lower() in q_lower:
            expansions.add(heb)

    if not expansions:
        return question
    return question + " " + " ".join(expansions)

# ── Shared LLM ──
def get_llm(model: str = "gpt-4o-mini", temperature: float = 0):
    return ChatOpenAI(model=model, temperature=temperature)


SYSTEM_PROMPT = """You are an expert analyst of elections in the United States and Israel.

U.S. DATA: You have access to U.S. federal election results (President, House, Senate)
from 2000-2024. Presidential data is available at the county level (2000-2024) and
precinct level (2016, 2020, 2024). House and Senate data is at precinct level (2016-2020).
Each record includes NCHS urban-rural classification (Urban/Suburban/Rural).

ISRAELI DATA: You have access to Israeli Knesset election data (K14-K25, 1996-2022)
covering 1,384 localities, party-level results, and socioeconomic data for 201 municipalities.

When answering:
- Use the data query tool for factual/numerical questions that need precise SQL lookups.
- Use the context search tool for background info, definitions, dataset coverage, or to get
  context before writing a complex query.
- Use the create chart tool when the user asks for a graph, plot, chart, visualization, or trend line.
- Use the coalition calculator for Israeli coalition questions.
- For current events, external background, or facts outside the database, use web search.
- When calling web search, preserve the user's timeframe exactly. Do not append guessed years
  like 2023, 2024, 2025, or 2026 unless the user explicitly asked for that year.
- Preserve the user's semantics exactly. Do not rewrite "when was X appointed?" into
  "current X appointment date" unless the user explicitly asked about the current officeholder.
- The database only covers election data through 2022. Do not use the data query tool for current office holders, biographies, general background, or news.
- Be precise with numbers — always look them up, don't guess.
- Cite the year and geography you're referencing.
- For U.S. questions, leverage the urban_rural classification to provide urbanization analysis.
- If a tool has already returned useful information, trust that tool output instead of falling back to training knowledge.
- Never say you cannot browse the web or mention a knowledge cutoff after using a tool.
- After a successful web search, synthesize from it instead of issuing multiple paraphrased web searches.
- Do not call the same tool repeatedly with the exact same query unless the previous call returned an error or no results.
- Show your reasoning step by step.
"""


WEB_SYNTHESIS_PROMPT = """You are writing the final answer from a web-search tool output.

Rules:
- Use ONLY the tool output provided.
- If the tool output starts with STATUS: OK, answer directly from those results.
- If the tool output starts with STATUS: NO_RESULTS or STATUS: ERROR, say the web search tool did not return usable results.
- Do NOT mention training data, knowledge cutoffs, or lack of internet access.
- If the results are news items, summarize the latest developments directly instead of telling the user to visit websites.
- If the results are news items, format them as concise bullets.
- For each news bullet, include: headline, publisher, publication date if available, and a direct article link.
- Prefer markdown links on the publisher name, linking directly to the article URL.
- Prefer 3-5 concrete developments with source names and dates when available.
- Never say "various sources" or "sources like".
- Always end with a `Sources:` section listing the source names and URLs found in the tool output.
- Keep the answer concise and cite the source names or URLs when available.
"""


def _parse_web_tool_output(tool_output: str) -> dict:
    parsed = {
        "status": "",
        "method": "",
        "window": "",
        "query_used": "",
        "items": [],
    }
    current_item = None
    for raw_line in str(tool_output).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("STATUS:"):
            parsed["status"] = line.replace("STATUS:", "", 1).strip()
            continue
        if line.startswith("Method:"):
            parsed["method"] = line.replace("Method:", "", 1).strip()
            continue
        if line.startswith("Window:"):
            parsed["window"] = line.replace("Window:", "", 1).strip()
            continue
        if line.startswith("Query Used:"):
            parsed["query_used"] = line.replace("Query Used:", "", 1).strip()
            continue
        if re.match(r"^\d+\.\s+", line):
            if current_item:
                parsed["items"].append(current_item)
            current_item = {
                "title": re.sub(r"^\d+\.\s+", "", line).strip(),
                "source": "",
                "snippet": "",
                "published": "",
                "url": "",
            }
            continue
        if current_item is None:
            continue
        if line.startswith("Source:"):
            current_item["source"] = line.replace("Source:", "", 1).strip()
        elif line.startswith("Snippet:"):
            current_item["snippet"] = line.replace("Snippet:", "", 1).strip()
        elif line.startswith("Published:"):
            current_item["published"] = line.replace("Published:", "", 1).strip()
        elif line.startswith("URL:"):
            current_item["url"] = line.replace("URL:", "", 1).strip()
    if current_item:
        parsed["items"].append(current_item)
    return parsed


def _is_news_web_result(parsed: dict) -> bool:
    return parsed.get("method") == "RSS news search" or bool(parsed.get("window"))


def _has_fresh_news_intent(question: str) -> bool:
    q = question.lower()
    freshness_hints = (
        "latest", "recent", "today", "current", "breaking",
        "last 24 hours", "last day", "last week", "this week",
    )
    news_hints = ("news", "headline", "headlines", "update", "updates", "developments")
    return any(hint in q for hint in freshness_hints) or any(hint in q for hint in news_hints)


def _is_background_web_question(question: str) -> bool:
    q = question.lower()
    if _has_fresh_news_intent(q):
        return False
    return any(phrase in q for phrase in [
        "background on", "tell me about", "give me background", "overview of",
        "history of", "who are the", "explain the",
    ])


def _format_sources(items: list[dict], inline: bool = False, prefer_title: bool = False) -> str:
    seen_urls = set()
    entries = []
    for item in items:
        url = item.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        if prefer_title:
            label = item.get("title") or item.get("source") or url
        else:
            label = item.get("source") or item.get("title") or url
        entries.append(f"[{label}]({url})")

    if not entries:
        return "Sources: unavailable"
    if inline:
        return "Sources: " + ", ".join(entries)
    lines = ["Sources:"]
    lines.extend(f"- {entry}" for entry in entries)
    return "\n".join(lines)


def _normalized_word_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z]{3,}", str(text).lower())
        if token not in {
            "the", "and", "for", "with", "that", "this", "from", "into", "about",
            "after", "before", "over", "under", "amid", "latest", "news", "article",
            "report", "reports",
        }
    }


def _is_title_heavy_summary(summary: str, title: str) -> bool:
    summary_words = _normalized_word_set(summary)
    title_words = _normalized_word_set(title)
    if not summary_words or not title_words:
        return False
    overlap = len(summary_words & title_words)
    ratio = overlap / max(1, len(summary_words))
    return ratio >= 0.7


def _summarize_news_items(items: list[dict], llm: ChatOpenAI) -> list[str]:
    if not items:
        return []

    def _extract_detail_terms(text: str) -> set[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z'’.-]{2,}", str(text))
        return {
            token.lower()
            for token in tokens
            if token.lower() not in {
                "the", "and", "for", "with", "that", "this", "from", "into", "about",
                "after", "before", "over", "under", "amid", "latest", "news", "article",
                "report", "reports", "said", "says",
            }
        }

    def _snippet_sentence(item: dict) -> str | None:
        snippet = str(item.get("snippet", "")).strip()
        if not snippet:
            return None
        parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", snippet) if part.strip()]
        if not parts:
            return None
        selected = " ".join(parts[: min(3, len(parts))])
        selected = re.sub(r"^[\-\u2022]\s*", "", selected)
        selected = re.sub(r"\s+", " ", selected).strip()
        if len(selected) < 40:
            return None
        lowered = selected.lower()
        banned_starts = (
            "the article", "this article", "the piece", "this piece",
            "the report", "this report", "read more", "click here",
        )
        if lowered.startswith(banned_starts):
            return None
        if _is_title_heavy_summary(selected, item.get("title", "")):
            return None
        if not selected.endswith((".", "!", "?")):
            selected += "."
        return selected

    item_blocks = []
    for idx, item in enumerate(items, 1):
        block = [f"Item {idx}"]
        if item.get("title"):
            block.append(f"Title: {item['title']}")
        if item.get("snippet"):
            block.append(f"Snippet: {item['snippet']}")
        if item.get("source"):
            block.append(f"Source: {item['source']}")
        item_blocks.append("\n".join(block))

    fallback = []
    need_model = []
    for item in items:
        snippet_summary = _snippet_sentence(item)
        if snippet_summary:
            fallback.append(snippet_summary)
            need_model.append(False)
        else:
            fallback.append(item.get("title", "").strip().rstrip(".") + ".")
            need_model.append(True)

    def _clean_summary(text: str, fallback_text: str, title: str, snippet: str) -> str:
        cleaned = str(text).strip()
        generic_prefixes = (
            "the article discusses",
            "the article explores",
            "the article describes",
            "the article reports",
            "the piece discusses",
            "the piece explores",
            "the report discusses",
            "this article discusses",
            "this article explores",
            "this piece discusses",
        )
        lower = cleaned.lower()
        if any(lower.startswith(prefix) for prefix in generic_prefixes):
            cleaned = fallback_text.strip()
        cleaned = re.sub(r"^(the article|this article|the piece|this piece|the report)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        if not cleaned:
            cleaned = fallback_text.strip()
        if _is_title_heavy_summary(cleaned, title):
            cleaned = fallback_text.strip()
        snippet_terms = _extract_detail_terms(snippet)
        cleaned_terms = _extract_detail_terms(cleaned)
        if snippet_terms and len(cleaned_terms & snippet_terms) < max(2, min(4, len(snippet_terms) // 3 or 1)):
            cleaned = fallback_text.strip()
        if not cleaned.endswith("."):
            cleaned += "."
        return cleaned

    if not any(need_model):
        return fallback

    try:
        resp = llm.invoke([
            SystemMessage(content=(
                "You are summarizing individual news articles from search results. "
                "For each item, write a short factual summary of what happened in 1-3 sentences. "
                "Do not write phrases like 'the article discusses', 'the piece explores', "
                "or 'the report describes'. Start directly with the event or development. "
                "Each sentence must have a clear subject and verb and sound like a news brief. "
                "Do not repeat or closely paraphrase the article title. Prefer the concrete development "
                "described in the snippet. Preserve concrete names, places, institutions, decisions, "
                "and actions from the snippet whenever available. "
                "Return ONLY a JSON array of strings in the same order as the items."
            )),
            HumanMessage(content="\n\n".join(item_blocks)),
        ])
        summaries = json.loads(str(resp.content).strip())
        if isinstance(summaries, list) and len(summaries) == len(items):
            cleaned = []
            for idx, summary in enumerate(summaries):
                if need_model[idx]:
                    cleaned.append(_clean_summary(
                        summary,
                        fallback[idx],
                        items[idx].get("title", ""),
                        items[idx].get("snippet", ""),
                    ))
                else:
                    cleaned.append(fallback[idx])
            return cleaned
    except Exception:
        pass

    return fallback


def _format_news_answer(question: str, parsed: dict, llm: ChatOpenAI) -> str:
    items = parsed.get("items", [])[:5]
    if not items:
        return "The web search tool did not return usable results."
    excerpt_blocks = []
    for idx, item in enumerate(items):
        block_lines = [f"{idx + 1}. {item.get('title', '')}"]
        if item.get("source"):
            block_lines.append(f"Source: {item['source']}")
        if item.get("published"):
            block_lines.append(f"Published: {item['published']}")
        if item.get("snippet"):
            block_lines.append(f"Snippet: {item['snippet']}")
        if item.get("url"):
            block_lines.append(f"URL: {item['url']}")
        excerpt_blocks.append("\n".join(block_lines))
    news_excerpt = "\n\n".join(excerpt_blocks)
    resp = llm.invoke([
        SystemMessage(content=(
            "You are summarizing recent news results. Using only the provided results, write a concise "
            "2-4 sentence overview of the main developments and the biggest themes across the articles. "
            "Focus on what happened, not on the article titles. Do not mention training data or say "
            "'various sources'."
        )),
        HumanMessage(content=f"Question: {question}\n\nNews Results:\n{news_excerpt}"),
    ])
    item_summaries = _summarize_news_items(items, llm)
    lines = [resp.content.strip(), "", "Recent coverage:"]
    for item, detail in zip(items, item_summaries):
        source = item.get("source") or "Source"
        url = item.get("url", "")
        published = item.get("published", "")
        meta = f"[{source}]({url})" if url else source
        if published:
            meta = f"{meta} ({published})"
        lines.append(f"- {detail} {meta}")
    return "\n".join(lines)


def _format_fact_answer(question: str, parsed: dict, llm: ChatOpenAI) -> str:
    items = parsed.get("items", [])
    if not items:
        return "The web search tool did not return usable results."
    top_items = items[:3]
    excerpt_blocks = []
    for idx, item in enumerate(top_items):
        block_lines = [f"{idx + 1}. {item['title']}"]
        if item.get("source"):
            block_lines.append(f"Source: {item['source']}")
        if item.get("snippet"):
            block_lines.append(f"Snippet: {item['snippet']}")
        if item.get("url"):
            block_lines.append(f"URL: {item['url']}")
        excerpt_blocks.append("\n".join(block_lines))
    tool_excerpt = "\n\n".join(excerpt_blocks)
    resp = llm.invoke([
        SystemMessage(content=(
            "Answer the user's factual web question concisely using only the provided search results. "
            "State the answer directly in 1-2 sentences. Do not say 'various sources'."
        )),
        HumanMessage(content=f"Question: {question}\n\nResults:\n{tool_excerpt}"),
    ])
    return f"{resp.content.strip()}\n\n{_format_sources(top_items, inline=True, prefer_title=True)}"


def _format_background_answer(question: str, parsed: dict, llm: ChatOpenAI) -> str:
    items = parsed.get("items", [])
    if not items:
        return "The web search tool did not return usable results."
    top_items = items[:3]
    excerpt_blocks = []
    for idx, item in enumerate(top_items):
        block_lines = [f"{idx + 1}. {item['title']}"]
        if item.get("source"):
            block_lines.append(f"Source: {item['source']}")
        if item.get("snippet"):
            block_lines.append(f"Snippet: {item['snippet']}")
        if item.get("url"):
            block_lines.append(f"URL: {item['url']}")
        excerpt_blocks.append("\n".join(block_lines))
    tool_excerpt = "\n\n".join(excerpt_blocks)
    resp = llm.invoke([
        SystemMessage(content=(
            "Answer the user's background question using only the provided search results. "
            "Write a concise overview in 3-5 flat bullet points covering the main facts or history. "
            "Do not say 'various sources'."
        )),
        HumanMessage(content=f"Question: {question}\n\nResults:\n{tool_excerpt}"),
    ])
    return f"{resp.content.strip()}\n\n{_format_sources(top_items, inline=True, prefer_title=True)}"


def _format_web_answer(question: str, tool_output: str, llm: ChatOpenAI) -> str:
    parsed = _parse_web_tool_output(tool_output)
    if parsed.get("status") != "OK":
        return "The web search tool did not return usable results."
    if _is_news_web_result(parsed):
        return _format_news_answer(question, parsed, llm)
    if _is_background_web_question(question):
        return _format_background_answer(question, parsed, llm)
    return _format_fact_answer(question, parsed, llm)


# ═══════════════════════════════════════════════════════
# CONFIG 1: Single-Pass LLM (no tools, pure baseline)
# ═══════════════════════════════════════════════════════
def run_single_pass(question: str, llm: ChatOpenAI | None = None) -> dict:
    llm = llm or get_llm()
    resp = llm.invoke([
        SystemMessage(content="You are an expert on U.S. federal elections (2000-2024) and "
                      "Israeli Knesset elections (1996-2022). "
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

    from db import fetch_all
    chunks = []

    # Election summaries
    for row in fetch_all("SELECT * FROM elections ORDER BY knesset"):
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
    from tools.chart import HEBREW_TO_ENGLISH
    for row in fetch_all(
        "SELECT knesset, name, bloc, vote_pct, seats FROM parties WHERE seats>0 ORDER BY knesset, seats DESC"
    ):
        name = row['name']
        eng_name = HEBREW_TO_ENGLISH.get(name, name)
        label = f"{eng_name} ({name})" if eng_name != name else name
        chunks.append(
            f"K{row['knesset']}: {label} ({row['bloc']}) — "
            f"{row['vote_pct']}% of votes, {row['seats']} seats."
        )

    # Socioeconomic summaries
    for row in fetch_all("SELECT * FROM socioeconomic LIMIT 201"):
        chunks.append(
            f"Socioeconomic — {row['name']}: pop {row['population']:.0f}, "
            f"median age {row['median_age']}, "
            f"academic degree {row['pct_academic_degree']:.1f}%, "
            f"income/capita {row['avg_monthly_income_per_capita']:.0f} NIS."
        )
    _rag_chunks = chunks
    return chunks


def _keyword_retrieve(question: str, top_k: int = 15) -> list[str]:
    """Keyword-based retrieval (no embeddings — used for ablation comparison only).

    Counts keyword overlaps between the question and each chunk.
    Kept as a named retrieval method for the ablation study:
    "keyword RAG vs. embedding RAG" comparison in the evaluation section.
    """
    chunks = _build_rag_chunks()
    q_lower = question.lower()

    keywords = set(re.findall(r'[a-zA-Z\u0590-\u05FF]{2,}', q_lower))
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


def _vector_retrieve(question: str, top_k: int = 15,
                     embedding_model: str = "minilm") -> list[str]:
    """Retrieve relevant chunks from ChromaDB using embedding similarity.

    Applies party-name expansion so romanized Hebrew (e.g. "HaAvoda")
    and English names (e.g. "Labor") both match bilingual chunks.

    Args:
        embedding_model: "minilm" (local all-MiniLM-L6-v2) or "openai" (text-embedding-3-small)

    Raises FileNotFoundError if ChromaDB is not built — no silent fallback.
    """
    from langchain_community.vectorstores import Chroma
    chroma_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    if not os.path.exists(chroma_dir):
        raise FileNotFoundError(
            "ChromaDB not built. Run 'python build_vectorstore.py' first, "
            "or download from https://github.com/YardenMorad2003/election-agent/releases/tag/v1.0"
        )

    if embedding_model == "openai":
        from langchain_openai import OpenAIEmbeddings
        embedding_fn = OpenAIEmbeddings(model="text-embedding-3-small")
        collection = "election_data_openai"
    elif embedding_model == "mpnet":
        from embeddings import MPNetEmbeddings
        embedding_fn = MPNetEmbeddings()
        collection = "election_data_mpnet"
    else:
        embedding_fn = LocalEmbeddings()
        collection = "election_data"

    vectorstore = Chroma(
        collection_name=collection,
        embedding_function=embedding_fn,
        persist_directory=chroma_dir,
    )
    expanded = _expand_party_names(question)
    results = vectorstore.similarity_search(expanded, k=top_k)
    return [doc.page_content for doc in results]


# ── Cross-encoder reranker ──
_reranker = None

def _get_reranker():
    """Lazy-load the cross-encoder reranker model."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def _rerank(question: str, chunks: list[str], top_k: int = 10) -> list[str]:
    """Rerank retrieved chunks using a cross-encoder model."""
    if not chunks:
        return chunks
    try:
        reranker = _get_reranker()
        pairs = [[question, chunk] for chunk in chunks]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: -x[0])
        return [chunk for _, chunk in ranked[:top_k]]
    except Exception:
        return chunks[:top_k]


def _vector_retrieve_and_rerank(question: str, retrieve_k: int = 25, final_k: int = 10,
                                embedding_model: str = "minilm") -> list[str]:
    """Retrieve from ChromaDB then rerank with cross-encoder."""
    raw_chunks = _vector_retrieve(question, top_k=retrieve_k, embedding_model=embedding_model)
    return _rerank(question, raw_chunks, top_k=final_k)


# ── Context search tool (for Config 4 dynamic agent) ──
from langchain_core.tools import tool

@tool
def context_search(question: str) -> str:
    """Search the election knowledge base for background context, definitions, summaries,
    and historical trends. Use this for conceptual questions about election data, NCHS
    urban-rural classifications, dataset coverage, or when you need context before writing
    a SQL query. Input is a natural language question."""
    chunks = _vector_retrieve_and_rerank(question, retrieve_k=25, final_k=10)
    if not chunks:
        return "No relevant context found."
    return "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))


# ── Embedding-based question classifier (for Config 3) ──
_routing_embeddings = None

def _get_routing_embeddings():
    """Build and cache reference embeddings for each tool category."""
    global _routing_embeddings
    if _routing_embeddings is not None:
        return _routing_embeddings

    embedding_fn = LocalEmbeddings()

    # Reference sentences that characterize each tool's domain
    coalition_refs = [
        "Can the right bloc form a coalition government?",
        "What are all possible 3-party coalitions reaching 61 seats?",
        "Can Likud form a government without Shas?",
        "Which party combinations can form a majority?",
        "Is a coalition possible without Arab parties?",
    ]
    data_query_refs = [
        "How many seats did Likud win in Knesset 25?",
        "What was Biden's vote share in suburban counties?",
        "Which state had the highest Republican turnout?",
        "Compare urban vs rural voting trends",
        "What was the turnout in the 2020 election?",
        "How many votes did Trump get in Georgia?",
    ]
    context_refs = [
        "What are the NCHS urban-rural classification codes?",
        "What data is available in this system?",
        "Explain the urban-rural divide in U.S. elections",
        "What years of election data do you have?",
        "What is the Israeli Knesset electoral system?",
    ]

    coal_embs = embedding_fn.embed_documents(coalition_refs)
    data_embs = embedding_fn.embed_documents(data_query_refs)
    ctx_embs = embedding_fn.embed_documents(context_refs)

    # Average each group into a centroid
    _routing_embeddings = {
        "coalition": np.mean(coal_embs, axis=0),
        "data_query": np.mean(data_embs, axis=0),
        "context_search": np.mean(ctx_embs, axis=0),
    }
    return _routing_embeddings


def _classify_question_embedding(question: str) -> str:
    """Classify a question to a tool using cosine similarity against reference embeddings."""
    try:
        embedding_fn = LocalEmbeddings()
        q_emb = np.array(embedding_fn.embed_query(question))
        centroids = _get_routing_embeddings()

        best_tool = "data_query"
        best_score = -1
        for tool_name, centroid in centroids.items():
            centroid = np.array(centroid)
            score = np.dot(q_emb, centroid) / (np.linalg.norm(q_emb) * np.linalg.norm(centroid))
            if score > best_score:
                best_score = score
                best_tool = tool_name
        return best_tool
    except Exception:
        # Fall back to keyword classification if embeddings fail
        return _classify_question(question)


def run_rag_only(question: str, llm: ChatOpenAI | None = None,
                 retrieval_method: str = "minilm") -> dict:
    """Config 2: RAG-Only with configurable retrieval method.

    retrieval_method:
        "minilm"  — local all-MiniLM-L6-v2 embeddings + cross-encoder reranking (default)
        "mpnet"   — local all-mpnet-base-v2 embeddings + cross-encoder reranking (higher quality)
        "openai"  — OpenAI text-embedding-3-small embeddings + cross-encoder reranking
        "keyword" — keyword overlap matching, no embeddings (ablation baseline)
    """
    llm = llm or get_llm()

    if retrieval_method == "keyword":
        retrieved = _keyword_retrieve(question, top_k=10)
        trace = [
            "Keyword-based retrieval (ablation baseline — no embeddings)",
            f"Retrieved {len(retrieved)} chunks by keyword overlap",
            "LLM synthesized answer from keyword-matched context",
        ]
        tools_used = ["keyword_retrieval"]
    else:
        retrieved = _vector_retrieve_and_rerank(
            question, retrieve_k=25, final_k=10, embedding_model=retrieval_method
        )
        model_names = {
            "minilm": "all-MiniLM-L6-v2 (local, 384-dim)",
            "mpnet": "all-mpnet-base-v2 (local, 768-dim)",
            "openai": "text-embedding-3-small (OpenAI, 1536-dim)",
        }
        model_name = model_names.get(retrieval_method, retrieval_method)
        trace = [
            f"Retrieved 25 chunks from ChromaDB using {model_name}",
            "Reranked to top 10 with cross-encoder (ms-marco-MiniLM-L-6-v2)",
            "LLM synthesized answer from reranked context",
        ]
        tools_used = ["rag_retrieval", "cross_encoder_reranker"]

    context = "\n".join(retrieved)
    resp = llm.invoke([
        SystemMessage(content="You are an expert on U.S. federal elections and Israeli Knesset elections. "
                      "Answer the question using ONLY the retrieved context below. "
                      "If the context doesn't contain enough info, say so.\n\n"
                      f"RETRIEVED CONTEXT:\n{context}"),
        HumanMessage(content=question),
    ])
    return {
        "answer": resp.content,
        "config": "rag_only",
        "retrieval_method": retrieval_method,
        "tools_used": tools_used,
        "trace": trace,
        "retrieved_chunks": retrieved,
    }


# ═══════════════════════════════════════════════════════
# CONFIG 3: Agent + Fixed Routing (keyword rules)
# ═══════════════════════════════════════════════════════
def _classify_question(question: str) -> str:
    q = question.lower()
    coalition_kw = ["coalition", "government", "majority", "61 seats", "form a",
                    "combine", "party combination", "קואליציה"]
    web_kw = [
        "current", "currently", "latest", "recent", "today", "news", "search",
        "web", "wikipedia", "who is", "prime minister", "background", "history",
        "outside the database", "not in the database",
    ]
    if any(kw in q for kw in coalition_kw):
        return "coalition"
    if any(kw in q for kw in web_kw):
        return "web_search"
    return "data_query"


def _should_direct_web_lookup(question: str) -> bool:
    q = question.lower().strip()
    prefixes = (
        "who is", "who's", "what is", "what's", "which is",
        "when was", "when did", "where is", "name",
    )
    web_fact_terms = (
        "current", "currently", "today", "latest", "appointed",
        "appointment", "minister", "secretary", "president",
        "prime minister", "leader", "governor", "mayor",
    )
    return q.startswith(prefixes) and any(term in q for term in web_fact_terms)


def _should_direct_news_lookup(question: str) -> bool:
    q = question.lower().strip()
    return _has_fresh_news_intent(q) or any(phrase in q for phrase in [
        "latest news", "recent news", "current news", "economic news",
        "political news", "headlines", "latest economic", "latest political",
        "news on the", "news about the", "give me the latest",
    ])


def _is_pronoun_followup(question: str) -> bool:
    q = question.lower()
    return any(token in q for token in [" he ", " she ", " they ", " it ", " his ", " her ", " their "]) or q.startswith(
        ("he ", "she ", "they ", "it ", "his ", "her ", "their ")
    )


def _extract_subject_from_fact_question(question: str) -> str | None:
    q = question.strip().rstrip("?.! ")
    lower = q.lower()
    prefixes = ("who is ", "who's ", "what is ", "what's ", "which is ", "where is ", "name ")
    for prefix in prefixes:
        if lower.startswith(prefix):
            return q[len(prefix):].strip()
    return None


def _normalize_role_phrase(subject: str) -> str:
    role = subject.strip()
    role = re.sub(r"^(the)\s+", "", role, flags=re.IGNORECASE)
    role = re.sub(r"^(current|latest|today'?s)\s+", "", role, flags=re.IGNORECASE)
    role = re.sub(r"\s+", " ", role).strip(" ,")
    return role


def _extract_entity_from_answer(answer: str) -> str | None:
    patterns = [
        r"\bis\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",
        r"\bwas\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match:
            return match.group(1).strip()
    return None


def _resolve_followup_web_question(messages: list[dict], latest_question: str) -> str:
    if not _is_pronoun_followup(f" {latest_question.lower()} "):
        return latest_question

    previous_user_questions = [m["content"] for m in messages if m["role"] == "user"]
    previous_assistant_answers = [m["content"] for m in messages if m["role"] == "assistant"]
    if len(previous_user_questions) < 2 or not previous_assistant_answers:
        return latest_question

    previous_question = previous_user_questions[-2]
    previous_answer = previous_assistant_answers[-1]
    subject = _extract_subject_from_fact_question(previous_question) or ""
    role = _normalize_role_phrase(subject) if subject else ""
    entity = _extract_entity_from_answer(previous_answer)

    resolved = latest_question.strip()

    if entity and role:
        appointment_phrase = f"{entity} appointed as {role}"
        resolved = re.sub(
            r"\bwhen was (he|she|they) appointed\b",
            f"when was {appointment_phrase}",
            resolved,
            flags=re.IGNORECASE,
        )
        resolved = re.sub(
            r"\bwhen did (he|she|they) become\b",
            f"when did {entity} become {role}",
            resolved,
            flags=re.IGNORECASE,
        )
    replacement = entity or role
    if replacement:
        resolved = re.sub(r"\bhe\b", replacement, resolved, flags=re.IGNORECASE)
        resolved = re.sub(r"\bshe\b", replacement, resolved, flags=re.IGNORECASE)
        resolved = re.sub(r"\bthey\b", replacement, resolved, flags=re.IGNORECASE)
        resolved = re.sub(r"\bit\b", replacement, resolved, flags=re.IGNORECASE)

    resolved = re.sub(r"\s+", " ", resolved).strip()
    return resolved


def _run_direct_web_lookup(question: str, llm: ChatOpenAI) -> dict:
    tool_result = web_search.invoke(question)
    answer = _format_web_answer(question, str(tool_result), llm)
    trace = ["Direct web lookup from original user question"]
    for line in str(tool_result).splitlines():
        if line.startswith("Query Used: "):
            trace.append(f"Web search executed query: {line.replace('Query Used: ', '', 1)}")
            break
    trace.append(f"Tool 'web_search' returned {len(str(tool_result))} chars")
    trace.append("Structured web formatter rendered final answer")
    return {
        "answer": answer,
        "tools_used": ["web_search"],
        "trace": trace,
        "chart_paths": [],
    }


def run_fixed_routing(question: str, llm: ChatOpenAI | None = None,
                      routing_method: str = "finetuned") -> dict:
    """Config 3: Fixed routing with configurable classification method.

    routing_method: "finetuned" (DistilBERT), "zeroshot" (BART-MNLI),
                    "embedding" (cosine centroids), or "keyword"
    """
    llm = llm or get_llm()
    routing_fallback = None
    if routing_method == "finetuned":
        from classifiers import classify_question_finetuned
        try:
            route = classify_question_finetuned(question)
        except FileNotFoundError as exc:
            route = _classify_question(question)
            routing_fallback = (
                f"Fine-tuned router unavailable ({exc}); fell back to keyword routing"
            )
    elif routing_method == "zeroshot":
        route = classify_question_zeroshot(question)
    elif routing_method == "embedding":
        route = _classify_question_embedding(question)
    else:
        route = _classify_question(question)
    data_query_tool = make_data_query_tool(llm)

    if route == "coalition":
        tool_result = coalition_calculator.invoke(question)
        tool_name = "coalition_calculator"
    elif route == "web_search":
        tool_result = web_search.invoke(question)
        tool_name = "web_search"
    elif route == "context_search":
        tool_result = context_search.invoke(question)
        tool_name = "context_search"
    else:
        tool_result = data_query_tool.invoke(question)
        tool_name = "data_query"

    # synthesize final answer
    if tool_name == "web_search":
        answer = _format_web_answer(question, str(tool_result), llm)
    else:
        resp = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nTool ({tool_name}) returned:\n{tool_result}\n\n"
                         "Provide a clear, well-formatted answer based on the tool output."),
        ])
        answer = resp.content
    return {
        "answer": answer,
        "config": "fixed_routing",
        "tools_used": [tool_name],
        "trace": [
            *([routing_fallback] if routing_fallback else []),
            f"{routing_method} routing -> {tool_name}",
            f"Tool returned {len(str(tool_result))} chars",
            "Structured web formatter rendered final answer" if tool_name == "web_search" else "LLM synthesized final answer",
        ],
        "tool_output": tool_result,
    }


# ═══════════════════════════════════════════════════════
# CONFIG 4: Agent + Dynamic Routing (LLM picks tools)
# ═══════════════════════════════════════════════════════
def run_dynamic_routing(question: str, llm: ChatOpenAI | None = None) -> dict:
    llm = llm or get_llm()
    if _should_direct_web_lookup(question) or _should_direct_news_lookup(question):
        result = _run_direct_web_lookup(question, llm)
        result["config"] = "dynamic_routing"
        return result

    data_query_tool = make_data_query_tool(llm)
    chart_tool = make_chart_tool(llm)
    tools = [data_query_tool, coalition_calculator, context_search, chart_tool, web_search]

    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 25},
    )

    # extract trace and chart paths
    messages = result["messages"]
    trace = []
    tools_used = []
    chart_paths = []
    final_answer = ""
    last_web_tool_output = None

    def _extract_query_used(content: str) -> str | None:
        for line in content.split("\n"):
            if line.startswith("Query Used: "):
                return line.replace("Query Used: ", "", 1).strip()
        return None

    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "web_search":
                    trace.append("LLM decided to call: web_search(...)")
                else:
                    trace.append(f"LLM decided to call: {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)[:200]})")
                tools_used.append(tc["name"])
        if msg.type == "tool":
            if msg.name == "web_search":
                last_web_tool_output = msg.content
                query_used = _extract_query_used(msg.content)
                if query_used:
                    trace.append(f"Web search executed query: {query_used}")
            trace.append(f"Tool '{msg.name}' returned {len(msg.content)} chars")
            # Extract chart paths from tool output
            if "CHART_PATH:" in msg.content:
                for line in msg.content.split("\n"):
                    if line.startswith("CHART_PATH:"):
                        chart_paths.append(line.replace("CHART_PATH:", "").strip())
        if msg.type == "ai" and not getattr(msg, "tool_calls", None):
            final_answer = msg.content

    unique_tools = list(dict.fromkeys(tools_used))
    if unique_tools == ["web_search"] and last_web_tool_output:
        final_answer = _format_web_answer(question, str(last_web_tool_output), llm)
        trace.append("Structured web formatter rendered final answer")

    return {
        "answer": final_answer,
        "config": "dynamic_routing",
        "tools_used": unique_tools,
        "trace": trace,
        "chart_paths": chart_paths,
    }


# ═══════════════════════════════════════════════════════
# CONFIG 5: Plan-and-Execute (multi-hop decomposition)
# ═══════════════════════════════════════════════════════
PLANNER_PROMPT = """You are a query planner for an electoral analyst system.

Decompose the user's question into an ordered list of sub-queries. Each step executes one tool call; earlier step outputs are injected as context into dependent steps.

Available tools:
- data_query: NL to SQL over the US/Israel election database. Use for factual/numerical lookups.
- context_search: RAG retrieval over 22K election chunks. Use for background, definitions, urban-rural classifications, bloc composition context.
- coalition_calculator: finds 61+ seat Israeli coalitions, feasibility-tagged. Israeli coalition questions only.
- web_search: Google News RSS / DuckDuckGo / Wikipedia for current events or facts outside the DB.

Output a JSON object with a 'plan' array. Each step has:
  step (int, 1-indexed), tool (one of the names above), input (self-contained NL input to the tool),
  depends_on (list of step numbers that provide context), rationale (one short sentence).

Rules:
- Simple single-lookup questions: ONE step.
- Comparison questions ("A vs B", "change from X to Y", "grew/shrank"): emit separate lookup steps; the synthesizer does the comparison.
- Multi-part filters ("biggest swing states 2016 to 2020"): separate step per year, then synthesis.
- Max 5 steps.
- Each step input must be executable standalone. Do NOT write "compare to step 1" as input.

Output ONLY the JSON object, no prose, no markdown fences.

Example:
{"plan": [
  {"step": 1, "tool": "data_query", "input": "Democrat vote share by urban/rural in 2000 US presidential election", "depends_on": [], "rationale": "baseline year"},
  {"step": 2, "tool": "data_query", "input": "Democrat vote share by urban/rural in 2024 US presidential election", "depends_on": [], "rationale": "comparison year"}
]}
"""


SYNTHESIZER_PROMPT = """You are producing the final answer from a multi-step plan's outputs.

Use ONLY the numbers and facts from the step outputs. Do not invent data.
- Perform the comparison / calculation / conclusion the question asks for.
- If a step failed or returned nothing, say so briefly and answer from remaining evidence.
- Cite years and geography.
- Be concise; do not re-list all intermediate data unless the question asked for it.
"""


def run_plan_and_execute(question: str, llm: ChatOpenAI | None = None,
                         fallback_to_react: bool = True) -> dict:
    """Config 5: Planner -> Executor -> Synthesizer.

    A planner LLM emits a JSON plan of ordered sub-queries. Each step calls one tool,
    threading prior outputs into later steps via depends_on. A synthesizer LLM then
    produces the final answer from accumulated step outputs.

    Falls back to ReAct (Config 4) if plan parsing fails or too many steps error out.
    """
    llm = llm or get_llm()
    data_query_tool = make_data_query_tool(llm)
    chart_tool = make_chart_tool(llm)
    tool_registry = {
        "data_query": data_query_tool,
        "context_search": context_search,
        "coalition_calculator": coalition_calculator,
        "web_search": web_search,
        "create_chart": chart_tool,
    }

    # ── Phase 1: plan ──
    plan_resp = llm.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=question),
    ])
    plan_text = plan_resp.content.strip()
    if plan_text.startswith("```"):
        plan_text = "\n".join(plan_text.split("\n")[1:])
        if plan_text.endswith("```"):
            plan_text = plan_text.rsplit("```", 1)[0]
    plan_text = plan_text.strip()

    try:
        plan_obj = json.loads(plan_text)
        plan = plan_obj.get("plan", [])
        if not plan:
            raise ValueError("empty plan")
    except Exception as e:
        if fallback_to_react:
            fallback = run_dynamic_routing(question, llm)
            fallback["config"] = "planned_routing"
            fallback["trace"] = [f"Planner parse failed ({e}); fell back to ReAct"] + fallback.get("trace", [])
            fallback["fallback_used"] = True
            fallback["plan"] = []
            return fallback
        raise

    # ── Phase 2: execute ──
    step_outputs: dict[int, str] = {}
    trace = [f"Planner emitted {len(plan)} step(s)"]
    tools_used: list[str] = []
    step_errors = 0

    for step in plan:
        sid = step.get("step")
        tool_name = step.get("tool")
        step_input = step.get("input", "")
        deps = step.get("depends_on", []) or []

        trace.append(f"Step {sid}: {tool_name}({(step_input or '')[:80]}{'...' if len(step_input)>80 else ''})")

        if deps:
            ctx_chunks = []
            for d in deps:
                if d in step_outputs:
                    ctx_chunks.append(f"[Step {d} returned]\n{str(step_outputs[d])[:500]}")
            if ctx_chunks:
                step_input = f"{step_input}\n\nPrior step context:\n" + "\n\n".join(ctx_chunks)

        tool_obj = tool_registry.get(tool_name)
        if tool_obj is None:
            step_outputs[sid] = f"ERROR: unknown tool '{tool_name}'"
            step_errors += 1
            trace.append(f"Step {sid} ERROR: unknown tool")
            continue

        try:
            output = tool_obj.invoke(step_input)
            step_outputs[sid] = output
            tools_used.append(tool_name)
            trace.append(f"Step {sid} returned {len(str(output))} chars")
        except Exception as e:
            step_outputs[sid] = f"ERROR: {e}"
            step_errors += 1
            trace.append(f"Step {sid} ERROR: {e}")

    # Fallback if majority of steps errored
    if fallback_to_react and len(plan) > 0 and step_errors > len(plan) / 2:
        fallback = run_dynamic_routing(question, llm)
        fallback["config"] = "planned_routing"
        fallback["trace"] = trace + [f"{step_errors} step errors; fell back to ReAct"] + fallback.get("trace", [])
        fallback["fallback_used"] = True
        fallback["plan"] = plan
        return fallback

    # ── Phase 3: synthesize ──
    step_lookup = {s["step"]: s for s in plan}
    step_summary = "\n\n".join(
        f"Step {sid} ({step_lookup.get(sid, {}).get('tool', '?')}):\n{str(out)[:1500]}"
        for sid, out in step_outputs.items()
    )
    synth_resp = llm.invoke([
        SystemMessage(content=SYNTHESIZER_PROMPT),
        HumanMessage(content=(
            f"Original question: {question}\n\n"
            f"Step outputs:\n{step_summary}\n\n"
            f"Final answer:"
        )),
    ])

    return {
        "answer": synth_resp.content,
        "config": "planned_routing",
        "tools_used": list(dict.fromkeys(tools_used)),
        "trace": trace,
        "plan": plan,
        "step_outputs": {str(k): str(v)[:400] for k, v in step_outputs.items()},
        "fallback_used": False,
    }


# ═══════════════════════════════════════════════════════
# Unified runner
# ═══════════════════════════════════════════════════════
CONFIGS = {
    "single_pass": run_single_pass,
    "rag_only": run_rag_only,
    "fixed_routing": run_fixed_routing,
    "dynamic_routing": run_dynamic_routing,
    "planned_routing": run_plan_and_execute,
}

def run_question(question: str, config: str = "dynamic_routing",
                 model: str = "gpt-4o-mini",
                 retrieval_method: str = "minilm") -> dict:
    llm = get_llm(model=model)
    if config == "rag_only":
        return CONFIGS[config](question, llm, retrieval_method=retrieval_method)
    return CONFIGS[config](question, llm)


def run_chat(messages: list, model: str = "gpt-4o-mini") -> dict:
    """Run the dynamic routing agent with full conversation history.

    Args:
        messages: list of dicts with 'role' ('user'/'assistant') and 'content'
        model: OpenAI model name

    Returns:
        dict with 'answer', 'tools_used', 'trace'
    """
    llm = get_llm(model=model)
    latest_user_question = next(
        (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
        "",
    )
    if latest_user_question and (
        _should_direct_web_lookup(latest_user_question)
        or _should_direct_news_lookup(latest_user_question)
    ):
        resolved_question = _resolve_followup_web_question(messages, latest_user_question)
        result = _run_direct_web_lookup(resolved_question, llm)
        if resolved_question != latest_user_question:
            result["trace"] = [
                f"Resolved follow-up question: {resolved_question}",
                *result["trace"],
            ]
        return result

    data_query_tool = make_data_query_tool(llm)
    chart_tool = make_chart_tool(llm)
    tools = [data_query_tool, coalition_calculator, context_search, chart_tool, web_search]

    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

    # Convert chat history to LangChain messages
    lc_messages = []
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    result = agent.invoke(
        {"messages": lc_messages},
        config={"recursion_limit": 25},
    )

    # Extract trace and chart paths
    out_messages = result["messages"]
    trace = []
    tools_used = []
    chart_paths = []
    final_answer = ""
    last_web_tool_output = None

    def _extract_query_used(content: str) -> str | None:
        for line in content.split("\n"):
            if line.startswith("Query Used: "):
                return line.replace("Query Used: ", "", 1).strip()
        return None

    for msg in out_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "web_search":
                    trace.append("Called: web_search(...)")
                else:
                    trace.append(f"Called: {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)[:200]})")
                tools_used.append(tc["name"])
        if msg.type == "tool":
            if msg.name == "web_search":
                last_web_tool_output = msg.content
                query_used = _extract_query_used(msg.content)
                if query_used:
                    trace.append(f"Web search executed query: {query_used}")
            trace.append(f"Tool '{msg.name}' returned {len(msg.content)} chars")
            # Extract chart paths from tool output
            if "CHART_PATH:" in msg.content:
                for line in msg.content.split("\n"):
                    if line.startswith("CHART_PATH:"):
                        chart_paths.append(line.replace("CHART_PATH:", "").strip())
        if msg.type == "ai" and not getattr(msg, "tool_calls", None):
            final_answer = msg.content

    unique_tools = list(dict.fromkeys(tools_used))
    if unique_tools == ["web_search"] and last_web_tool_output:
        final_answer = _format_web_answer(latest_user_question, str(last_web_tool_output), llm)
        trace.append("Structured web formatter rendered final answer")

    return {
        "answer": final_answer,
        "tools_used": unique_tools,
        "trace": trace,
        "chart_paths": chart_paths,
    }


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
