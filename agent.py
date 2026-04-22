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
from tools.web_search import web_search
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
- The database only covers election data through 2022. Do not use the data query tool for current office holders, biographies, general background, or news.
- Be precise with numbers — always look them up, don't guess.
- Cite the year and geography you're referencing.
- For U.S. questions, leverage the urban_rural classification to provide urbanization analysis.
- If a tool has already returned useful information, trust that tool output instead of falling back to training knowledge.
- Never say you cannot browse the web or mention a knowledge cutoff after using a tool.
- Do not call the same tool repeatedly with the exact same query unless the previous call returned an error or no results.
- Show your reasoning step by step.
"""


WEB_SYNTHESIS_PROMPT = """You are writing the final answer from a web-search tool output.

Rules:
- Use ONLY the tool output provided.
- If the tool output starts with STATUS: OK, answer directly from those results.
- If the tool output starts with STATUS: NO_RESULTS or STATUS: ERROR, say the web search tool did not return usable results.
- Do NOT mention training data, knowledge cutoffs, or lack of internet access.
- Keep the answer concise and cite the source names or URLs when available.
"""


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
    synthesis_prompt = WEB_SYNTHESIS_PROMPT if tool_name == "web_search" else SYSTEM_PROMPT
    resp = llm.invoke([
        SystemMessage(content=synthesis_prompt),
        HumanMessage(content=f"Question: {question}\n\nTool ({tool_name}) returned:\n{tool_result}\n\n"
                     "Provide a clear, well-formatted answer based on the tool output."),
    ])
    return {
        "answer": resp.content,
        "config": "fixed_routing",
        "tools_used": [tool_name],
        "trace": [
            *([routing_fallback] if routing_fallback else []),
            f"{routing_method} routing -> {tool_name}",
            f"Tool returned {len(str(tool_result))} chars",
            "LLM synthesized final answer",
        ],
        "tool_output": tool_result,
    }


# ═══════════════════════════════════════════════════════
# CONFIG 4: Agent + Dynamic Routing (LLM picks tools)
# ═══════════════════════════════════════════════════════
def run_dynamic_routing(question: str, llm: ChatOpenAI | None = None) -> dict:
    llm = llm or get_llm()
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
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                trace.append(f"LLM decided to call: {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)[:200]})")
                tools_used.append(tc["name"])
        if msg.type == "tool":
            trace.append(f"Tool '{msg.name}' returned {len(msg.content)} chars")
            # Extract chart paths from tool output
            if "CHART_PATH:" in msg.content:
                for line in msg.content.split("\n"):
                    if line.startswith("CHART_PATH:"):
                        chart_paths.append(line.replace("CHART_PATH:", "").strip())
        if msg.type == "ai" and not getattr(msg, "tool_calls", None):
            final_answer = msg.content

    return {
        "answer": final_answer,
        "config": "dynamic_routing",
        "tools_used": list(dict.fromkeys(tools_used)),
        "trace": trace,
        "chart_paths": chart_paths,
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
    for msg in out_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                trace.append(f"Called: {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)[:200]})")
                tools_used.append(tc["name"])
        if msg.type == "tool":
            trace.append(f"Tool '{msg.name}' returned {len(msg.content)} chars")
            # Extract chart paths from tool output
            if "CHART_PATH:" in msg.content:
                for line in msg.content.split("\n"):
                    if line.startswith("CHART_PATH:"):
                        chart_paths.append(line.replace("CHART_PATH:", "").strip())
        if msg.type == "ai" and not getattr(msg, "tool_calls", None):
            final_answer = msg.content

    return {
        "answer": final_answer,
        "tools_used": tools_used,
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
