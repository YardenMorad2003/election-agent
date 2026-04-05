"""
Streamlit UI — Agentic Electoral Analyst (Chat Interface)
Conversational chatbot backed by LangGraph ReAct agent with SQL + coalition tools.
"""
import streamlit as st
import time, re, os, json
from agent import run_chat, run_question, run_all_configs, get_llm, CONFIGS

st.set_page_config(page_title="Agentic Electoral Analyst", layout="wide", page_icon="🗳️")

# ── Custom CSS ──
st.markdown("""
<style>
    .tool-chip {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 4px;
        background: #1a3a5c;
        color: #7fb3e0;
    }
    .trace-step {
        font-size: 0.78rem;
        color: #888;
        padding: 1px 0;
        font-family: monospace;
    }
    /* Style suggestion buttons */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        font-size: 0.82rem;
        padding: 4px 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state init ──
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

# ── Sidebar ──
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1"], index=0)

    st.divider()

    compare_mode = st.toggle("Compare all 4 configs", value=False,
                             help="Run the next question through all 4 routing configs side by side")

    st.divider()
    st.subheader("Try these")

    us_examples = [
        "How did Biden perform in suburban counties in 2020?",
        "Which state had the highest Republican vote share in 2024?",
        "Compare urban vs rural voting trends from 2000 to 2024",
        "Which counties flipped from R to D between 2016 and 2020?",
    ]
    il_examples = [
        "How many seats did Likud win in Knesset 25?",
        "List all 3-party coalitions reaching 61 seats in K25",
        "How did right-bloc share change from K14 to K25?",
        "Which locality had the highest turnout in K25?",
        "Who is the current Prime Minister of Israel?",
        "Give me background on the Joint List party from the web",
    ]

    st.caption("U.S. Elections")
    for ex in us_examples:
        if st.button(ex, key=f"us_{ex}", use_container_width=True):
            st.session_state.pending_question = ex
            st.session_state.suggestions = []
            st.rerun()

    st.caption("Israeli Elections")
    for ex in il_examples:
        if st.button(ex, key=f"il_{ex}", use_container_width=True):
            st.session_state.pending_question = ex
            st.session_state.suggestions = []
            st.rerun()

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.suggestions = []
        st.rerun()

# ── Header ──
st.title("Agentic Electoral Analyst")
st.caption("U.S. federal elections (2000-2024) · Israeli Knesset elections (1996-2022)")


# ── Helpers ──

def _display_charts(chart_paths: list):
    """Display chart images in Streamlit."""
    for path in chart_paths:
        if os.path.exists(path):
            st.image(path, use_container_width=True)


def _render_comparison(results):
    """Render 4-config comparison in a 2x2 grid."""
    labels = {
        "single_pass": ("1. Single-Pass LLM", "🔴"),
        "rag_only": ("2. RAG-Only", "🟡"),
        "fixed_routing": ("3. Fixed Routing", "🟠"),
        "dynamic_routing": ("4. Dynamic Routing", "🟢"),
    }
    col1, col2 = st.columns(2)
    for i, (config_name, result) in enumerate(results.items()):
        label, emoji = labels[config_name]
        col = col1 if i % 2 == 0 else col2
        with col:
            with st.expander(f"{emoji} {label}", expanded=True):
                st.markdown(result["answer"])
                if result.get("tools_used"):
                    st.markdown("**Tools:** " + ", ".join(f"`{t}`" for t in result["tools_used"]))
                if result.get("trace"):
                    with st.popover("Trace"):
                        for step in result["trace"]:
                            st.text(step)


SUGGEST_PROMPT = """Based on this conversation about elections, suggest 3 short follow-up questions the user might want to ask next.

Rules:
- Each question should be a natural follow-up to what was just discussed
- Keep questions under 60 characters each
- Vary the types: one could dig deeper, one could compare, one could ask for a chart
- If the conversation was about U.S. data, suggest U.S. follow-ups (and vice versa for Israeli)
- Return ONLY a JSON array of 3 strings, nothing else

Example: ["What about 2024?", "Show this as a chart", "Compare with rural counties"]"""


def _generate_suggestions(chat_history: list) -> list[str]:
    """Generate follow-up suggestions based on conversation context."""
    if not chat_history:
        return []
    try:
        llm = get_llm(model="gpt-4o-mini", temperature=0.7)
        # Build a compact summary of recent conversation (last 4 messages)
        recent = chat_history[-4:]
        convo = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:200]}"
            for m in recent
        )
        resp = llm.invoke([
            {"role": "system", "content": SUGGEST_PROMPT},
            {"role": "user", "content": convo},
        ])
        text = resp.content.strip()
        # Strip markdown fences
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        suggestions = json.loads(text.strip())
        if isinstance(suggestions, list) and len(suggestions) > 0:
            return suggestions[:3]
    except Exception:
        pass
    return []


# ── Render chat history ──
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("chart_paths"):
            _display_charts(msg["chart_paths"])
        if msg["role"] == "assistant" and msg.get("tools_used"):
            tools_html = " ".join(f'<span class="tool-chip">{t}</span>' for t in msg["tools_used"])
            elapsed = msg.get("time", 0)
            st.markdown(f"{tools_html} &nbsp; <span style='color:#666;font-size:0.75rem'>{elapsed:.1f}s</span>",
                        unsafe_allow_html=True)
        if msg["role"] == "assistant" and msg.get("trace"):
            with st.expander("Execution trace", expanded=False):
                for step in msg["trace"]:
                    st.markdown(f"<div class='trace-step'>{step}</div>", unsafe_allow_html=True)
        if msg["role"] == "assistant" and msg.get("comparison"):
            _render_comparison(msg["comparison"])

# ── Show follow-up suggestions ──
if st.session_state.suggestions and st.session_state.chat_history:
    cols = st.columns(len(st.session_state.suggestions))
    for i, suggestion in enumerate(st.session_state.suggestions):
        with cols[i]:
            if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                st.session_state.pending_question = suggestion
                st.session_state.suggestions = []
                st.rerun()

# ── Handle input ──
pending = st.session_state.pop("pending_question", None)
user_input = st.chat_input("Ask about U.S. or Israeli elections...")

question = pending or user_input

if question:
    # Clear old suggestions
    st.session_state.suggestions = []

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })

    if compare_mode:
        with st.chat_message("assistant"):
            with st.spinner("Running all 4 configurations..."):
                start = time.time()
                results = run_all_configs(question, model=model)
                elapsed = time.time() - start

            summary = f"Here are the results from all 4 routing configs ({elapsed:.1f}s):"
            st.markdown(summary)
            _render_comparison(results)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": summary,
            "tools_used": [],
            "trace": [],
            "time": elapsed,
            "comparison": results,
        })

    else:
        chat_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.chat_history
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start = time.time()
                result = run_chat(chat_messages, model=model)
                elapsed = time.time() - start

            st.markdown(result["answer"])

            chart_paths = result.get("chart_paths", [])
            if chart_paths:
                _display_charts(chart_paths)

            if result["tools_used"]:
                tools_html = " ".join(f'<span class="tool-chip">{t}</span>' for t in result["tools_used"])
                st.markdown(f"{tools_html} &nbsp; <span style='color:#666;font-size:0.75rem'>{elapsed:.1f}s</span>",
                            unsafe_allow_html=True)

            if result["trace"]:
                with st.expander("Execution trace", expanded=False):
                    for step in result["trace"]:
                        st.markdown(f"<div class='trace-step'>{step}</div>", unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "tools_used": result.get("tools_used", []),
            "trace": result.get("trace", []),
            "time": elapsed,
            "chart_paths": chart_paths,
        })

    # Generate follow-up suggestions — runs after the answer is already displayed
    st.session_state.suggestions = _generate_suggestions(st.session_state.chat_history)
    st.rerun()
