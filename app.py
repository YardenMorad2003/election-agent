"""
Streamlit UI — Agentic Electoral Analyst
Side-by-side comparison of 4 routing configurations.
"""
import streamlit as st
import time
from agent import run_question, run_all_configs, CONFIGS

st.set_page_config(page_title="Agentic Electoral Analyst", layout="wide", page_icon="🇮🇱")

# ── Custom CSS ──
st.markdown("""
<style>
    .config-card {
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        background: #0e1117;
    }
    .config-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .trace-item {
        font-size: 0.8rem;
        color: #888;
        padding: 2px 0;
    }
    .tool-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.title("Agentic Electoral Analyst")
st.caption("Comparing tool routing strategies for Israeli Knesset election analysis (K14–K25, 1996–2022)")

# ── Sidebar ──
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"], index=0)
    mode = st.radio("Run mode", ["Compare all 4 configs", "Single config"])
    if mode == "Single config":
        selected_config = st.selectbox("Configuration", list(CONFIGS.keys()),
                                       format_func=lambda x: {
                                           "single_pass": "1. Single-Pass LLM (no tools)",
                                           "rag_only": "2. RAG-Only",
                                           "fixed_routing": "3. Fixed Routing (keyword rules)",
                                           "dynamic_routing": "4. Dynamic Routing (LLM decides)",
                                       }[x])

    st.divider()
    st.subheader("Example questions")
    examples = [
        "How many seats did Likud win in Knesset 25?",
        "What was the average turnout across all elections?",
        "Which elections had Arab parties with more than 13 combined seats?",
        "List all possible 3-party coalitions reaching 61 seats in K25",
        "How did right-bloc vote share change from K14 to K25?",
        "What is the correlation between academic degree % and left-bloc voting?",
        "Compare Likud seats in K24 vs K25",
        "Which locality had the highest turnout in K25?",
        "Who is the current Prime Minister of Israel?",
        "What are the latest Israeli political developments from RSS feeds?",
        "Give me background on the Joint List party from the web",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["question"] = ex

# ── Main ──
question = st.text_input("Ask a question about Israeli elections:",
                         value=st.session_state.get("question", ""),
                         placeholder="e.g., How many seats did Likud win in K25?")

CONFIG_LABELS = {
    "single_pass": ("1. Single-Pass LLM", "No tools — pure LLM baseline", "🔴"),
    "rag_only": ("2. RAG-Only", "Keyword retrieval → LLM synthesis", "🟡"),
    "fixed_routing": ("3. Fixed Routing", "Keyword rules pick the tool", "🟠"),
    "dynamic_routing": ("4. Dynamic Routing", "LLM decides which tools to call", "🟢"),
}

if question and st.button("Run", type="primary", use_container_width=True):
    if mode == "Compare all 4 configs":
        with st.spinner("Running all 4 configurations..."):
            start = time.time()
            results = run_all_configs(question, model=model)
            total_time = time.time() - start

        st.info(f"Completed all 4 configs in {total_time:.1f}s")

        # Display in 2x2 grid
        col1, col2 = st.columns(2)
        for i, (config_name, result) in enumerate(results.items()):
            label, desc, emoji = CONFIG_LABELS[config_name]
            col = col1 if i % 2 == 0 else col2
            with col:
                with st.expander(f"{emoji} {label}", expanded=True):
                    st.caption(desc)
                    st.markdown(result["answer"])
                    if result["tools_used"]:
                        st.markdown("**Tools used:** " + ", ".join(f"`{t}`" for t in result["tools_used"]))
                    with st.popover("Trace"):
                        for step in result["trace"]:
                            st.text(step)
    else:
        with st.spinner(f"Running {selected_config}..."):
            start = time.time()
            result = run_question(question, config=selected_config, model=model)
            elapsed = time.time() - start

        st.info(f"Completed in {elapsed:.1f}s")
        label, desc, emoji = CONFIG_LABELS[selected_config]
        st.subheader(f"{emoji} {label}")
        st.caption(desc)
        st.markdown(result["answer"])

        if result["tools_used"]:
            st.markdown("**Tools used:** " + ", ".join(f"`{t}`" for t in result["tools_used"]))

        with st.expander("Execution trace"):
            for step in result["trace"]:
                st.text(step)

        if "retrieved_chunks" in result:
            with st.expander("Retrieved chunks (RAG)"):
                for i, chunk in enumerate(result["retrieved_chunks"], 1):
                    st.text(f"{i}. {chunk}")

        if "tool_output" in result:
            with st.expander("Raw tool output"):
                st.code(result["tool_output"])
