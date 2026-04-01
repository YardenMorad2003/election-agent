# Agentic Electoral Analyst

A tool-augmented LLM system for analyzing Israeli Knesset elections (1996–2022), built to compare four different agent routing strategies side by side.

## Overview

This project explores how different levels of tool access and routing logic affect LLM accuracy on structured data questions. It ships with a Streamlit UI that lets you run the same question through all four configurations and compare the outputs.

**Data coverage:**
- 12 elections (Knesset 14–25, 1996–2022)
- 1,384 localities with per-election bloc breakdowns
- Party-level vote shares and seat counts
- Socioeconomic indicators for 201 municipalities

## Routing Configurations

| # | Config | How it works |
|---|--------|-------------|
| 1 | **Single-Pass LLM** | No tools — answers from training knowledge only (baseline) |
| 2 | **RAG-Only** | Keyword retrieval over pre-built text chunks → LLM synthesis |
| 3 | **Fixed Routing** | Keyword rules classify the question and pick a tool |
| 4 | **Dynamic Routing** | LangGraph ReAct agent — LLM decides which tools to call |

## Tools

- **Data Query** — Translates natural language questions into SQL and executes them against the election database. Handles factual lookups, aggregations, correlations, and comparisons.
- **Coalition Calculator** — Brute-force search over party combinations to find coalitions reaching 61+ seats. Supports filters for must-include parties, max coalition size, and bloc constraints.
- **Israeli Politics RSS** — Pulls fresh RSS headlines about Israeli politics so the agent can ground answers in recent developments.
- **Web Search** — Searches public web endpoints for current events and background facts that are outside the local election database.

## Project Structure

```
election-agent/
├── agent.py          # Core logic for all 4 routing configs
├── app.py            # Streamlit UI
├── build_db.py       # JSON → SQLite ingestion script
├── elections.db      # SQLite database (pre-built)
├── requirements.txt
└── tools/
    ├── data_query.py # NL → SQL tool
    ├── coalition.py  # Coalition calculator tool
    ├── israel_politics_rss.py # Recent Israeli politics RSS headlines
    └── web_search.py # Public web search tool
```

## Database Schema

```
elections       — National-level stats per election (turnout, bloc percentages)
parties         — Party results per election (votes, seats, bloc)
localities      — Per-locality per-election bloc breakdowns
party_locality  — Party-level vote share per locality per election
socioeconomic   — Municipal indicators (income, education, demographics)
```

## Setup

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your-key-here
```

If rebuilding the database from source JSON:

```bash
python build_db.py
```

## Usage

```bash
streamlit run app.py
```

The UI provides:
- Model selection (gpt-4o-mini, gpt-4o, gpt-4-turbo)
- Run a single config or compare all four side by side
- Example questions in the sidebar
- Execution traces and raw tool output for inspection

## Example Questions

- "How many seats did Likud win in Knesset 25?"
- "What was the average turnout across all elections?"
- "List all possible 3-party coalitions reaching 61 seats in K25"
- "What is the correlation between academic degree % and left-bloc voting?"
- "Which locality had the highest turnout in K25?"
- "Who is the current Prime Minister of Israel?"
- "What are the latest Israeli political developments from RSS feeds?"
