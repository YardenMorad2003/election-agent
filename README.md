# Agentic Electoral Analyst

A conversational AI system for analyzing **U.S. federal elections (2000-2024)** and **Israeli Knesset elections (1996-2022)**. The system uses multiple ML models in the pipeline — vector embeddings, cross-encoder reranking, embedding-based classification, and LLM-driven tool selection — to answer questions, generate charts, and compare routing strategies.

Project for DS-UA 301 (Large Language Models) at NYU.

---

## Table of Contents

- [How It Works](#how-it-works)
- [ML Pipeline](#ml-pipeline)
- [Tools](#tools)
- [Routing Configurations](#routing-configurations)
- [Data Coverage](#data-coverage)
- [Database Schema](#database-schema)
- [Israeli Political Blocs](#israeli-political-blocs)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Benchmark](#benchmark)
- [Example Questions](#example-questions)

---

## How It Works

The app is a **conversational chatbot** built with Streamlit. You ask questions about elections in natural language, and a LangGraph ReAct agent autonomously decides how to answer — choosing between SQL queries, vector search, chart generation, or a coalition calculator depending on the question.

### Architecture

```
User question
     |
     v
[LangGraph ReAct Agent] ---- conversation history maintained across turns
     |
     |  Agent decides which tool(s) to call (can chain multiple)
     |
     +---> [Data Query]        NL -> SQL -> SQLite -> exact results
     |         |                   (with Reflexion: retries on error/empty)
     |
     +---> [Context Search]    Question -> OpenAI embeddings -> ChromaDB (25 chunks)
     |         |                   -> cross-encoder rerank (top 10) -> context
     |
     +---> [Create Chart]      NL -> SQL + chart config -> matplotlib -> PNG
     |         |                   (with Reflexion: retries on error/empty)
     |
     +---> [Coalition Calc]    Brute-force search over party combos (61+ seats)
     |
     v
[LangGraph ReAct Agent] ---- synthesizes final answer from tool results
     |
     v
Chat response
  - Answer text
  - Inline chart (if generated)
  - Tool badges showing which tools were used
  - Expandable execution trace
  - 3 contextual follow-up suggestions (LLM-generated)
```

### Key Features

- **Conversational memory** -- the agent sees the full chat history, so follow-up questions work naturally ("What about 2024?" after asking about 2020)
- **Smart follow-up suggestions** -- after every response, 3 contextual follow-ups are generated based on the conversation so far, shown as clickable buttons
- **Inline chart generation** -- ask for any visualization and the agent writes SQL, builds a matplotlib chart, and renders it directly in the chat
- **Reflexion pattern** -- when SQL queries fail or return empty results, the agent reflects on what went wrong (wrong table, wrong name spelling, etc.) and automatically retries with a corrected query (up to 2 retries)
- **Hebrew city name resolution** -- Israeli locality names are in Hebrew in the database; the system includes a lookup table of 30+ common cities (English -> Hebrew) so queries like "How did Kiryat Ata vote?" work correctly
- **4-config comparison mode** -- toggle in the sidebar to run the same question through all 4 routing strategies and compare their outputs side by side

---

## ML Pipeline

The system chains **7 distinct ML models** in its pipeline — 6 of which are local/open-source HuggingFace models, with only the core LLM calling the OpenAI API. Our central research question (from the [project proposal](group13_proposal.pdf)) is: *How does tool routing strategy in an LLM-based agent affect answer quality for complex, multi-step electoral analysis questions?*

| # | Model | Provider | Type | Course Module | Where Used |
|---|-------|----------|------|---------------|------------|
| 1 | `all-MiniLM-L6-v2` | HuggingFace (local) | Text embeddings (384-dim) | Module 3 | ChromaDB retrieval (22K+ chunks), embedding-based question routing (Config 3) |
| 2 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace (local) | Cross-encoder reranker | Module 5 | Reranks retrieved chunks from 25→10 in context_search tool |
| 3 | `facebook/bart-large-mnli` | HuggingFace (local) | Zero-shot NLI classifier | Module 3 + 9 | Question routing in Config 3 — classifies questions to tools via natural language inference |
| 4 | `dslim/bert-base-NER` | HuggingFace (local) | Named Entity Recognition | Module 3 | Extracts city names from questions for Hebrew name resolution before SQL generation |
| 5 | `distilbert-base-uncased` (fine-tuned) | HuggingFace (local) | Sequence classifier | Module 7 | Question routing in Config 3 — trained on labeled benchmark questions *(planned)* |
| 6 | `gpt-4o-mini` / `gpt-4o` | OpenAI (API) | Large Language Model | Module 4, 9, 10 | SQL generation, ReAct reasoning, answer synthesis, chart config, Reflexion, LLM-as-judge |

### How the ML models connect (Config 4 / Chat mode)

```
User question
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 0: NER Preprocessing (Module 3)                            │
│ BERT-NER (dslim/bert-base-NER) extracts city names from the     │
│ question, maps them to exact Hebrew DB names via lookup table.   │
│ "Kiryat Ata" → "Kiryat Ata (Hebrew: קרית אתא)"                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: ReAct Agent (Module 10 — Agentic AI)                   │
│ The LLM follows the ReAct pattern (Yao et al., 2023):          │
│ Thought → Action → Observation → Thought → ...                 │
│ It autonomously decides which tool(s) to call and can chain     │
│ multiple tools in a single turn.                                │
└─────────┬───────────────┬──────────────┬───────────────┬────────┘
          │               │              │               │
          ▼               ▼              ▼               ▼
┌─────────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────┐
│ data_query  │ │context_search│ │create_chart│ │coalition_calc│
│ (Module 9)  │ │ (Module 5)   │ │ (Module 9) │ │ (Module 9)   │
└──────┬──────┘ └──────┬───────┘ └─────┬──────┘ └──────┬───────┘
       │               │               │               │
       ▼               ▼               ▼               ▼
  ┌─────────┐   ┌────────────┐   ┌──────────┐   ┌───────────┐
  │ LLM     │   │ Embeddings │   │ LLM      │   │ Brute-    │
  │ writes  │   │ encode     │   │ writes   │   │ force     │
  │ SQL     │   │ question   │   │ SQL +    │   │ seat      │
  │(Mod. 4) │   │ (Module 3) │   │ chart    │   │ combos    │
  └────┬────┘   └─────┬──────┘   │ config   │   │ (≥61)     │
       │              │          └────┬─────┘   └───────────┘
       ▼              ▼               ▼
  ┌─────────┐   ┌────────────┐   ┌──────────┐
  │ SQLite  │   │ ChromaDB   │   │ SQLite   │
  │ execute │   │ retrieve   │   │ execute  │
  └────┬────┘   │ 25 chunks  │   └────┬─────┘
       │        └─────┬──────┘        │
       │              ▼               ▼
       │        ┌────────────┐   ┌──────────┐
       │        │Cross-encoder│  │matplotlib│
       │        │ rerank→top │   │ render   │
       │        │ 10 (Mod. 5)│   │ chart    │
       │        └────────────┘   └──────────┘
       │
  ┌────▼─────────────────────────────┐
  │ On error or empty results:       │
  │ REFLEXION (Module 10)            │
  │ LLM reflects on what went wrong  │
  │ (wrong table? wrong name?) and   │
  │ generates a corrected SQL query. │
  │ Up to 2 retries.                 │
  └──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Answer Synthesis (Module 4 — Prompt Engineering)        │
│ The ReAct agent receives tool outputs as "Observations" and     │
│ synthesizes a final natural language answer with citations.      │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Follow-Up Generation                                    │
│ A separate LLM call generates 3 contextual follow-up            │
│ suggestions based on the conversation so far.                   │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed AI Process by Step

1. **Question preprocessing with NER (Module 3)** — Before any LLM call, a BERT-based Named Entity Recognition model (`dslim/bert-base-NER`) extracts location entities from the question. Detected city names (e.g., "Kiryat Ata") are matched against a Hebrew lookup table and the exact Hebrew equivalent (e.g., `קרית אתא`) is injected into the question. This removes a common failure mode where the LLM generates incorrect Hebrew strings in SQL. The NER model handles cities the hardcoded dictionary might miss, with dictionary fallback for robustness.

2. **ReAct reasoning (Module 10)** — The LangGraph ReAct agent implements the Thought-Action-Observation loop from [Yao et al., 2023]. The LLM receives the conversation history + available tool descriptions, reasons about which tool is needed, emits a structured tool call, observes the result, and decides whether to call another tool or produce a final answer. This is the "dynamic routing" strategy — the LLM has full autonomy over tool selection, unlike Config 3 (fixed routing) where an embedding classifier pre-determines the tool.

3. **SQL generation via prompt engineering (Module 4)** — The data_query and create_chart tools use carefully engineered system prompts containing the full database schema, query patterns (e.g., how to compute flipped counties using window functions), Hebrew name lookups, and domain-specific rules. The LLM generates a SQL query from natural language — this is a form of *tool-assisted LLM* (Module 9) where the LLM's output is executed as code against an external system.

4. **Reflexion self-correction (Module 10)** — When a SQL query fails or returns empty results, the system implements the Reflexion pattern: the failed query and error message are fed back to the LLM with diagnostic hints (e.g., "candidate names must be UPPERCASE", "use exact match for Hebrew locality names"). The LLM reflects on what went wrong and generates a corrected query. This retry loop runs up to 2 times, enabling the system to recover from errors autonomously rather than surfacing failures to the user.

5. **Dense retrieval + reranking (Module 5)** — The context_search tool implements a two-stage RAG pipeline. First, the question is encoded with `all-MiniLM-L6-v2` (a local sentence-transformer, no API calls) and the 25 nearest chunks are retrieved from ChromaDB by cosine similarity (dense retrieval). Then, a cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks those 25 chunks to the top 10. The cross-encoder is more accurate than bi-encoder cosine similarity because it processes the (query, chunk) pair jointly through BERT's attention layers, capturing token-level interactions that independent embeddings miss.

6. **Zero-shot question routing (Module 3 + 9)** — In Config 3 (Fixed Routing), questions are classified to tools using `facebook/bart-large-mnli`, a zero-shot NLI classifier. The model checks entailment between the user's question and descriptive candidate labels for each tool category (e.g., "looking up election results, vote counts, percentages" → data_query). This replaces simple keyword matching with ML-based classification that generalizes to unseen question phrasings. An alternative embedding-based routing method (cosine similarity to pre-computed centroids using `all-MiniLM-L6-v2`) is also available.

7. **LLM-as-judge evaluation (Module 13)** — The benchmark suite evaluates answer quality using two methods: deterministic soft matching (substring + numeric tolerance) and LLM-as-judge scoring on a 0-5 rubric. A separate LLM call scores each answer on correctness, completeness, and relevance, providing partial credit for answers that are directionally correct but imprecise — something exact-match metrics cannot capture.

---

## Tools

### Data Query (`data_query`)
Translates natural language questions into SQL and executes them against the SQLite election database. Supports both U.S. and Israeli datasets. The LLM picks the right table based on context clues (state names, Knesset numbers, etc.).

**Reflexion**: If the SQL query errors or returns empty results, the tool sends the failed query + error back to the LLM with common fix hints (uppercase candidate names, Hebrew LIKE matching, correct table selection). Retries up to 2 times.

### Context Search (`context_search`)
Queries the ChromaDB vector store containing **22,799 embedded text chunks**. A "chunk" is a short, self-contained piece of text (1-3 sentences) that describes one fact about the election data. The system pre-generates these from the database at build time (`build_vectorstore.py`), then embeds each one as a 384-dimensional vector using `all-MiniLM-L6-v2` so they can be searched semantically — like index cards in a library catalog.

#### What's in the 22,799 chunks

| Chunk Type | Count | What it contains | Example |
|-----------|-------|-----------------|---------|
| County summaries | 22,083 | One per county per election year — candidate names, parties, vote counts, percentages, urban/rural classification | *"In 2020, Maricopa, AZ (Large central metro): JOSEPH R BIDEN (DEMOCRAT) received 1,040,774 votes (50.3%); DONALD J TRUMP (REPUBLICAN) received 995,665 votes (48.1%)."* |
| State summaries | 357 | One per state per year — winner, party totals, urban/suburban/rural Democratic vote share | *"In 2020, GA: DEMOCRAT won with 2,473,633 votes (49.5%). Urban: 72% Dem. Rural: 28% Dem."* |
| Israeli data | 349 | National election stats (12 elections), party results (seats, blocs), and socioeconomic profiles (201 municipalities) | *"K25: הליכוד (right) — 23.4% of votes, 32 seats."* |
| Documentation | 7 | Bloc definitions, dataset descriptions, coalition formation context, what tables/columns are available | *"Israeli aggregate bloc groupings: 'Right+Haredi bloc' combines the right and haredi blocs..."* |
| NCHS context | 3 | Urban-rural classification code definitions and data coverage | *"The NCHS Urban-Rural Classification divides U.S. counties into 6 categories..."* |

#### How retrieval works

When you ask a question, the system converts it to a vector using the same embedding model, finds the 25 most similar chunk vectors in ChromaDB (dense retrieval), then a cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores those 25 and keeps the top 10. Those 10 chunks become the context the LLM uses to answer.

### Create Chart (`create_chart`)
Generates matplotlib visualizations from election data. The LLM writes both the SQL query and a chart configuration (type, title, axes, grouping). Supports: bar, grouped bar, stacked bar, line, pie, and scatter charts. Includes party-specific colors (blue for Democrat, red for Republican). Charts are saved as PNG files and rendered inline in the Streamlit chat.

**Reflexion**: If the SQL fails or returns no data (common with Hebrew name transliteration), retries up to 3 times with error context.

### Coalition Calculator (`coalition_calculator`)
Brute-force search over Israeli party combinations to find coalitions reaching 61+ seats (out of 120). Supports:
- Must-include parties (e.g., "coalitions including Likud")
- Maximum coalition size (e.g., "3-party coalitions")
- Bloc filters (e.g., "right-bloc coalitions only")

### Web Search (`web_search`)
Searches public web endpoints for current events, background facts, and information that is outside the local election database. Useful for questions about current office holders, recent news, or general political background that the database (which only covers through 2022) cannot answer.

---

## Routing Configurations

The system implements 4 routing strategies to compare how different levels of tool access and ML affect answer quality:

### Config 1: Single-Pass LLM (Baseline)
- No tools, no retrieval
- LLM answers from training knowledge only
- Tests: what does the LLM know without any data access?

### Config 2: RAG-Only
- ChromaDB vector retrieval (OpenAI embeddings)
- Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
- LLM synthesizes answer from retrieved context only (no SQL)
- Tests: can retrieval + reranking provide accurate answers without structured queries?

### Config 3: Fixed Routing (Embedding-Based)
- Questions are classified to tools using **cosine similarity** between the question embedding and pre-computed reference centroids for each tool category (data_query, coalition_calculator, context_search)
- This replaces simple keyword matching (`if "coalition" in question`) with actual ML-based classification
- The selected tool runs, then the LLM synthesizes the answer
- Tests: does embedding-based routing pick the right tool?

### Config 4: Dynamic Routing (ReAct Agent) -- **Default**
- LangGraph ReAct agent with all 5 tools available
- The LLM autonomously decides which tool(s) to call, can chain multiple tools, and reasons step by step
- Full conversation history is maintained
- Tests: does giving the LLM full autonomy produce better results?

**Compare mode**: Toggle "Compare all 4 configs" in the sidebar to run a question through all 4 strategies and see the outputs side by side in a 2x2 grid.

---

## Data Coverage

### U.S. Federal Elections

| Table | Scope | Years | Rows |
|-------|-------|-------|------|
| `us_president_county` | County-level presidential results | 2000-2024 (7 elections) | ~75K |
| `us_president_precinct` | Precinct-level presidential results | 2016, 2020, 2024 | ~3.4M |
| `us_house_precinct` | Precinct-level House results | 2016, 2018, 2020 | ~2.2M |
| `us_senate_precinct` | Precinct-level Senate results | 2016, 2018, 2020 | ~1.9M |

Every U.S. record includes **NCHS urban-rural classification**:
- **Urban** -- Large central metro (code 1) + Medium metro (code 3)
- **Suburban** -- Large fringe metro (code 2) + Small metro (code 4)
- **Rural** -- Micropolitan (code 5) + Noncore (code 6)

### Israeli Knesset Elections

| Table | Scope | Coverage | Rows |
|-------|-------|----------|------|
| `elections` | National-level stats per election | K14-K25 (1996-2022) | 12 |
| `parties` | National party results (votes, seats, bloc) | K14-K25 | 151 |
| `localities` | Per-locality bloc breakdowns | 1,384 localities x 12 elections | ~14K |
| `party_locality` | Per-locality party vote percentages | 1,384 localities x 12 elections | ~179K |
| `socioeconomic` | Municipal indicators | 201 municipalities | 201 |

---

## Database Schema

### U.S. Tables

```sql
us_president_county(
    year, state, state_fips, county_name, county_fips,
    candidate, party, votes,
    nchs_code, nchs_label, urban_rural, cbsa_title
)
-- party: DEMOCRAT, REPUBLICAN, LIBERTARIAN, OTHER
-- candidate names are UPPERCASE (e.g., 'JOSEPH R BIDEN', 'DONALD J TRUMP')
-- state is 2-letter code (e.g., 'GA', 'PA')
-- county_fips is zero-padded 5-digit TEXT
-- PRIMARY KEY (year, county_fips, candidate)

-- Precinct tables have the same schema plus: precinct, district
```

### Israeli Tables

```sql
elections(knesset PK, year, total_eligible, localities_count, turnout_pct,
         right_pct, haredi_pct, center_pct, left_pct, arab_pct,
         opposition_right_pct, right_haredi_pct, center_left_arab_pct)
-- National-level only

parties(knesset, code, name, bloc, vote_pct, votes, seats)
-- NATIONAL-level party results. Do NOT use for city/locality queries.
-- bloc: right, left, center, haredi, arab, opposition_right

localities(name, knesset, eligible, turnout_pct,
          right_pct, haredi_pct, center_pct, left_pct, arab_pct,
          right_haredi_pct, center_left_arab_pct)
-- Per-locality BLOC-level breakdowns. Names are in Hebrew.

party_locality(knesset, locality, party_code, vote_pct)
-- Per-locality PARTY-level vote percentages.
-- Use this for "how did [city] vote by party?" questions.
-- JOIN with parties ON code = party_code AND knesset = knesset to get party names.

socioeconomic(name PK, population, median_age, dependency_ratio,
             pct_academic_degree, avg_years_schooling, pct_with_work_income,
             avg_monthly_income_per_capita, pct_below_min_wage,
             pct_above_2x_avg_wage, vehicles_per_100)
-- JOIN with localities ON name for socioeconomic + voting correlations.
```

---

## Israeli Political Blocs

The database classifies Israeli parties into 6 blocs, with two aggregate groupings:

### Bloc Definitions

| Bloc | Parties (K25 example) | Description |
|------|----------------------|-------------|
| **right** | Likud, Religious Zionism, Jewish Home | Nationalist right |
| **haredi** | Shas, United Torah Judaism (UTJ) | Ultra-Orthodox religious parties |
| **center** | Yesh Atid, National Unity | Centrist / liberal |
| **left** | Labor, Meretz | Social-democratic / progressive |
| **arab** | Ra'am, Hadash-Ta'al, Balad | Arab-majority parties |
| **opposition_right** | Yisrael Beiteinu | Right-leaning secular, historically in opposition |

### Aggregate Groupings

| Field | Formula | Meaning |
|-------|---------|---------|
| `right_haredi_pct` | right + haredi | Natural coalition partners (e.g., K25 coalition: Likud + RZ + Shas + UTJ = 64 seats) |
| `center_left_arab_pct` | center + left + arab + **opposition_right** | The opposition bloc. Yisrael Beiteinu is counted here, NOT in right_haredi. |

The ChromaDB vector store contains detailed bloc compositions for all 12 Knesset elections (K14-K25), including which specific parties belonged to each bloc in each election, with both Hebrew party codes and English names.

---

## Project Structure

```
election-agent/
├── agent.py              # Core: 4 routing configs, chat mode, reranker,
│                         #   embedding router, context_search tool
├── app.py                # Streamlit chatbot UI with follow-up suggestions
├── embeddings.py         # Local embedding model wrapper (all-MiniLM-L6-v2)
├── classifiers.py        # Zero-shot (BART-MNLI) + fine-tuned (DistilBERT) classifiers
├── ner_preprocessor.py   # BERT-NER entity extraction for Hebrew name resolution
├── build_db.py           # Israeli JSON -> SQLite ingestion (run once)
├── build_us_db.py        # U.S. CSV -> SQLite ingestion (run once)
├── build_vectorstore.py  # ChromaDB vector store builder (22K+ chunks)
├── elections.db          # SQLite database (Israeli + U.S. data, ~1.2GB)
├── chroma_db/            # ChromaDB vector store (persisted, local embeddings)
├── charts/               # Generated chart PNGs (auto-created)
├── data/                 # U.S. election CSV source files (~900MB)
├── requirements.txt      # Python dependencies
├── .env                  # OpenAI API key (not in git)
├── .gitignore            # Excludes elections.db, chroma_db/, data/, charts/
├── tools/
│   ├── data_query.py     # NL -> SQL tool with Reflexion (both datasets)
│   ├── coalition.py      # Coalition calculator tool
│   ├── chart.py          # Chart generation tool with Reflexion + Hebrew name mapping
│   └── web_search.py     # Public web search tool for current events
└── benchmark/
    ├── questions.json    # 70 evaluation questions (4 categories)
    └── run_benchmark.py  # Benchmark runner with LLM-as-judge scoring
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `langgraph`, `langchain`, `langchain-openai`, `langchain-community`, `chromadb`, `openai`, `streamlit`, `pandas`, `python-dotenv`, `sentence-transformers`, `numpy`, `transformers`, `torch`

### 2. Set OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-key-here
```

### 3. Build the database

The Israeli data is pre-built in `elections.db`. To add U.S. data:

```bash
# Load all U.S. tables (county + precinct, ~900MB CSV -> ~1.2GB DB)
python build_us_db.py

# Or load only the county table (fast, ~8MB CSV)
python build_us_db.py --county-only
```

### 4. Build the vector store

```bash
python build_vectorstore.py
```

This embeds 22,000+ text chunks into ChromaDB using a local sentence-transformer model (`all-MiniLM-L6-v2`, ~80MB, no API key needed). Takes ~4-5 minutes on CPU. The cross-encoder reranker model (`ms-marco-MiniLM-L-6-v2`, ~80MB), BART-MNLI zero-shot classifier (~1.6GB), and BERT-NER model (~400MB) all download automatically on first use.

---

## Usage

```bash
python -m streamlit run app.py
```

### Chat Mode (Default)
- Type a question in natural language
- The agent picks the right tool(s) automatically
- Ask follow-ups -- the agent remembers the conversation
- Click the suggested follow-up buttons for quick exploration
- Ask for charts: "Show me a chart of Republican vs Democrat votes in NY"
- Tool badges show which tools/models were used
- Expand "Execution trace" to see the full reasoning chain

### Comparison Mode
- Toggle "Compare all 4 configs" in the sidebar
- The same question runs through all 4 routing strategies
- Results appear in a 2x2 grid for side-by-side comparison

### Model Selection
Choose between `gpt-4o-mini` (fastest/cheapest), `gpt-4o`, or `gpt-4-turbo` in the sidebar.

---

## Benchmark

The benchmark suite evaluates all 4 routing configurations with two evaluation methods:

### Evaluation Methods

1. **Soft match** -- substring matching + numeric tolerance (5%). Fast, deterministic, but can miss correct answers phrased differently.
2. **LLM-as-judge** -- a separate LLM call scores each answer 0-5 on a rubric (perfect/good/acceptable/partial/poor/wrong). Handles free-text answers and gives partial credit.

### Running

```bash
# Run all configs with LLM-as-judge (280+ API calls)
python -m benchmark.run_benchmark

# Single config (faster)
python -m benchmark.run_benchmark --config dynamic_routing

# Different model
python -m benchmark.run_benchmark --model gpt-4o

# Skip LLM-as-judge (faster, cheaper)
python -m benchmark.run_benchmark --no-judge
```

### Question Categories (70 total)

| Category | Count | Evaluation | Example |
|----------|-------|------------|---------|
| Factual | ~20 | Exact match | "Who won Georgia in 2020?" |
| Numerical | ~15 | Tolerance-based | "What was Biden's two-party share in urban counties?" |
| Multi-step | ~25 | Reasoning quality | "Did the urban-rural gap grow between 2000 and 2024?" |
| Coalition | ~10 | Correctness | "Can the right bloc form a government without Shas in K25?" |

Results are saved to `benchmark/results.json` with per-question details, and summary tables are printed for: overall, by category, and by dataset (U.S. / Israel / both).

---

## Example Questions

### U.S. Elections (Data Query)
- "How did Biden perform in suburban counties in 2020 vs 2024?"
- "Which state had the highest Republican vote share in 2024?"
- "Compare urban vs rural presidential voting trends from 2000 to 2024"
- "Which counties flipped from Republican to Democrat between 2016 and 2020?"
- "What was the two-party vote share in Maricopa County across all elections?"

### Israeli Elections (Data Query)
- "How many seats did Likud win in Knesset 25?"
- "How did Tel Aviv vote by party in K25?"
- "What was the turnout in Kiryat Ata in K17?"
- "What is the correlation between academic degree % and left-bloc voting?"
- "Which locality had the highest turnout in K25?"

### Coalition Analysis
- "List all possible 3-party coalitions reaching 61 seats in K25"
- "Can the right bloc form a government without Shas?"
- "What is the smallest coalition including both Likud and Yesh Atid?"

### Conceptual (Context Search)
- "What are the NCHS urban-rural classification codes?"
- "What election data is available in this system?"
- "Which parties are in the right bloc in K22?"
- "What is the center-left-arab bloc?"

### Charts
- "Show me a chart of Republican vs Democrat votes in NY from 2000 to 2024"
- "Bar chart of party vote percentages in Haifa for K25"
- "Line chart of right-haredi bloc percentage across all Knesset elections"

### Web Search
- "Who is the current Prime Minister of Israel?"
- "What were the results of the most recent U.S. midterm elections?"
- "What is the latest polling data for the next Israeli election?"

---

## Course Module Mapping

This project demonstrates techniques from DS-UA 301: Advanced Topics in Data Science (Introduction to LLMs), Spring 2026 at NYU. The course covers deep learning systems with an emphasis on LLM-based generative AI, including transformer architectures, prompt engineering, RAG, tool-augmented LLMs, agentic AI, and benchmarking.

| Module | Topic (from syllabus) | How it's implemented in this project |
|--------|----------------------|--------------------------------------|
| 3 | Attention, Transformers, Embeddings | **Text embeddings** (`text-embedding-3-small`) encode 22K+ document chunks and user questions into dense vectors for semantic similarity search. The same embedding model is used for **embedding-based question classification** in Config 3, where cosine similarity against tool-category centroids replaces keyword matching. |
| 4 | Prompt Engineering, LangChain | **System prompts** for SQL generation contain full database schemas, query patterns, Hebrew name lookups, and domain-specific rules. **LangChain** provides the tool framework (`@tool` decorator, `ChatOpenAI` wrapper). Prompt design directly determines SQL quality — e.g., including the pattern for "flipped counties" using window functions prevents the LLM from generating incorrect queries. |
| 5 | RAG (Retrieval-Augmented Generation) | **Two-stage RAG pipeline**: (1) dense retrieval from ChromaDB (25 nearest chunks by cosine similarity), then (2) **cross-encoder reranking** (`ms-marco-MiniLM-L-6-v2`) to top 10. The cross-encoder processes (query, chunk) pairs jointly through BERT attention layers — more accurate than bi-encoder cosine similarity because it captures token-level interactions. The 22K+ chunk vector store covers county summaries, state aggregates, Israeli election data, bloc definitions, and dataset documentation. |
| 9 | Tool-Assisted LLMs | **Five specialized tools** extend the LLM's capabilities beyond its training knowledge: (1) `data_query` translates natural language to SQL and executes against SQLite, (2) `create_chart` generates SQL + matplotlib chart configs, (3) `coalition_calculator` does brute-force combinatorial search over party seat combinations, (4) `context_search` retrieves from the vector store, (5) `web_search` queries the web for current events and background facts outside the database. Each tool uses **function calling** — the LLM emits structured tool invocations that the system executes and returns results from. This follows the Tool-Augmented Language Model (TALM) paradigm covered in Module 9. |
| 10 | Agentic AI (ReACT, Reflexion) | **LangGraph ReAct agent** implements the Thought-Action-Observation loop from [Yao et al., 2023]: the LLM reasons about which tool to call, observes the result, and decides next steps autonomously. **Reflexion** pattern enables self-correction: when SQL queries fail or return empty results, the error is fed back to the LLM with diagnostic hints, and it generates a corrected query (up to 2 retries). The **4-config comparison** (single-pass, RAG-only, fixed routing, dynamic routing) directly tests how routing strategy affects answer quality — the central research question from our proposal. |
| 13 | LLM Benchmarking | **70-question benchmark suite** with two evaluation methods: (1) deterministic soft matching (substring + 5% numeric tolerance) for factual/numerical questions, and (2) **LLM-as-judge** scoring on a 0-5 rubric (perfect/good/acceptable/partial/poor/wrong) for open-ended questions. Results are broken down by question category (factual, numerical, multi-step, coalition) and by dataset (U.S., Israel, both). This follows the benchmarking methodology from Module 13, using LLM evaluation as an alternative to human annotation. |

