# Agentic Electoral Analyst

An AI-powered chatbot for exploring **U.S. federal elections (2000-2024)** and **Israeli Knesset elections (1996-2022)**. It chains together vector embeddings, cross-encoder reranking, NER, zero-shot classification, and LLM-driven tool selection to answer questions, generate charts, and calculate coalition scenarios — all through natural language.

Built for DS-UA 301 (LLMs) at NYU.

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

You type a question about elections. A LangGraph ReAct agent figures out how to answer it — maybe it writes a SQL query, maybe it searches the vector store, maybe it builds a chart, or some combination. The whole thing runs in a Streamlit chat interface.

### Architecture

```
User question
     |
     v
[LangGraph ReAct Agent] ---- conversation history maintained across turns
     |
     |  Agent decides which tool(s) to call (can chain multiple)
     |
     +---> [Data Query]        NL -> SQL -> PostgreSQL -> exact results
     |         |                   (with Reflexion: retries on error/empty)
     |
     +---> [Context Search]    Question -> embeddings -> ChromaDB (25 chunks)
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

### What makes it interesting

- **Conversational memory** -- follow-ups work naturally ("What about 2024?" after asking about 2020)
- **Follow-up suggestions** -- 3 contextual follow-ups generated after every response, shown as clickable buttons
- **Inline charts** -- ask for a visualization and it writes SQL, builds a matplotlib chart, and renders it in the chat
- **Reflexion** -- when a SQL query fails or returns nothing, the agent reflects on what went wrong and retries with a corrected query (up to 2 times)
- **Hebrew city name resolution** -- Israeli localities are stored in Hebrew; a BERT-NER model + lookup table translates English city names so queries like "How did Kiryat Ata vote?" just work
- **Cross-lingual party name expansion** -- queries mentioning romanized Hebrew ("HaAvoda"), English ("Labor"), or Hebrew party names all resolve correctly through query expansion before embedding search
- **4-config comparison** -- run the same question through all 4 routing strategies side by side

---

## ML Pipeline

We chain **7 ML models** through the pipeline — 6 local HuggingFace models plus GPT-4o-mini as the core LLM. The research question driving the project: *How does tool routing strategy affect answer quality for complex electoral analysis questions?*

| # | Model | Type | Where it's used |
|---|-------|------|-----------------|
| 1 | `all-MiniLM-L6-v2` | Text embeddings (384-dim, local) | ChromaDB retrieval, embedding-based routing |
| 2 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker (local) | Reranks 25 retrieved chunks down to top 10 |
| 3 | `facebook/bart-large-mnli` | Zero-shot NLI classifier (local) | Routes questions to tools in Config 3 |
| 4 | `dslim/bert-base-NER` | Named Entity Recognition (local) | Extracts city names for Hebrew resolution |
| 5 | `distilbert-base-uncased` (fine-tuned) | Sequence classifier (local) | Alternative question router for Config 3 |
| 6 | `all-mpnet-base-v2` | Text embeddings (768-dim, local) | Higher-quality embedding alternative |
| 7 | `gpt-4o-mini` / `gpt-4o` | LLM (OpenAI API) | SQL generation, reasoning, synthesis, evaluation |

### How they connect in practice (Config 4)

```
User question
     |
     v
+-----------------------------------------------------------------+
| STEP 0: NER Preprocessing                                       |
| BERT-NER extracts city names, maps to Hebrew via lookup table.  |
| "Kiryat Ata" -> "Kiryat Ata (Hebrew: קרית אתא)"               |
+-----------------------------------------------------------------+
     |
     v
+-----------------------------------------------------------------+
| STEP 1: ReAct Agent (Yao et al., 2023)                         |
| Thought -> Action -> Observation -> Thought -> ...              |
| Autonomously picks tool(s), can chain multiple in one turn.     |
+-----------------------------------------------------------------+
     |               |              |               |
     v               v              v               v
+-----------+ +-------------+ +----------+ +--------------+
| data_query| |context_search| |create_chart| |coalition_calc|
+-----------+ +-------------+ +----------+ +--------------+
     |               |               |               |
     v               v               v               v
  LLM writes    Embeddings      LLM writes      Brute-force
  SQL           encode query    SQL + chart     seat combos
     |               |          config           (>=61)
     v               v               v
  PostgreSQL    ChromaDB        PostgreSQL
  execute       retrieve 25     execute
     |               |               |
     |               v               v
     |          Cross-encoder    matplotlib
     |          rerank -> 10    render chart
     |
     v
  On error/empty: Reflexion
  LLM reflects on failure, generates corrected SQL (up to 2 retries)
     |
     v
+-----------------------------------------------------------------+
| STEP 2: Synthesis                                                |
| Agent combines tool outputs into a natural language answer.      |
+-----------------------------------------------------------------+
     |
     v
+-----------------------------------------------------------------+
| STEP 3: Follow-Up Generation                                    |
| Separate LLM call produces 3 contextual follow-up suggestions.  |
+-----------------------------------------------------------------+
```

### Step-by-step breakdown

1. **NER preprocessing** -- Before anything else, BERT-NER (`dslim/bert-base-NER`) pulls location entities out of the question. If it finds "Kiryat Ata", it looks up the Hebrew equivalent (`קרית אתא`) and injects it. This prevents the LLM from guessing (usually wrong) Hebrew strings in SQL.

2. **ReAct reasoning** -- The LangGraph agent follows the Thought-Action-Observation loop. It sees the conversation history and available tools, reasons about what to do, calls a tool, observes the result, and either calls another tool or produces a final answer. In Config 4 (the default), the LLM has full autonomy over tool selection.

3. **SQL generation** -- The data_query and create_chart tools use system prompts containing the full database schema, query patterns, Hebrew name lookups, and domain rules. The LLM writes SQL from natural language that gets executed against PostgreSQL.

4. **Reflexion** -- When SQL fails or comes back empty, the error gets fed back to the LLM with hints ("candidate names must be UPPERCASE", "use exact Hebrew locality names"). It figures out what went wrong and tries again, up to 2 times.

5. **Dense retrieval + reranking** -- The context_search tool encodes the question with MiniLM (locally, no API calls), pulls the 25 nearest chunks from ChromaDB, then a cross-encoder reranks them to the top 10. The cross-encoder is more accurate because it processes each (query, chunk) pair jointly through BERT's attention layers, rather than comparing independent embeddings.

6. **Zero-shot routing** -- In Config 3, questions get classified to tools using `bart-large-mnli`. The model checks entailment between the question and descriptive labels for each tool ("looking up election results, vote counts" -> data_query). An embedding-based alternative (cosine similarity to pre-computed centroids) is also available.

7. **LLM-as-judge** -- The benchmark evaluates answers with both soft matching and a separate LLM scoring call on a 0-5 rubric, giving partial credit for answers that are directionally right but imprecise.

---

## Tools

### Data Query (`data_query`)
Natural language to SQL, executed against PostgreSQL (SQLite fallback if `DATABASE_URL` isn't set). Covers both U.S. and Israeli datasets. The LLM picks the right table from context clues.

- **Security**: Read-only connections. `DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER` are blocked. `LIMIT 50` auto-appended.
- **NER**: BERT-NER extracts city names and injects Hebrew equivalents before SQL generation.
- **Reflexion**: Failed queries get fed back with diagnostic hints; the LLM retries up to 2 times.

### Context Search (`context_search`)
Two-stage RAG over **22,799 embedded chunks** in ChromaDB. Each chunk is a 1-3 sentence fact about the election data, pre-generated from the database at build time.

#### What's in the chunks

| Type | Count | Example |
|------|-------|---------|
| County summaries | 22,083 | *"In 2020, Maricopa, AZ (Large central metro): JOSEPH R BIDEN (DEMOCRAT) received 1,040,774 votes (50.3%)..."* |
| State summaries | 357 | *"In 2020, GA: DEMOCRAT won with 2,473,633 votes (49.5%). Urban: 72% Dem."* |
| Israeli data | 349 | *"K25: Likud (הליכוד) (right) - 23.4% of votes, 32 seats."* |
| Documentation | 7 | Bloc definitions, table descriptions, coalition context |
| NCHS context | 3 | Urban-rural classification codes and coverage |

#### Retrieval pipeline

Question comes in -> party name expansion (romanized/English/Hebrew aliases appended) -> MiniLM encodes to 384-dim vector -> 25 nearest chunks from ChromaDB -> cross-encoder reranks to top 10 -> passed to LLM as context.

Three embedding models are available for comparison: MiniLM (default), MPNet (768-dim), and OpenAI `text-embedding-3-small` (1536-dim).

### Create Chart (`create_chart`)
The LLM writes both the SQL and a chart config (type, title, axes, grouping). Supports bar, grouped bar, stacked bar, line, pie, and scatter. Hebrew party names auto-translate to English (54 mappings). Charts render inline in the Streamlit chat. Reflexion retries on failure.

### Coalition Calculator (`coalition_calculator`)
Brute-force search over Israeli party seat combinations to find coalitions reaching 61+ seats. Supports must-include parties, max coalition size, and bloc filters.

### Web Search (`web_search`)
DuckDuckGo + Wikipedia for current events and background facts outside the database's coverage.

---

## Routing Configurations

Four strategies, compared head-to-head in the benchmark:

### Config 1: Single-Pass LLM (Baseline)
No tools, no retrieval. The LLM answers from training knowledge only. Tests what the model knows without data access.

### Config 2: RAG-Only
ChromaDB retrieval with MiniLM embeddings + cross-encoder reranking. No SQL. Tests whether retrieval alone can provide accurate answers.

### Config 3: Fixed Routing
An embedding classifier (or zero-shot NLI) decides which single tool to run based on the question. The tool executes, then the LLM synthesizes the answer. Tests ML-based routing vs. LLM autonomy.

### Config 4: Dynamic Routing (ReAct) -- Default
Full ReAct agent with all 5 tools. The LLM decides what to call, can chain tools, and reasons step by step. Tests whether full autonomy beats fixed routing.

**Compare mode**: Toggle in the sidebar to run all 4 configs on the same question in a 2x2 grid.

---

## Data Coverage

### U.S. Federal Elections

| Table | Scope | Years | Rows |
|-------|-------|-------|------|
| `us_president_county` | County-level presidential | 2000-2024 (7 elections) | ~75K |
| `us_president_precinct` | Precinct-level presidential | 2016, 2020, 2024 | ~3.4M |
| `us_house_precinct` | Precinct-level House | 2016, 2018, 2020 | ~2.2M |
| `us_senate_precinct` | Precinct-level Senate | 2016, 2018, 2020 | ~1.9M |

Every U.S. record includes **NCHS urban-rural classification**:
- **Urban** -- Large central metro (code 1) + Medium metro (code 3)
- **Suburban** -- Large fringe metro (code 2) + Small metro (code 4)
- **Rural** -- Micropolitan (code 5) + Noncore (code 6)

### Israeli Knesset Elections

| Table | Scope | Coverage | Rows |
|-------|-------|----------|------|
| `elections` | National stats per election | K14-K25 (1996-2022) | 12 |
| `parties` | National party results | K14-K25 | 151 |
| `localities` | Per-locality bloc breakdowns | 1,384 localities x 12 elections | ~14K |
| `party_locality` | Per-locality party votes | 1,384 localities x 12 elections | ~179K |
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

Israeli parties fall into 6 blocs, with two aggregate groupings used throughout the database:

| Bloc | Parties (K25 example) | Description |
|------|----------------------|-------------|
| **right** | Likud, Religious Zionism, Jewish Home | Nationalist right |
| **haredi** | Shas, United Torah Judaism | Ultra-Orthodox religious |
| **center** | Yesh Atid, National Unity | Centrist / liberal |
| **left** | Labor, Meretz | Social-democratic |
| **arab** | Ra'am, Hadash-Ta'al, Balad | Arab-majority parties |
| **opposition_right** | Yisrael Beiteinu | Right-leaning secular, historically in opposition |

### Aggregate Groupings

| Field | Formula | Meaning |
|-------|---------|---------|
| `right_haredi_pct` | right + haredi | Natural coalition partners (e.g., K25: Likud + RZ + Shas + UTJ = 64 seats) |
| `center_left_arab_pct` | center + left + arab + **opposition_right** | The opposition bloc. Yisrael Beiteinu is counted here, NOT in right_haredi. |

The vector store includes detailed bloc compositions for all 12 elections with both Hebrew and English party names.

---

## Project Structure

```
election-agent/
├── agent.py              # Core: 4 routing configs, reranker, query expansion,
│                         #   embedding router, context_search tool
├── app.py                # Streamlit chatbot UI
├── db.py                 # Database connection (PostgreSQL primary, SQLite fallback)
├── embeddings.py         # Local embedding wrappers (MiniLM, MPNet)
├── classifiers.py        # Zero-shot (BART-MNLI) + fine-tuned (DistilBERT)
├── ner_preprocessor.py   # BERT-NER for Hebrew city name resolution
├── build_db.py           # Israeli JSON -> SQLite ingestion
├── build_us_db.py        # U.S. CSV -> SQLite ingestion
├── build_vectorstore.py  # ChromaDB builder (22K+ chunks, multi-embedding)
├── migrate_to_postgres.py # SQLite -> PostgreSQL migration
├── elections.db          # SQLite database (~1.2GB, fallback)
├── chroma_db/            # ChromaDB vector store (3 collections)
├── charts/               # Generated chart PNGs
├── data/                 # U.S. election CSV source files (~900MB)
├── requirements.txt
├── .env                  # API keys (not in git)
├── tools/
│   ├── data_query.py     # NL -> SQL with Reflexion
│   ├── coalition.py      # Coalition calculator
│   ├── chart.py          # Chart generation + Hebrew name mapping
│   └── web_search.py     # DuckDuckGo + Wikipedia search
└── benchmark/
    ├── questions.json    # 70 evaluation questions
    └── run_benchmark.py  # Benchmark runner with LLM-as-judge
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment variables

Create `.env` in the project root:

```
OPENAI_API_KEY=your-key-here
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/election_agent
```

### 3. Get the data

Download pre-built files from the [v1.0 release](https://github.com/YardenMorad2003/election-agent/releases/tag/v1.0):

- **`elections.db`** (1.2 GB) -- SQLite with all election data
- **`chroma_db.tar.gz`** (47 MB) -- Pre-embedded vector store (22,799 chunks)
- **`distilbert-router.tar.gz`** (235 MB) -- Fine-tuned question router

```bash
# Place elections.db in project root, then:
tar -xzf chroma_db.tar.gz
mkdir -p models && tar -xzf distilbert-router.tar.gz -C models/
```

Or build everything from source:
```bash
python build_db.py          # Israeli tables
python build_us_db.py       # U.S. tables
python build_vectorstore.py # ChromaDB (~5 min)
python train_classifier.py  # Fine-tune DistilBERT (~2 min)
```

### 4. PostgreSQL setup

The app uses PostgreSQL as its primary database with enforced read-only connections, so LLM-generated SQL can't modify data. If `DATABASE_URL` isn't set, it falls back to SQLite.

```bash
# Create the database
psql -U postgres -c "CREATE DATABASE election_agent;"

# Build SQLite source (needed for migration)
python build_db.py
python build_us_db.py          # Full: ~1.2GB
# python build_us_db.py --county-only  # Quick: ~8MB

# Migrate to PostgreSQL (~5-10 min for 7.5M rows)
python migrate_to_postgres.py
```

All connections are read-only. Destructive SQL is blocked. `LIMIT 50` is auto-appended.

### 5. Build vector store (if not downloaded)

```bash
python build_vectorstore.py
# Optional: additional embedding models for comparison
python build_vectorstore.py --with-mpnet
python build_vectorstore.py --with-openai
python build_vectorstore.py --with-all
```

Embeds 22K+ chunks using local sentence-transformers (~5 min on CPU). The cross-encoder, BART-MNLI, and BERT-NER models download automatically on first use.

---

## Usage

```bash
python -m streamlit run app.py
```

### Chat Mode (Default)
Type questions in natural language. The agent picks tools automatically, remembers conversation history, and suggests follow-ups. Ask for charts, coalition scenarios, or data lookups. Expand "Execution trace" to see the reasoning chain.

### Comparison Mode
Toggle "Compare all 4 configs" in the sidebar. Same question, 4 strategies, side-by-side in a 2x2 grid.

### Model Selection
Choose between `gpt-4o-mini` (default), `gpt-4o`, or `gpt-4-turbo` in the sidebar.

---

## Benchmark

70 questions evaluated across all 4 configs using soft matching and LLM-as-judge scoring.

### Running

```bash
python -m benchmark.run_benchmark                          # All configs
python -m benchmark.run_benchmark --config dynamic_routing # Single config
python -m benchmark.run_benchmark --model gpt-4o           # Different model
python -m benchmark.run_benchmark --no-judge               # Skip LLM judge
python -m benchmark.run_benchmark --config rag_only --retrieval-method keyword  # Ablation
```

### Questions (70 total)

| Category | Count | Example |
|----------|-------|---------|
| Factual | 23 | "Who won Georgia in 2020?" |
| Numerical | 20 | "What was Biden's two-party share in urban counties?" |
| Multi-step | 21 | "Did the urban-rural gap grow between 2000 and 2024?" |
| Coalition | 6 | "Can the right bloc form a government without Shas in K25?" |

### Results (gpt-4o-mini)

#### Overall

| Config | Soft Match | Judge Score | Errors |
|--------|-----------|-------------|--------|
| Single-pass (no tools) | 22.9% (16/70) | 3.6/5 | 0 |
| RAG-only (embeddings + reranker) | 34.3% (24/70) | 2.5/5 | 0 |
| Fixed routing (DistilBERT) | 30.0% (21/70) | 3.3/5 | 0 |
| Dynamic routing (ReAct) | 32.9% (23/70) | 3.3/5 | 5 |

#### By Category

| Category | Single-Pass | RAG-Only | Fixed | Dynamic |
|----------|------------|----------|-------|---------|
| Factual (23) | 35% / 3.7 | 52% / 2.9 | 48% / 3.6 | 57% / 3.9 |
| Numerical (20) | 40% / 3.5 | 55% / 2.5 | 40% / 2.9 | 45% / 3.0 |
| Multi-step (21) | 0% / 3.6 | 5% / 2.0 | 10% / 3.4 | 5% / 3.0 |
| Coalition (6) | 0% / 3.2 | 0% / 3.2 | 0% / 3.3 | 0% / 3.5 |

*Format: soft match % / judge score*

#### By Dataset

| Dataset | Single-Pass | RAG-Only | Fixed | Dynamic |
|---------|------------|----------|-------|---------|
| U.S. (39) | 13% / 3.5 | 28% / 1.9 | 21% / 3.1 | 23% / 2.9 |
| Israel (30) | 37% / 3.6 | 43% / 3.3 | 40% / 3.6 | 43% / 3.8 |

#### Embedding Ablation

| Retrieval Method | Soft Match | Judge Score | US | Israel |
|-----------------|-----------|-------------|-----|--------|
| Keyword (no embeddings) | 20.0% (14/70) | 1.4/5 | 8% | 37% |
| MiniLM + cross-encoder | 34.3% (24/70) | 2.5/5 | 28% | 43% |

Embeddings nearly doubled the judge score and added 14 percentage points to soft match. The biggest gains were on U.S. questions (8% -> 28%), where 22K county chunks require semantic understanding that keyword overlap can't provide.

#### Takeaways

1. **Dynamic routing is best for factual questions** -- 57% match and 3.9/5 judge, the highest in that category. The agent can chain tools and self-correct.

2. **RAG has the best overall recall but worst coherence** -- 34.3% soft match but only 2.5/5 judge. It finds the right data but struggles to synthesize it without SQL.

3. **Single-pass LLM gives the most polished answers** -- highest judge score (3.6/5) despite lowest accuracy (22.9%). Fluent but often wrong.

4. **Multi-step and coalition questions are hard for everyone** -- 0-10% across all configs. These need better multi-query chaining.

5. **Israel is easier than U.S.** -- ~40% vs ~20% across configs. Smaller, more structured dataset.

6. **Embeddings matter a lot** -- keyword ablation drops from 34% to 20% soft match. Dense retrieval with reranking is doing real work.

---

## Example Questions

**U.S. Elections**
- "How did Biden perform in suburban counties in 2020 vs 2024?"
- "Which state had the highest Republican vote share in 2024?"
- "Which counties flipped from Republican to Democrat between 2016 and 2020?"

**Israeli Elections**
- "How many seats did Likud win in Knesset 25?"
- "How did Tel Aviv vote by party in K25?"
- "What is the correlation between academic degree % and left-bloc voting?"

**Coalitions**
- "List all possible 3-party coalitions reaching 61 seats in K25"
- "Can the right bloc form a government without Shas?"

**Context / Background**
- "What are the NCHS urban-rural classification codes?"
- "Which parties are in the right bloc in K22?"

**Charts**
- "Show me Republican vs Democrat votes in NY from 2000 to 2024"
- "Bar chart of party vote percentages in Haifa for K25"

**Web Search**
- "Who is the current Prime Minister of Israel?"

---

## Course Module Mapping

| Module | Topic | Implementation |
|--------|-------|----------------|
| 3 | Embeddings & Transformers | MiniLM/MPNet embeddings for 22K+ chunks, BERT-NER for city name extraction, zero-shot classification for routing, embedding-based question classification |
| 4 | Prompt Engineering | System prompts with full schemas, query patterns, and domain rules drive SQL generation quality |
| 5 | RAG | Two-stage pipeline: dense retrieval from ChromaDB + cross-encoder reranking. Cross-encoder processes (query, chunk) pairs jointly for better accuracy than cosine similarity alone |
| 9 | Tool-Augmented LLMs | Five tools (data_query, context_search, create_chart, coalition_calc, web_search) via LangChain function calling, following the TALM paradigm |
| 10 | Agentic AI | LangGraph ReAct agent with Thought-Action-Observation loop. Reflexion for self-correction on SQL failures. 4-config comparison tests routing strategies |
| 13 | Benchmarking | 70-question suite with soft matching + LLM-as-judge (0-5 rubric). Results broken down by category and dataset |
