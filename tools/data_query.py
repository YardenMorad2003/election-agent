"""
Data Query Tool — translates natural language questions into SQL,
executes against the election SQLite database, returns results.
"""
import sqlite3, os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tools.chart import HEBREW_TO_ENGLISH, _translate_hebrew, _preprocess_israeli_question

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "elections.db")

SCHEMA = """
Tables in the database:

=== ISRAELI KNESSET TABLES ===

elections(knesset INTEGER PK, year INTEGER, total_eligible INTEGER, localities_count INTEGER,
         turnout_pct REAL, right_pct REAL, haredi_pct REAL, center_pct REAL,
         left_pct REAL, arab_pct REAL, opposition_right_pct REAL,
         right_haredi_pct REAL, center_left_arab_pct REAL)

parties(knesset INTEGER, code TEXT, name TEXT, bloc TEXT,
        vote_pct REAL, votes INTEGER, seats INTEGER)
  -- NATIONAL-level party results only. Do NOT use this for city/locality questions.
  -- bloc is one of: right, left, center, haredi, arab, opposition_right
  -- code is a Hebrew letter code; name is the Hebrew party name
  -- Key parties across elections: מחל=Likud, אמת=Labor/Zionist Camp, פה=Yesh Atid/Blue&White,
  --   שס=Shas, ג=UTJ, טב/ט=Religious Zionism/Jewish Home, ל=Yisrael Beiteinu,
  --   מרצ/מרץ=Meretz, ודעם=Joint List, ום=Hadash-Taal, עם=Raam, ד=Balad, כן=Kadima/National Camp

localities(name TEXT, knesset INTEGER, eligible INTEGER, turnout_pct REAL,
           right_pct REAL, haredi_pct REAL, center_pct REAL,
           left_pct REAL, arab_pct REAL,
           right_haredi_pct REAL, center_left_arab_pct REAL)
  -- Per-locality bloc-level breakdowns per election. Use for city-level bloc questions.

party_locality(knesset INTEGER, locality TEXT, party_code TEXT, vote_pct REAL)
  -- Per-locality PARTY-level vote percentages. Use this for "how did X city vote by party" questions.
  -- JOIN with parties ON parties.code = party_locality.party_code AND parties.knesset = party_locality.knesset to get party names.
  -- locality names are in Hebrew — use exact match (= 'name') when the full Hebrew name is known.

socioeconomic(name TEXT PK, population REAL, median_age REAL,
              dependency_ratio REAL, pct_academic_degree REAL,
              avg_years_schooling REAL, pct_with_work_income REAL,
              avg_monthly_income_per_capita REAL, pct_below_min_wage REAL,
              pct_above_2x_avg_wage REAL, vehicles_per_100 REAL)

=== U.S. FEDERAL ELECTION TABLES ===

us_president_county(year, state, state_fips, county_name, county_fips, candidate, party, votes, nchs_code, nchs_label, urban_rural, cbsa_title)
  -- Presidential results by county, 2000-2024. 7 election years, ~3,100 counties each.
  -- party is one of: DEMOCRAT, REPUBLICAN, LIBERTARIAN, OTHER
  -- nchs_code 1-6 (1=Large central metro, 2=Large fringe metro, 3=Medium metro, 4=Small metro, 5=Micropolitan, 6=Noncore rural)
  -- urban_rural is one of: Urban, Suburban, Rural
  -- PRIMARY KEY (year, county_fips, candidate)

us_president_precinct(year, state, state_fips, county_name, county_fips, precinct, district, candidate, party, votes, nchs_code, nchs_label, urban_rural, cbsa_title)
  -- Presidential results by precinct, 2016/2020/2024.

us_house_precinct(year, state, state_fips, county_name, county_fips, precinct, district, candidate, party, votes, nchs_code, nchs_label, urban_rural, cbsa_title)
  -- House results by precinct, 2016/2018/2020.

us_senate_precinct(year, state, state_fips, county_name, county_fips, precinct, district, candidate, party, votes, nchs_code, nchs_label, urban_rural, cbsa_title)
  -- Senate results by precinct, 2016/2018/2020.

=== NOTES ===

Israeli data:
- Knesset numbers: 14(1996), 15(1999), 16(2003), 17(2006), 18(2009), 19(2013), 20(2015), 21(2019-Apr), 22(2019-Sep), 23(2020), 24(2021), 25(2022)
- Locality names and party names are in Hebrew
- Bloc definitions: right=Likud+Religious Zionism, haredi=Shas+UTJ, center=Yesh Atid+National Unity, left=Labor+Meretz, arab=Ra'am+Hadash-Ta'al+Balad+Joint List, opposition_right=Yisrael Beiteinu
- right_haredi_pct = right + haredi blocs combined (typical coalition partners)
- center_left_arab_pct = center + left + arab + opposition_right blocs combined (the opposition bloc)
- opposition_right (Yisrael Beiteinu) is counted in the center_left_arab bloc, NOT in right_haredi
- Common Israeli city Hebrew names (use EXACT match with these): Tel Aviv=תל אביב - יפו, Jerusalem=ירושלים, Haifa=חיפה, Beer Sheva=באר שבע, Netanya=נתניה, Rishon LeZion=ראשון לציון, Petah Tikva=פתח תקווה, Ashdod=אשדוד, Ashkelon=אשקלון, Kiryat Ata=קרית אתא, Kiryat Bialik=קרית ביאליק, Kiryat Gat=קרית גת, Kiryat Shmona=קרית שמונה, Nazareth=נצרת, Ramat Gan=רמת גן, Bnei Brak=בני ברק, Herzliya=הרצליה, Kfar Saba=כפר סבא, Bat Yam=בת ים, Holon=חולון, Eilat=אילת, Rehovot=רחובות
- IMPORTANT: Use exact match (=) for locality names when the full Hebrew name is known from the lookup above. Only use LIKE for partial/fuzzy matching when you don't know the exact spelling.
- IMPORTANT: LIKE '%partial%' can match unrelated cities (e.g. '%אתא%' matches both קרית אתא and בוקעאתא). Prefer exact match to avoid mixing data from different localities.
- vote_pct columns are percentages (0-100)

U.S. data:
- state is 2-letter postal code (e.g. 'AZ', 'PA')
- county_fips is zero-padded 5-digit string
- Candidate names are uppercase (e.g. 'JOSEPH R BIDEN', 'DONALD J TRUMP')
- For U.S. presidential trends over time, use us_president_county (fastest, most complete).
- For U.S. precinct-level analysis, use the appropriate precinct table.
- For urban vs rural analysis, GROUP BY urban_rural or nchs_label.
- For two-party vote share: SUM(votes) WHERE party='DEMOCRAT' / SUM(votes) WHERE party IN ('DEMOCRAT','REPUBLICAN')
"""

SQL_SYSTEM = f"""You are a SQL expert. Given a natural language question about elections (Israeli or U.S.),
write a SQLite query to answer it. Return ONLY the SQL query, no explanation.

{SCHEMA}

Rules:
- Use ONLY the tables and columns listed above.
- Always return readable results (use party name not just code where possible).
- When a question contains "(Hebrew: ...)" after a city name, use that EXACT Hebrew string with = (not LIKE) in the WHERE clause. Example: if the question says "Kiryat Ata (Hebrew: קרית אתא)", write: WHERE locality = 'קרית אתא'
- For "Likud" use: name LIKE '%ליכוד%' OR code = 'מחל'
- Limit results to 50 rows max.
- When asked about seats, query the parties table seats column.
- When computing averages/correlations across localities with socioeconomic data, JOIN localities with socioeconomic.
- For U.S. presidential trends over time, prefer us_president_county (fastest, most complete).
- Alaska (AK) reports results by state house district (DISTRICT 1-40), not by county/borough. When listing county-level results, either exclude AK (WHERE state != 'AK') or note that AK entries are legislative districts, not counties.
- For U.S. precinct-level analysis, use the appropriate precinct table.
- For urban vs rural analysis, GROUP BY urban_rural or nchs_label.
- For two-party vote share: SUM(CASE WHEN party='DEMOCRAT' THEN votes END) * 100.0 / SUM(CASE WHEN party IN ('DEMOCRAT','REPUBLICAN') THEN votes END)
- For "flipped" counties (party changed winner between elections): a county flips when the party with the MOST VOTES changes. Use a CTE or subquery to find the winner (party with MAX votes) per county per year, then compare winners across years. Example pattern:
  WITH w AS (SELECT year, county_fips, county_name, state, party, votes, ROW_NUMBER() OVER (PARTITION BY year, county_fips ORDER BY votes DESC) as rn FROM us_president_county WHERE year IN (YEAR1, YEAR2) AND party IN ('DEMOCRAT','REPUBLICAN')) SELECT w1.county_name, w1.state FROM w w1 JOIN w w2 ON w1.county_fips=w2.county_fips WHERE w1.year=YEAR1 AND w1.rn=1 AND w1.party='REPUBLICAN' AND w2.year=YEAR2 AND w2.rn=1 AND w2.party='DEMOCRAT'
- For Israeli questions, use the Israeli tables (elections, parties, localities, etc.)
- When querying Israeli data by Knesset number, ALWAYS include the year in the SELECT output by JOINing with the elections table (elections.knesset = X) or hardcoding the year. Knesset-to-year mapping: 14=1996, 15=1999, 16=2003, 17=2006, 18=2009, 19=2013, 20=2015, 21=2019, 22=2019, 23=2020, 24=2021, 25=2022.
- IMPORTANT: For questions about a specific Israeli city/locality's party results, use party_locality (not parties). The parties table is NATIONAL only.
- To get party names for party_locality results, JOIN: party_locality pl JOIN parties p ON p.code = pl.party_code AND p.knesset = pl.knesset
- Determine the correct country from context clues (state names, county, Knesset, etc.)
"""


def _get_sql(question: str, llm: ChatOpenAI, context: list | None = None) -> str:
    """Generate SQL from a question. Optionally include prior attempt context for reflexion."""
    messages = [{"role": "system", "content": SQL_SYSTEM}]
    if context:
        messages.extend(context)
    messages.append({"role": "user", "content": question})

    resp = llm.invoke(messages)
    sql = resp.content.strip()
    # strip markdown fences if present
    if sql.startswith("```"):
        sql = "\n".join(sql.split("\n")[1:])
    if sql.endswith("```"):
        sql = sql.rsplit("```", 1)[0]
    return sql.strip()


def _run_query(sql: str) -> tuple[str, bool, str]:
    """Execute SQL and return (result_text, success, error_type).

    error_type is one of: None, "empty", "sql_error"
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql).fetchall()
        if not rows:
            return "Query returned no results.", False, "empty"
        # format as markdown table, translating Hebrew party names to English
        cols = rows[0].keys()
        header = " | ".join(cols)
        sep = " | ".join(["---"] * len(cols))
        lines = [header, sep]
        for row in rows[:50]:
            cells = []
            for c in cols:
                val = row[c]
                if isinstance(val, str):
                    val = _translate_hebrew(val)
                cells.append(str(val))
            lines.append(" | ".join(cells))
        return "\n".join(lines), True, None
    except Exception as e:
        return f"SQL Error: {e}\nQuery was: {sql}", False, "sql_error"
    finally:
        conn.close()


REFLEXION_PROMPT = """Your previous SQL query failed. Reflect on what went wrong and write a corrected query.

Original question: {question}
Previous SQL: {sql}
Error: {error}

Common issues to consider:
- Wrong table name (Israeli vs U.S. tables)
- Candidate names must be uppercase for U.S. data (e.g. 'DONALD J TRUMP' not 'Trump')
- county_fips is a TEXT field (zero-padded), not INTEGER
- Party names are uppercase for U.S. (DEMOCRAT, REPUBLICAN) but Hebrew for Israeli data
- For "no results": check the year/state values exist, or try LIKE for partial match if exact match returned nothing
- state is 2-letter code (e.g. 'GA' not 'Georgia')
- Israeli locality names are in Hebrew — use exact match (=) when the full name is known, LIKE only as a fallback
- LIKE '%partial%' can match wrong cities (e.g. '%אתא%' matches both קרית אתא and בוקעאתא) — prefer exact match

Write ONLY the corrected SQL query, no explanation."""

MAX_RETRIES = 2


def make_data_query_tool(llm: ChatOpenAI):
    @tool
    def data_query(question: str) -> str:
        """Query the election database (Israeli Knesset + U.S. federal elections). Use this for any
        factual or numerical question about election results, party performance, voting patterns,
        turnout, urban/rural trends, county or precinct data. Input is a natural language question."""
        processed_question = _preprocess_israeli_question(question)
        sql = _get_sql(processed_question, llm)
        result, success, error_type = _run_query(sql)

        if success:
            return f"SQL: {sql}\n\nResult:\n{result}"

        # Reflexion: reflect on failure and retry
        trace = [f"Attempt 1: {sql} -> {error_type}"]

        for attempt in range(MAX_RETRIES):
            reflection = REFLEXION_PROMPT.format(
                question=question,
                sql=sql,
                error=result,
            )
            # Pass the reflection as context for the next attempt
            sql = _get_sql(reflection, llm)
            result, success, error_type = _run_query(sql)
            trace.append(f"Attempt {attempt + 2}: {sql} -> {'success' if success else error_type}")

            if success:
                trace_str = "\n".join(trace)
                return (f"[Reflexion: succeeded after {attempt + 2} attempts]\n"
                        f"SQL: {sql}\n\nResult:\n{result}\n\nTrace:\n{trace_str}")

        # All retries exhausted
        trace_str = "\n".join(trace)
        return (f"[Reflexion: failed after {MAX_RETRIES + 1} attempts]\n"
                f"SQL: {sql}\n\nResult:\n{result}\n\nTrace:\n{trace_str}")
    return data_query
