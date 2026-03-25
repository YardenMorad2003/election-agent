"""
Data Query Tool — translates natural language questions into SQL,
executes against the election SQLite database, returns results.
"""
import sqlite3, os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "elections.db")

SCHEMA = """
Tables in the database:

elections(knesset INTEGER PK, year INTEGER, total_eligible INTEGER, localities_count INTEGER,
         turnout_pct REAL, right_pct REAL, haredi_pct REAL, center_pct REAL,
         left_pct REAL, arab_pct REAL, opposition_right_pct REAL,
         right_haredi_pct REAL, center_left_arab_pct REAL)

parties(knesset INTEGER, code TEXT, name TEXT, bloc TEXT,
        vote_pct REAL, votes INTEGER, seats INTEGER)
  -- bloc is one of: right, left, center, haredi, arab, opposition_right
  -- code is a Hebrew letter code; name is the Hebrew party name
  -- Key parties across elections: מחל=Likud, אמת=Labor/Zionist Camp, פה=Yesh Atid/Blue&White,
  --   שס=Shas, ג=UTJ, טב/ט=Religious Zionism/Jewish Home, ל=Yisrael Beiteinu,
  --   מרצ/מרץ=Meretz, ודעם=Joint List, ום=Hadash-Taal, עם=Raam, ד=Balad, כן=Kadima/National Camp

localities(name TEXT, knesset INTEGER, eligible INTEGER, turnout_pct REAL,
           right_pct REAL, haredi_pct REAL, center_pct REAL,
           left_pct REAL, arab_pct REAL,
           right_haredi_pct REAL, center_left_arab_pct REAL)

party_locality(knesset INTEGER, locality TEXT, party_code TEXT, vote_pct REAL)

socioeconomic(name TEXT PK, population REAL, median_age REAL,
              dependency_ratio REAL, pct_academic_degree REAL,
              avg_years_schooling REAL, pct_with_work_income REAL,
              avg_monthly_income_per_capita REAL, pct_below_min_wage REAL,
              pct_above_2x_avg_wage REAL, vehicles_per_100 REAL)

Notes:
- Knesset numbers: 14(1996), 15(1999), 16(2003), 17(2006), 18(2009), 19(2013), 20(2015), 21(2019-Apr), 22(2019-Sep), 23(2020), 24(2021), 25(2022)
- Locality names and party names are in Hebrew
- The socioeconomic table can be JOINed with localities ON socioeconomic.name = localities.name
- vote_pct columns are percentages (0-100)
"""

SQL_SYSTEM = f"""You are a SQL expert. Given a natural language question about Israeli Knesset elections,
write a SQLite query to answer it. Return ONLY the SQL query, no explanation.

{SCHEMA}

Rules:
- Use ONLY the tables and columns listed above.
- Always return readable results (use party name not just code where possible).
- For "Likud" use: name LIKE '%ליכוד%' OR code = 'מחל'
- Limit results to 50 rows max.
- When asked about seats, query the parties table seats column.
- When computing averages/correlations across localities with socioeconomic data, JOIN localities with socioeconomic.
"""


def _get_sql(question: str, llm: ChatOpenAI) -> str:
    resp = llm.invoke([
        {"role": "system", "content": SQL_SYSTEM},
        {"role": "user", "content": question},
    ])
    sql = resp.content.strip()
    # strip markdown fences if present
    if sql.startswith("```"):
        sql = "\n".join(sql.split("\n")[1:])
    if sql.endswith("```"):
        sql = sql.rsplit("```", 1)[0]
    return sql.strip()


def _run_query(sql: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql).fetchall()
        if not rows:
            return "Query returned no results."
        # format as markdown table
        cols = rows[0].keys()
        header = " | ".join(cols)
        sep = " | ".join(["---"] * len(cols))
        lines = [header, sep]
        for row in rows[:50]:
            lines.append(" | ".join(str(row[c]) for c in cols))
        return "\n".join(lines)
    except Exception as e:
        return f"SQL Error: {e}\nQuery was: {sql}"
    finally:
        conn.close()


def make_data_query_tool(llm: ChatOpenAI):
    @tool
    def data_query(question: str) -> str:
        """Query the Israeli Knesset election database. Use this for any factual or numerical
        question about election results, party performance, voting patterns, turnout,
        socioeconomic correlations, or locality-level data. Input is a natural language question."""
        sql = _get_sql(question, llm)
        result = _run_query(sql)
        return f"SQL: {sql}\n\nResult:\n{result}"
    return data_query
