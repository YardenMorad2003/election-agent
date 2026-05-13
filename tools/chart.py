"""
Chart Tool — generates matplotlib visualizations from SQL query results.
Returns the file path to the saved chart image.
"""
import sqlite3, os, uuid, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "elections.db")
CHART_DIR = os.path.join(os.path.dirname(__file__), "..", "charts")

# Ensure chart output directory exists
os.makedirs(CHART_DIR, exist_ok=True)

CHART_SYSTEM = """You are a data visualization expert. Given a natural language request about election data,
write TWO things separated by "---":

1. A SQLite query that returns the data needed for the chart. The query should return
   columns suitable for plotting (e.g., x-axis values, y-axis values, optional grouping column).
   Keep results to under 50 rows.

2. A Python dictionary (as a JSON string) with chart configuration:
   - "chart_type": one of "bar", "horizontal_bar", "line", "grouped_bar", "stacked_bar", "pie", "scatter"
   - "title": chart title
   - "xlabel": x-axis label
   - "ylabel": y-axis label
   - "x_col": column name for x-axis (from your SQL query)
   - "y_col": column name for y-axis — use this when the SQL is in LONG format (one row per series-point) with a separate group_col.
   - "y_cols": (alternative to y_col) LIST of column names for multi-series WIDE-format SQL. Use this when the SQL returns several numeric columns side-by-side that should each be plotted as their own line/bar (e.g., right_pct, left_pct, center_pct on the same x=year).
   - "group_col": (optional) column name for grouping/coloring — for LONG-format SQL only. Do NOT set group_col when using y_cols.
   - "legend": true/false

WIDE vs LONG format — pick one, do NOT mix:
- LONG: SQL returns one row per (x, series) combo, with a group column naming the series. Example: `SELECT year, party, SUM(votes) FROM ... GROUP BY year, party`. Config uses x_col, y_col, group_col.
- WIDE: SQL returns one row per x with each series as its own column. Example: `SELECT year, right_pct, left_pct, center_pct FROM elections`. Config uses x_col + y_cols (list).
NEVER set group_col to one of the numeric y-columns — that creates one fake "series" per distinct percentage value (the chart will be a mess of single-point series).

Example LONG output:
```
SELECT year, party, SUM(votes) as total_votes
FROM us_president_county
WHERE state = 'NY'
GROUP BY year, party
HAVING party IN ('DEMOCRAT', 'REPUBLICAN')
ORDER BY year
---
{"chart_type": "grouped_bar", "title": "NY Presidential Votes by Party", "xlabel": "Year", "ylabel": "Total Votes", "x_col": "year", "y_col": "total_votes", "group_col": "party", "legend": true}
```

Example WIDE output (multi-series time chart from `elections` table):
```
SELECT year, right_pct, left_pct, center_pct FROM elections ORDER BY year
---
{"chart_type": "line", "title": "Bloc Trends K14-K25", "xlabel": "Year", "ylabel": "Bloc Share (%)", "x_col": "year", "y_cols": ["right_pct", "left_pct", "center_pct"], "legend": true}
```

Database schema:

U.S. tables:
- us_president_county(year, state, county_name, county_fips, candidate, party, votes, nchs_code, nchs_label, urban_rural, cbsa_title)
- us_president_precinct, us_house_precinct, us_senate_precinct (same schema + precinct, district)

Israeli tables:
- elections(knesset, year, total_eligible, turnout_pct, right_pct, haredi_pct, center_pct, left_pct, arab_pct, right_haredi_pct, center_left_arab_pct)
  -- NATIONAL-level stats per election. Use for bloc trends, turnout trends.
- parties(knesset, code, name, bloc, vote_pct, votes, seats)
  -- NATIONAL-level party results, ONE row per (knesset, party). Use for any
  -- "party X over time", "popularity", "seats by Knesset", "vote share trend"
  -- chart. The vote_pct here is the national aggregate (already a percentage 0-100)
  -- — DO NOT sum or aggregate it across rows.
- localities(name, knesset, eligible, turnout_pct, right_pct, haredi_pct, center_pct, left_pct, arab_pct, right_haredi_pct, center_left_arab_pct)
  -- Per-locality BLOC-level breakdowns (not party-level)
- party_locality(knesset, locality, party_code, vote_pct)
  -- Per-locality PARTY-level vote percentages, one row per (knesset, locality, party).
  -- Use ONLY for single-city or city-comparison charts. NEVER aggregate vote_pct
  -- across localities (SUM/AVG of percentages without population weights is meaningless).
  -- For national party trends, use the `parties` table directly.
  -- JOIN with parties: party_locality pl JOIN parties p ON p.code = pl.party_code AND p.knesset = pl.knesset
- socioeconomic(name, population, median_age, pct_academic_degree, avg_monthly_income_per_capita, ...)

Table-selection cheat sheet:
- "How did X party perform across all Knessets" / "party X popularity over time" / "trend of seats" → `parties` table (national, one row per knesset)
- "Bloc trends over time" / "right vs left over years" → `elections` table (right_pct, left_pct, etc.)
- "How did X party do in city Y" / "compare cities" / "city-level breakdown" → `party_locality` JOIN `parties`
- "Turnout by city" / "bloc breakdown of city Z" → `localities`
- Never SUM(vote_pct) or AVG(vote_pct) across localities for a national figure — the `parties.vote_pct` column already has it.

Rules:
- U.S. state is 2-letter code (e.g. 'NY')
- Candidate names are UPPERCASE
- Party is DEMOCRAT, REPUBLICAN, LIBERTARIAN, or OTHER
- When a question contains "(Hebrew: ...)" after a city name, use that EXACT Hebrew string with = (not LIKE) in the WHERE clause. Example: if the question says "Kiryat Ata (Hebrew: קרית אתא)", write: WHERE locality = 'קרית אתא'
- Israeli locality/party names are in Hebrew — use exact match (=) when the full Hebrew name is known. LIKE '%partial%' can match wrong cities.
- For Israeli city party results, use party_locality (NOT parties table which is national only)
- For Israeli party vote breakdowns (single election, single city), ALWAYS use "horizontal_bar" chart type — it handles many parties much better than vertical bars. Sort by vote_pct DESC and filter to parties with vote_pct >= 1.5 to keep charts clean.
- Hebrew party names will be automatically translated to English in the chart — no need to translate in SQL.
- PARTIES ACROSS MULTIPLE KNESSETS: filter by the stable letter `code`, NOT by `name`. Party names mutate across alliances (e.g. Likud appears as 'ליכוד', 'הליכוד', 'הליכוד ישראל ביתנו'; Labor as 'עבודה', 'העבודה', 'המחנה הציוני', 'העבודה-גשר', 'העבודה-גשר-מרצ'; Yesh Atid as 'יש עתיד', 'כחול לבן', 'מפלגת המרכז'). Stable codes:
    * Likud = 'מחל'
    * Labor / Zionist Camp = 'אמת'
    * Yesh Atid / Blue & White = 'פה'
    * Shas = 'שס'
    * UTJ = 'ג'
    * Religious Zionism / Jewish Home = 'טב' (also 'ט')
    * Yisrael Beiteinu = 'ל'
    * Meretz = 'מרצ' (also 'מרץ')
    * Joint List / Hadash-Taal = 'ודעם' / 'ום'
    * Ra'am = 'עם'
  Always inject a canonical English label via CASE so each party is one consistent legend entry, e.g.:
    SELECT knesset,
           CASE code WHEN 'מחל' THEN 'Likud' WHEN 'אמת' THEN 'Labor' WHEN 'פה' THEN 'Yesh Atid' END AS party,
           seats
    FROM parties WHERE code IN ('מחל','אמת','פה') AND knesset BETWEEN 14 AND 25 ORDER BY knesset, code
  Use "line" chart_type (or "grouped_bar") for multi-Knesset party-seat trends, with x_col="knesset", y_col="seats", group_col="party".
- Knesset-to-year: 14=1996, 15=1999, 16=2003, 17=2006, 18=2009, 19=2013, 20=2015, 21=2019, 22=2019, 23=2020, 24=2021, 25=2022
- Common Israeli city Hebrew names: Tel Aviv=תל אביב, Jerusalem=ירושלים, Haifa=חיפה, Beer Sheva=באר שבע, Netanya=נתניה, Rishon LeZion=ראשון לציון, Petah Tikva=פתח תקווה, Ashdod=אשדוד, Ashkelon=אשקלון, Kiryat Ata=קרית אתא, Kiryat Bialik=קרית ביאליק, Kiryat Yam=קרית ים, Kiryat Motzkin=קרית מוצקין, Kiryat Gat=קרית גת, Kiryat Shmona=קרית שמונה, Nazareth=נצרת, Ramat Gan=רמת גן, Bnei Brak=בני ברק, Herzliya=הרצליה, Kfar Saba=כפר סבא, Modiin=מודיעין, Acre/Akko=עכו, Tiberias=טבריה, Lod=לוד, Ramla=רמלה, Bat Yam=בת ים, Holon=חולון, Eilat=אילת, Rehovot=רחובות, Ra'anana=רעננה
- Return ONLY the SQL and the config dict separated by ---
"""


PARTY_COLORS = {
    "DEMOCRAT": "#2166ac",
    "REPUBLICAN": "#b2182b",
    "LIBERTARIAN": "#f4a582",
    "OTHER": "#999999",
    "right": "#b2182b",
    "left": "#2166ac",
    "center": "#92c5de",
    "haredi": "#333333",
    "arab": "#4dac26",
}

# Hebrew party name -> English name mapping
HEBREW_TO_ENGLISH = {
    # Likud
    "ליכוד": "Likud", "הליכוד": "Likud", "הליכוד ישראל ביתנו": "Likud Yisrael Beiteinu",
    # Labor variants
    "עבודה": "Labor", "העבודה": "Labor", "העבודה-גשר-מרצ": "Labor-Gesher-Meretz",
    "העבודה-גשר": "Labor-Gesher", "המחנה הציוני": "Zionist Camp",
    # Religious / Right
    "הבית היהודי": "Jewish Home", "ימינה": "Yamina", 'מפד"ל': "NRP (Mafdal)",
    "יהדות התורה": "United Torah Judaism", 'ש"ס': "Shas",
    "הציונות הדתית": "Religious Zionism", "האיחוד הלאומי": "National Union",
    "מולדת": "Moledet", "איחוד לאומי-מפדל": "National Union-NRP",
    "איחוד מפלגות הימין": "Union of Right-Wing Parties",
    "עוצמה יהודית": "Jewish Power (Otzma Yehudit)", "עוצמה לישראל": "Otzma L'Yisrael",
    # Center
    "יש עתיד": "Yesh Atid", "כחול לבן": "Blue & White", "מפלגת המרכז": "Center Party",
    "קדימה": "Kadima", "המחנה הממלכתי": "National Unity", "שינוי": "Shinui",
    "כולנו": "Kulanu", "תקווה חדשה": "New Hope", "התנועה": "HaTnuah",
    "הימין החדש": "New Right", "הדרך השלישית": "Third Way",
    # Left
    "מרצ": "Meretz", "המחנה הדמוקרטי": "Democrats (Meretz)",
    # Arab parties
    'בל"ד': "Balad", 'רע"מ': "Ra'am", 'חד"ש': "Hadash",
    'חד"ש-תע"ל': "Hadash-Ta'al", 'חד"ש-בל"ד': "Hadash-Balad",
    "הרשימה המשותפת": "Joint List", 'רע"מ-בל"ד': "Ra'am-Balad",
    'רע"מ-תע"ל': "Ra'am-Ta'al", 'מד"ע-רע"מ': "Mada-Ra'am",
    # Yisrael Beiteinu
    "ישראל ביתנו": "Yisrael Beiteinu", "ישראל ביתנו-האיחוד הלאומי": "Yisrael Beiteinu-National Union",
    # Other
    "ישראל בעלייה": "Yisrael BaAliyah", "עם אחד": "Am Ehad",
    "גשר": "Gesher", "יחד": "Yahad", "זהות": "Zehut",
    'גיל (גמלאים)': "Gil (Pensioners)", "גמלאים": "Pensioners",
    "הירוקים-מימד": "Greens-Meimad", "חרות": "Herut",
    "ברית לאומית מתקדמת": "Progressive National Alliance",
}


# English city name -> exact Hebrew name for SQL queries
CITY_NAME_LOOKUP = {
    "tel aviv": "תל אביב - יפו", "jerusalem": "ירושלים", "haifa": "חיפה",
    "beer sheva": "באר שבע", "be'er sheva": "באר שבע", "beersheba": "באר שבע",
    "netanya": "נתניה", "rishon lezion": "ראשון לציון", "rishon le zion": "ראשון לציון",
    "petah tikva": "פתח תקווה", "ashdod": "אשדוד", "ashkelon": "אשקלון",
    "kiryat ata": "קרית אתא", "kiryat bialik": "קרית ביאליק",
    "kiryat yam": "קרית ים", "kiryat motzkin": "קרית מוצקין",
    "kiryat gat": "קרית גת", "kiryat shmona": "קרית שמונה",
    "nazareth": "נצרת", "ramat gan": "רמת גן", "bnei brak": "בני ברק",
    "herzliya": "הרצליה", "kfar saba": "כפר סבא", "modiin": "מודיעין",
    "acre": "עכו", "akko": "עכו", "tiberias": "טבריה", "lod": "לוד",
    "ramla": "רמלה", "bat yam": "בת ים", "holon": "חולון", "eilat": "אילת",
    "rehovot": "רחובות", "ra'anana": "רעננה", "raanana": "רעננה",
}


def _preprocess_israeli_question(question: str) -> str:
    """Replace English city names with Hebrew equivalents using NER + dictionary fallback.

    Uses a BERT-based NER model (dslim/bert-base-NER) to extract location entities,
    then maps them to exact Hebrew names via a lookup table. Falls back to dictionary
    matching for cities the NER model doesn't detect.
    """
    try:
        from ner_preprocessor import preprocess_question_ner
        return preprocess_question_ner(question)
    except Exception:
        # Fallback to dictionary-only if NER fails to load
        q_lower = question.lower()
        replacements = []
        for eng, heb in CITY_NAME_LOOKUP.items():
            idx = q_lower.find(eng)
            if idx != -1:
                replacements.append((idx, len(eng), eng, heb))
        if not replacements:
            return question
        replacements.sort(key=lambda x: x[0], reverse=True)
        result = question
        for idx, length, eng, heb in replacements:
            result = result[:idx] + f"{result[idx:idx+length]} (Hebrew: {heb})" + result[idx+length:]
        return result


def _translate_hebrew(text: str) -> str:
    """Translate Hebrew party/bloc names to English if a mapping exists."""
    if not isinstance(text, str):
        return text
    # Exact match first
    if text in HEBREW_TO_ENGLISH:
        return HEBREW_TO_ENGLISH[text]
    # Check if any Hebrew key is a substring (for partial matches)
    for heb, eng in HEBREW_TO_ENGLISH.items():
        if heb == text.strip():
            return eng
    return text


def _run_chart_query(sql: str) -> list[dict]:
    """Run SQL and return list of row dicts.
    Uses PostgreSQL if DATABASE_URL is set, otherwise SQLite (read-only).
    """
    from db import execute_query
    rows, cols = execute_query(sql)
    return rows


_BLOC_COLOR_HINTS = {
    "right": "#b2182b", "right_pct": "#b2182b",
    "left": "#2166ac", "left_pct": "#2166ac",
    "center": "#92c5de", "center_pct": "#92c5de",
    "haredi": "#333333", "haredi_pct": "#333333",
    "arab": "#4dac26", "arab_pct": "#4dac26",
    "opposition_right": "#fb6a4a", "opposition_right_pct": "#fb6a4a",
    "right_haredi": "#cb181d", "right_haredi_pct": "#cb181d",
    "center_left_arab": "#3690c0", "center_left_arab_pct": "#3690c0",
}


def _label_for_column(col: str) -> str:
    """Turn a column name like 'right_pct' into a readable legend label 'Right'."""
    if not col:
        return col
    base = col.removesuffix("_pct")
    return base.replace("_", " ").title()


def _build_chart(data: list[dict], config: dict) -> str:
    """Build a matplotlib chart and return the saved file path."""
    chart_type = config.get("chart_type", "bar")
    title = config.get("title", "Chart")
    xlabel = config.get("xlabel", "")
    ylabel = config.get("ylabel", "")
    x_col = config.get("x_col", "")
    y_col = config.get("y_col", "")
    y_cols = config.get("y_cols") or None
    group_col = config.get("group_col", None)
    show_legend = config.get("legend", True)

    # Translate any Hebrew names to English in the data
    for row in data:
        for key in list(row.keys()):
            if isinstance(row[key], str):
                row[key] = _translate_hebrew(row[key])

    # Defensive: pct columns that came back > 100 mean the SQL aggregated
    # percentages (e.g., SUM(vote_pct) across localities). Refuse to plot.
    pct_like_cols = [c for c in [y_col, *(y_cols or [])] if c and "pct" in c.lower()]
    for col in pct_like_cols:
        vals = [row.get(col) for row in data if isinstance(row.get(col), (int, float))]
        if vals and max(vals) > 100:
            plt.close(fig)
            raise ValueError(
                f"Column '{col}' is labelled as a percentage but contains a value > 100 "
                f"(max = {max(vals):.1f}). The SQL likely aggregated per-locality percentages "
                "(SUM/AVG of vote_pct across localities is not meaningful — use the parties "
                "table's national vote_pct directly)."
            )

    # Defensive: if the LLM set group_col to one of the wide-format numeric
    # columns, it would mint a fake "series" per distinct value. Convert that
    # to a y_cols multi-series chart instead.
    if (group_col and not y_cols and data
            and isinstance(data[0].get(group_col), (int, float))
            and isinstance(data[0].get(y_col), (int, float))):
        y_cols = [y_col, group_col]
        group_col = None

    fig, ax = plt.subplots(figsize=(10, 6))

    # WIDE format: multiple y columns, each plotted as its own series.
    if y_cols and chart_type in ("line", "grouped_bar", "stacked_bar"):
        xs_wide = [row[x_col] for row in data]
        if chart_type == "line":
            for col in y_cols:
                ys_series = [row.get(col) for row in data]
                color = _BLOC_COLOR_HINTS.get(col) or PARTY_COLORS.get(col)
                ax.plot(xs_wide, ys_series, marker="o", linewidth=2,
                        label=_label_for_column(col), color=color)
        else:  # grouped_bar / stacked_bar
            import numpy as np
            x_indices = np.arange(len(xs_wide))
            width = 0.8 / max(len(y_cols), 1)
            bottom = np.zeros(len(xs_wide)) if chart_type == "stacked_bar" else None
            for i, col in enumerate(y_cols):
                ys_series = [row.get(col, 0) or 0 for row in data]
                color = _BLOC_COLOR_HINTS.get(col) or PARTY_COLORS.get(col)
                label = _label_for_column(col)
                if chart_type == "grouped_bar":
                    ax.bar(x_indices + i * width, ys_series, width, label=label, color=color)
                else:
                    ax.bar(x_indices, ys_series, 0.6, bottom=bottom, label=label, color=color)
                    bottom = bottom + np.array(ys_series)
            ax.set_xticks(x_indices + width * (len(y_cols) - 1) / 2 if chart_type == "grouped_bar" else x_indices)
            ax.set_xticklabels([str(x) for x in xs_wide], rotation=45, ha="right")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if show_legend:
            ax.legend()
        plt.tight_layout()
        path = os.path.join(CHART_DIR, f"chart_{abs(hash(title)) % 10_000_000}.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return path

    xs = [row[x_col] for row in data]
    ys = [row[y_col] for row in data]

    if chart_type == "horizontal_bar":
        # Horizontal bar — ideal for many categories (e.g., party breakdowns)
        colors = [PARTY_COLORS.get(str(x), "#4292c6") for x in xs]
        y_pos = range(len(xs))
        ax.barh(y_pos, ys, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([str(x) for x in xs])
        ax.invert_yaxis()  # Largest at top
        # Add value labels on bars
        for i, v in enumerate(ys):
            ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
    elif group_col and chart_type in ("grouped_bar", "line", "stacked_bar"):
        # Grouped data
        groups = sorted(set(row[group_col] for row in data))
        x_vals = sorted(set(row[x_col] for row in data))

        if chart_type == "line":
            for group in groups:
                gdata = [row for row in data if row[group_col] == group]
                gx = [row[x_col] for row in gdata]
                gy = [row[y_col] for row in gdata]
                color = PARTY_COLORS.get(group, None)
                ax.plot(gx, gy, marker="o", label=str(group), color=color, linewidth=2)
        elif chart_type in ("grouped_bar", "stacked_bar"):
            import numpy as np
            x_indices = np.arange(len(x_vals))
            width = 0.8 / max(len(groups), 1)
            bottom = np.zeros(len(x_vals)) if chart_type == "stacked_bar" else None

            for i, group in enumerate(groups):
                gdata = {row[x_col]: row[y_col] for row in data if row[group_col] == group}
                gy = [gdata.get(x, 0) for x in x_vals]
                color = PARTY_COLORS.get(group, None)

                if chart_type == "grouped_bar":
                    ax.bar(x_indices + i * width, gy, width, label=str(group), color=color)
                else:
                    ax.bar(x_indices, gy, 0.6, bottom=bottom, label=str(group), color=color)
                    bottom += np.array(gy)

            ax.set_xticks(x_indices + width * (len(groups) - 1) / 2)
            ax.set_xticklabels([str(x) for x in x_vals], rotation=45, ha="right")
    elif chart_type == "pie":
        colors = [PARTY_COLORS.get(str(x), None) for x in xs]
        colors = [c for c in colors if c] or None
        ax.pie(ys, labels=xs, autopct="%1.1f%%", colors=colors)
        ax.set_aspect("equal")
    elif chart_type == "scatter":
        ax.scatter(xs, ys, alpha=0.6)
    elif chart_type == "line":
        ax.plot(xs, ys, marker="o", linewidth=2)
    else:  # bar
        colors = [PARTY_COLORS.get(str(x), "#4292c6") for x in xs]
        ax.bar(range(len(xs)), ys, color=colors)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(x) for x in xs], rotation=45, ha="right")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Format large numbers on y-axis
    if ys and all(isinstance(y, (int, float)) for y in ys):
        max_val = max(abs(y) for y in ys if y is not None)
        if max_val > 1_000_000:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
        elif max_val > 1_000:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

    if show_legend and group_col:
        ax.legend()

    plt.tight_layout()

    # Save
    fname = f"chart_{uuid.uuid4().hex[:8]}.png"
    fpath = os.path.join(CHART_DIR, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return fpath


def make_chart_tool(llm: ChatOpenAI):
    @tool
    def create_chart(question: str) -> str:
        """Generate a chart or graph from election data. Use this when the user asks for a
        visualization, plot, graph, chart, or trend line. Input is a natural language description
        of the desired chart (e.g. 'Show Republican vs Democrat votes in NY from 2000 to 2024').
        Returns the file path to the generated chart image."""
        import json as _json

        # Match the data_query coverage rules so charts can't surface bad data.
        from tools.data_query import (
            _detect_invalid_knessets, _references_us_2024,
            KNESSET_MIN, KNESSET_MAX,
        )
        invalid_k = _detect_invalid_knessets(question)
        if invalid_k:
            bad = ", ".join(f"K{n}" for n in sorted(set(invalid_k)))
            return (
                f"[Data coverage] The Israeli Knesset dataset only covers K{KNESSET_MIN}-K{KNESSET_MAX} "
                f"(1996-2022). The question references {bad}, which is outside coverage."
            )
        if _references_us_2024(question):
            return (
                "[Data coverage] The 2024 U.S. presidential dataset has known quality issues and "
                "isn't being surfaced. Reliable U.S. presidential coverage is 2000-2020."
            )

        # Pre-process: inject Hebrew names for Israeli cities
        processed_question = _preprocess_israeli_question(question)

        # Get SQL + chart config from LLM
        resp = llm.invoke([
            {"role": "system", "content": CHART_SYSTEM},
            {"role": "user", "content": processed_question},
        ])
        text = resp.content.strip()
        # Strip markdown fences
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        # Split SQL and config
        parts = text.split("---")
        if len(parts) < 2:
            return f"Error: Could not parse chart specification. LLM returned:\n{text}"

        sql = parts[0].strip()
        config_str = parts[1].strip()

        # Clean SQL of markdown fences
        if sql.startswith("```"):
            sql = "\n".join(sql.split("\n")[1:])
        if sql.endswith("```"):
            sql = sql.rsplit("```", 1)[0]
        sql = sql.strip()

        # Parse config
        try:
            # Find JSON object in the config string
            json_match = re.search(r'\{[^{}]+\}', config_str, re.DOTALL)
            if json_match:
                config = _json.loads(json_match.group())
            else:
                config = _json.loads(config_str)
        except _json.JSONDecodeError as e:
            return f"Error parsing chart config: {e}\nRaw config: {config_str}"

        # Run query with reflexion retry (max 2 retries)
        data = None
        error_msg = None
        for attempt in range(2):
            try:
                data = _run_chart_query(sql)
            except ValueError as e:
                error_msg = str(e)
                data = None

            if data:
                break

            # Reflexion: ask LLM to fix the query
            fix_prompt = (
                f"Your SQL query failed or returned no data.\n"
                f"SQL: {sql}\n"
                f"Error: {error_msg or 'Query returned 0 rows'}\n\n"
                f"Common issues:\n"
                f"- Israeli locality names are in HEBREW. Use exact match (=) with the known Hebrew name, not LIKE (which can match wrong cities).\n"
                f"- For city party results use party_locality, not parties (which is national only)\n"
                f"- JOIN party_locality pl JOIN parties p ON p.code = pl.party_code AND p.knesset = pl.knesset\n\n"
                f"Write a corrected SQL query and the same chart config. Format: SQL --- config JSON"
            )
            resp = llm.invoke([
                {"role": "system", "content": CHART_SYSTEM},
                {"role": "user", "content": fix_prompt},
            ])
            text = resp.content.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            text = text.strip()

            parts = text.split("---")
            if len(parts) >= 2:
                sql = parts[0].strip()
                if sql.startswith("```"):
                    sql = "\n".join(sql.split("\n")[1:])
                if sql.endswith("```"):
                    sql = sql.rsplit("```", 1)[0]
                sql = sql.strip()
                try:
                    json_match = re.search(r'\{[^{}]+\}', parts[1], re.DOTALL)
                    if json_match:
                        config = _json.loads(json_match.group())
                except Exception:
                    pass
            error_msg = None

        if not data:
            return f"Query returned no data after 2 attempts. Last SQL:\n{sql}"

        # Build chart
        try:
            fpath = _build_chart(data, config)
            return f"CHART_PATH:{fpath}\nSQL: {sql}\nData rows: {len(data)}"
        except Exception as e:
            return f"Error building chart: {e}\nSQL: {sql}\nData sample: {data[:3]}"

    return create_chart
