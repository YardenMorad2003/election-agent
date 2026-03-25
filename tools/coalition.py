"""
Coalition Calculator Tool — finds party combinations that reach 61+ seats.
"""
import sqlite3, os
from itertools import combinations
from langchain_core.tools import tool

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "elections.db")


def _get_parties(knesset: int) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT name, code, bloc, seats FROM parties WHERE knesset=? AND seats>0 ORDER BY seats DESC",
        (knesset,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _find_coalitions(parties: list[dict], min_seats: int = 61,
                     min_parties: int = 2, max_parties: int = 8,
                     must_include: list[str] | None = None,
                     bloc_filter: str | None = None) -> list[dict]:
    results = []
    for size in range(min_parties, min(max_parties + 1, len(parties) + 1)):
        for combo in combinations(parties, size):
            total = sum(p["seats"] for p in combo)
            if total < min_seats:
                continue
            # check must_include
            if must_include:
                names_in_combo = {p["name"] for p in combo}
                codes_in_combo = {p["code"] for p in combo}
                if not all(m in names_in_combo or m in codes_in_combo for m in must_include):
                    continue
            # check bloc filter
            if bloc_filter:
                blocs = {p["bloc"] for p in combo}
                if bloc_filter == "right_bloc" and ("left" in blocs or "arab" in blocs):
                    continue
                if bloc_filter == "left_bloc" and "right" in blocs:
                    continue
            results.append({
                "parties": [f"{p['name']} ({p['seats']})" for p in combo],
                "total_seats": total,
                "num_parties": len(combo),
                "blocs": list({p["bloc"] for p in combo}),
            })
            if len(results) >= 100:
                return results
    return results


@tool
def coalition_calculator(query: str) -> str:
    """Calculate possible coalition governments for a given Knesset election.
    Use for questions about which party combinations can form a majority (61 seats).
    Input should mention the Knesset number (e.g. 'K25' or 'Knesset 25').
    Can also filter by: must-include parties, max coalition size, or bloc constraints."""
    import re

    # parse knesset number
    m = re.search(r'[Kk](?:nesset\s*)?(\d{2})', query)
    if not m:
        m = re.search(r'(\d{2})', query)
    if not m:
        return "Please specify a Knesset number (e.g., K25 or Knesset 25)."
    knesset = int(m.group(1))

    parties = _get_parties(knesset)
    if not parties:
        return f"No party data found for Knesset {knesset}."

    # detect constraints from query
    must_include = []
    if "likud" in query.lower() or "ליכוד" in query:
        must_include.append("מחל")
    if "yesh atid" in query.lower() or "יש עתיד" in query:
        must_include.append("פה")

    max_p = 8
    m2 = re.search(r'(\d)\s*(?:-?\s*)?part(?:y|ies)', query.lower())
    if m2:
        max_p = int(m2.group(1))

    bloc_filter = None
    if "right" in query.lower() and "left" not in query.lower():
        bloc_filter = "right_bloc"
    elif "left" in query.lower() and "right" not in query.lower():
        bloc_filter = "left_bloc"

    coalitions = _find_coalitions(parties, must_include=must_include or None,
                                  max_parties=max_p, bloc_filter=bloc_filter)

    # format output
    party_summary = ", ".join(f"{p['name']} ({p['seats']})" for p in parties)
    lines = [f"Knesset {knesset} — Parties with seats: {party_summary}",
             f"Found {len(coalitions)} possible coalitions"
             + (f" (showing first 100)" if len(coalitions) >= 100 else "") + ":\n"]

    for i, c in enumerate(coalitions[:20], 1):
        lines.append(f"{i}. [{c['total_seats']} seats, {c['num_parties']} parties] "
                      f"{', '.join(c['parties'])}")

    if len(coalitions) > 20:
        lines.append(f"\n... and {len(coalitions) - 20} more coalitions.")

    return "\n".join(lines)
