"""Regression test for the web_search -> _format_web_answer synthesis path.

Exercises the same code the Streamlit UI hits when the agent calls web_search,
across a fixed set of probe queries. Reports per-query pass/fail and timing.

Usage:
    python -m benchmark.test_web_search_synthesis
    python -m benchmark.test_web_search_synthesis --runs 3   # repeat each query
    python -m benchmark.test_web_search_synthesis --delay 2  # seconds between calls
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import _format_web_answer, get_llm
from tools.web_search import web_search


# (query, list of acceptable substrings in the answer, group)
# Substring matching is case-insensitive. Pass = any acceptable substring appears.
# `expected=None` means we record the answer but don't pass/fail it.
PROBES: list[tuple[str, list[str] | None, str]] = [
    # Current US officeholders
    ("who is the us president?",                              ["Trump"],                  "US-current"),
    ("who is the current us president",                       ["Trump"],                  "US-current"),
    ("who is the president of the united states",             ["Trump"],                  "US-current"),
    ("current president of the united states",                ["Trump"],                  "US-current"),
    ("us president right now",                                ["Trump"],                  "US-current"),
    ("who is the us vice president",                          ["Vance"],                  "US-current"),
    ("who is the speaker of the us house",                    ["Johnson"],                "US-current"),
    ("who is the us secretary of state",                      ["Rubio"],                  "US-current"),
    ("who is the us secretary of defense",                    ["Hegseth"],                "US-current"),
    ("who is the us secretary of the treasury",               ["Bessent"],                "US-current"),
    ("who is the us attorney general",                        ["Bondi"],                  "US-current"),
    ("who is the us secretary of homeland security",          ["Noem"],                   "US-current"),
    ("who is the senate majority leader",                     ["Thune"],                  "US-current"),
    ("who is the senate minority leader",                     ["Schumer"],                "US-current"),
    ("who is the chief justice of the supreme court",         ["Roberts"],                "US-current"),
    # International leaders
    ("who is the prime minister of israel",                   ["Netanyahu"],              "Intl-current"),
    ("who is the current israeli prime minister",             ["Netanyahu"],              "Intl-current"),
    ("who is the prime minister of the uk",                   ["Starmer"],                "Intl-current"),
    ("who is the chancellor of germany",                      ["Merz"],                   "Intl-current"),
    ("who is the president of france",                        ["Macron"],                 "Intl-current"),
    ("who is the prime minister of canada",                   ["Carney"],                 "Intl-current"),
    ("who is the prime minister of australia",                ["Albanese"],               "Intl-current"),
    ("who is the prime minister of india",                    ["Modi"],                   "Intl-current"),
    ("who is the prime minister of japan",                    None,                       "Intl-current"),
    ("who is the president of mexico",                        ["Sheinbaum"],              "Intl-current"),
    ("who is the president of brazil",                        ["Lula", "da Silva"],       "Intl-current"),
    # US states
    ("who is the governor of california",                     ["Newsom"],                 "US-states"),
    ("who is the governor of texas",                          ["Abbott"],                 "US-states"),
    ("who is the governor of new york",                       ["Hochul"],                 "US-states"),
    ("who is the governor of florida",                        ["DeSantis"],               "US-states"),
    ("who is the mayor of new york city",                     None,                       "US-states"),
    ("who is the mayor of los angeles",                       ["Bass"],                   "US-states"),
    ("who is the mayor of chicago",                           ["Johnson"],                "US-states"),
    # Historical (control — should NOT regress)
    ("who was the first us president",                        ["Washington"],             "Historical"),
    ("who was the 16th us president",                         ["Lincoln"],                "Historical"),
    ("who wrote hamlet",                                      ["Shakespeare"],            "Historical"),
    ("who invented the telephone",                            ["Bell"],                   "Historical"),
    ("who painted the mona lisa",                             ["Vinci", "Leonardo"],      "Historical"),
    ("who was the first man on the moon",                     ["Armstrong"],              "Historical"),
    ("who founded apple",                                     ["Jobs"],                   "Historical"),
    ("who wrote 1984",                                        ["Orwell"],                 "Historical"),
    ("when did world war 2 end",                              ["1945"],                   "Historical"),
    ("when did the berlin wall fall",                         ["1989"],                   "Historical"),
    # Geography / general (control)
    ("what is the capital of france",                         ["Paris"],                  "Geography"),
    ("what is the capital of japan",                          ["Tokyo"],                  "Geography"),
    ("what is the capital of australia",                      ["Canberra"],               "Geography"),
    ("what is the largest country by area",                   ["Russia"],                 "Geography"),
    ("what is the longest river in the world",                ["Nile", "Amazon"],         "Geography"),
    ("what is the highest mountain",                          ["Everest"],                "Geography"),
    ("what is the smallest country in the world",             ["Vatican"],                "Geography"),
    ("what is the most populous country",                     ["India", "China"],         "Geography"),
]


def check(answer: str, expected: list[str] | None) -> str:
    if expected is None:
        return "REC"  # recorded, no judgment
    lower = answer.lower()
    return "OK" if any(e.lower() in lower for e in expected) else "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between queries")
    ap.add_argument("--runs", type=int, default=1, help="repeat each query N times")
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    llm = get_llm(args.model, 0)
    rows = []
    for q, expected, group in PROBES:
        for run_i in range(args.runs):
            t0 = time.time()
            try:
                tool_out = web_search.invoke(q)
                ans = _format_web_answer(q, tool_out, llm)
                first = ans.split("\n", 1)[0].strip()
                status = check(first, expected)
                err = ""
            except Exception as e:
                first, status, err = "", "ERR", str(e)
            elapsed = time.time() - t0
            rows.append((group, q, expected, first, status, err, elapsed))
            tag = f"[{status}]"
            print(f"{tag:<7} {group:<14} {q[:46]:<48} -> {first[:80]}  ({elapsed:.1f}s)")
            time.sleep(args.delay)

    print()
    print("=" * 100)
    counts = {"OK": 0, "FAIL": 0, "REC": 0, "ERR": 0}
    by_group: dict[str, dict[str, int]] = {}
    for group, q, expected, ans, status, err, elapsed in rows:
        counts[status] = counts.get(status, 0) + 1
        by_group.setdefault(group, {"OK": 0, "FAIL": 0, "REC": 0, "ERR": 0})
        by_group[group][status] = by_group[group].get(status, 0) + 1
    total_judged = counts["OK"] + counts["FAIL"]
    print(f"Total: {len(rows)}  |  OK={counts['OK']}  FAIL={counts['FAIL']}  REC={counts['REC']}  ERR={counts['ERR']}")
    if total_judged:
        print(f"Pass rate (judged only): {counts['OK']}/{total_judged} = {100*counts['OK']/total_judged:.1f}%")
    print()
    print("Per group:")
    for g, c in by_group.items():
        judged = c["OK"] + c["FAIL"]
        pct = f"{100*c['OK']/judged:.0f}%" if judged else "-"
        print(f"  {g:<14}  OK={c['OK']:>2}  FAIL={c['FAIL']:>2}  REC={c['REC']:>2}  ERR={c['ERR']:>2}  ({pct})")

    fails = [r for r in rows if r[4] == "FAIL"]
    if fails:
        print()
        print("Failures:")
        for group, q, expected, ans, status, err, elapsed in fails:
            print(f"  [{group}] {q}")
            print(f"     expected: {expected}")
            print(f"     got     : {ans[:120]}")


if __name__ == "__main__":
    main()
