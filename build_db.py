"""
Load election JSON data into a normalized SQLite database.
Run once: python build_db.py
Creates: elections.db
"""
import json, sqlite3, os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Downloads",
                        "election-dashboard-main", "election-dashboard-main", "data")
DB_PATH = os.path.join(os.path.dirname(__file__), "elections.db")


def load_json(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def build():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ── 1. elections table (national-level stats per knesset) ──
    c.execute("""CREATE TABLE elections (
        knesset INTEGER PRIMARY KEY,
        year    INTEGER,
        total_eligible INTEGER,
        localities_count INTEGER,
        turnout_pct REAL,
        right_pct REAL, haredi_pct REAL, center_pct REAL,
        left_pct REAL, arab_pct REAL, opposition_right_pct REAL,
        right_haredi_pct REAL, center_left_arab_pct REAL
    )""")

    core = load_json("core.json")
    for k, info in core["national"]["elections"].items():
        c.execute("""INSERT INTO elections VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (int(k), info["year"], info["total_eligible"],
                   info["localities_count"], info["turnout_pct"],
                   info["right_pct"], info["haredi_pct"], info["center_pct"],
                   info["left_pct"], info["arab_pct"], info.get("opposition_right_pct", 0),
                   info["right_haredi_pct"], info["center_left_arab_pct"]))

    # ── 2. parties table (party results per knesset) ──
    c.execute("""CREATE TABLE parties (
        knesset INTEGER, code TEXT, name TEXT, bloc TEXT,
        vote_pct REAL, votes INTEGER, seats INTEGER,
        PRIMARY KEY (knesset, code)
    )""")

    parties_nat = load_json("parties_national.json")
    for k, edata in parties_nat.items():
        ki = int(k)
        party_lookup = {p["code"]: p for p in edata["party_list"]}
        for code, pct in edata["national"].items():
            p = party_lookup.get(code, {})
            c.execute("INSERT INTO parties VALUES (?,?,?,?,?,?,?)",
                      (ki, code, p.get("name", code), p.get("bloc", "unknown"),
                       pct, edata["national_votes"].get(code, 0),
                       edata["seats"].get(code, 0)))

    # ── 3. localities table (per-locality per-election bloc data) ──
    c.execute("""CREATE TABLE localities (
        name TEXT, knesset INTEGER,
        eligible INTEGER, turnout_pct REAL,
        right_pct REAL, haredi_pct REAL, center_pct REAL,
        left_pct REAL, arab_pct REAL,
        right_haredi_pct REAL, center_left_arab_pct REAL,
        PRIMARY KEY (name, knesset)
    )""")

    locs = load_json("localities.json")
    for loc in locs:
        name = loc["name"]
        for k, d in loc["data"].items():
            c.execute("INSERT OR IGNORE INTO localities VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (name, int(k), d.get("eligible", 0),
                       d.get("turnout_pct", 0),
                       d.get("right_pct", 0), d.get("haredi_pct", 0),
                       d.get("center_pct", 0), d.get("left_pct", 0),
                       d.get("arab_pct", 0), d.get("right_haredi_pct", 0),
                       d.get("center_left_arab_pct", 0)))

    # ── 4. party_votes_by_locality (party-level votes per locality per election) ──
    c.execute("""CREATE TABLE party_locality (
        knesset INTEGER, locality TEXT, party_code TEXT, vote_pct REAL,
        PRIMARY KEY (knesset, locality, party_code)
    )""")

    pbl = load_json("parties_by_locality.json")
    for k, localities_dict in pbl.items():
        ki = int(k)
        for loc_name, party_pcts in localities_dict.items():
            for code, pct in party_pcts.items():
                c.execute("INSERT OR IGNORE INTO party_locality VALUES (?,?,?,?)",
                          (ki, loc_name, code, pct))

    # ── 5. socioeconomic table ──
    c.execute("""CREATE TABLE socioeconomic (
        name TEXT PRIMARY KEY,
        population REAL, median_age REAL,
        dependency_ratio REAL,
        pct_academic_degree REAL,
        avg_years_schooling REAL,
        pct_with_work_income REAL,
        avg_monthly_income_per_capita REAL,
        pct_below_min_wage REAL,
        pct_above_2x_avg_wage REAL,
        vehicles_per_100 REAL
    )""")

    socio = load_json("socioeconomic.json")
    for entry in socio:
        c.execute("INSERT OR IGNORE INTO socioeconomic VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (entry["name"], entry.get("population"),
                   entry.get("median_age"), entry.get("dependency_ratio"),
                   entry.get("pct_academic_degree"), entry.get("avg_years_schooling"),
                   entry.get("pct_with_work_income"),
                   entry.get("avg_monthly_income_per_capita"),
                   entry.get("pct_below_min_wage"),
                   entry.get("pct_above_2x_avg_wage"),
                   entry.get("vehicles_per_100_residents")))

    # ── indexes ──
    c.execute("CREATE INDEX idx_parties_knesset ON parties(knesset)")
    c.execute("CREATE INDEX idx_localities_knesset ON localities(knesset)")
    c.execute("CREATE INDEX idx_localities_name ON localities(name)")
    c.execute("CREATE INDEX idx_pl_knesset ON party_locality(knesset)")
    c.execute("CREATE INDEX idx_pl_locality ON party_locality(locality)")

    conn.commit()

    # summary
    for tbl in ["elections", "parties", "localities", "party_locality", "socioeconomic"]:
        cnt = c.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl}: {cnt} rows")

    conn.close()
    print(f"\nDatabase saved to {DB_PATH}")


if __name__ == "__main__":
    build()
