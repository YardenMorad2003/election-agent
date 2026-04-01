"""
Load U.S. election CSV data into the existing elections.db SQLite database.
Run once: python build_us_db.py

Adds tables:
  - us_president_county   (county-level presidential results, 2000-2024)
  - us_president_precinct (precinct-level presidential results, 2016/2020/2024)
  - us_house_precinct     (precinct-level House results, 2016/2018/2020)
  - us_senate_precinct    (precinct-level Senate results, 2016/2018/2020)
"""
import csv, sqlite3, os, sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(os.path.dirname(__file__), "elections.db")

# CSV file mappings
COUNTY_FILES = {
    "us_president_county": ["president_county_2000_2024.csv"],
}

PRECINCT_FILES = {
    "us_president_precinct": [
        "president_2016_precinct.csv",
        "president_2020_precinct.csv",
        "president_2024_nyt_precinct (1).csv",
    ],
    "us_house_precinct": [
        "house_2016_precinct.csv",
        "house_2018_precinct.csv",
        "house_2020_precinct.csv",
    ],
    "us_senate_precinct": [
        "senate_2016_precinct.csv",
        "senate_2018_precinct.csv",
        "senate_2020_precinct.csv",
    ],
}

# Unified schema columns (same for county and precinct tables)
COUNTY_COLS = [
    "year", "state", "state_fips", "county_name", "county_fips",
    "candidate", "party", "votes", "nchs_code", "nchs_label",
    "urban_rural", "cbsa_title",
]

PRECINCT_COLS = [
    "year", "state", "state_fips", "county_name", "county_fips",
    "precinct", "district", "candidate", "party", "votes",
    "nchs_code", "nchs_label", "urban_rural", "cbsa_title",
]


def _clean_fips(val, width):
    """Zero-pad a FIPS code to the given width, handling floats like '1.0'."""
    if not val or val.strip() == "":
        return None
    try:
        return str(int(float(val))).zfill(width)
    except (ValueError, TypeError):
        return val.strip().zfill(width)


def _clean_int(val):
    """Convert a possibly-float string like '4942.0' to int."""
    if not val or val.strip() == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _clean_nchs(val):
    """Convert NCHS code float like '3.0' to int."""
    if not val or val.strip() == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _load_csv(filepath, table_type="precinct"):
    """Read a CSV and yield cleaned row tuples."""
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = _clean_int(row.get("year"))
            state = (row.get("state") or "").strip()
            state_fips = _clean_fips(row.get("state_fips"), 2)
            county_name = (row.get("county_name") or "").strip()
            county_fips = _clean_fips(row.get("county_fips"), 5)
            candidate = (row.get("candidate") or "").strip()
            party = (row.get("party") or "").strip()
            votes = _clean_int(row.get("votes"))
            nchs_code = _clean_nchs(row.get("nchs_code"))
            nchs_label = (row.get("nchs_label") or "").strip() or None
            urban_rural = (row.get("urban_rural") or "").strip() or None
            cbsa_title = (row.get("cbsa_title") or "").strip() or None

            if table_type == "county":
                yield (year, state, state_fips, county_name, county_fips,
                       candidate, party, votes, nchs_code, nchs_label,
                       urban_rural, cbsa_title)
            else:
                precinct = (row.get("precinct") or "").strip() or None
                district = (row.get("district") or "").strip() or None
                yield (year, state, state_fips, county_name, county_fips,
                       precinct, district, candidate, party, votes,
                       nchs_code, nchs_label, urban_rural, cbsa_title)


def build(load_precinct=True):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Drop existing U.S. tables if they exist (for re-runs)
    for tbl in ["us_president_county", "us_president_precinct",
                "us_house_precinct", "us_senate_precinct"]:
        c.execute(f"DROP TABLE IF EXISTS {tbl}")

    # ── County table ──
    c.execute("""CREATE TABLE us_president_county (
        year INTEGER,
        state TEXT,
        state_fips TEXT,
        county_name TEXT,
        county_fips TEXT,
        candidate TEXT,
        party TEXT,
        votes INTEGER,
        nchs_code INTEGER,
        nchs_label TEXT,
        urban_rural TEXT,
        cbsa_title TEXT,
        PRIMARY KEY (year, county_fips, candidate)
    )""")

    for fname in COUNTY_FILES["us_president_county"]:
        path = os.path.join(DATA_DIR, fname)
        print(f"  Loading {fname}...")
        rows = list(_load_csv(path, table_type="county"))
        c.executemany(
            f"INSERT OR IGNORE INTO us_president_county VALUES ({','.join('?' * len(COUNTY_COLS))})",
            rows,
        )
        print(f"    -> {len(rows)} rows")

    c.execute("CREATE INDEX idx_upc_year ON us_president_county(year)")
    c.execute("CREATE INDEX idx_upc_state ON us_president_county(state)")
    c.execute("CREATE INDEX idx_upc_county ON us_president_county(county_fips)")

    if load_precinct:
        # ── Precinct tables ──
        for table_name, files in PRECINCT_FILES.items():
            c.execute(f"""CREATE TABLE {table_name} (
                year INTEGER,
                state TEXT,
                state_fips TEXT,
                county_name TEXT,
                county_fips TEXT,
                precinct TEXT,
                district TEXT,
                candidate TEXT,
                party TEXT,
                votes INTEGER,
                nchs_code INTEGER,
                nchs_label TEXT,
                urban_rural TEXT,
                cbsa_title TEXT
            )""")

            total = 0
            for fname in files:
                path = os.path.join(DATA_DIR, fname)
                if not os.path.exists(path):
                    print(f"  WARNING: {fname} not found, skipping.")
                    continue
                print(f"  Loading {fname} into {table_name}...")
                batch = []
                for row in _load_csv(path, table_type="precinct"):
                    batch.append(row)
                    if len(batch) >= 50000:
                        c.executemany(
                            f"INSERT INTO {table_name} VALUES ({','.join('?' * len(PRECINCT_COLS))})",
                            batch,
                        )
                        total += len(batch)
                        batch = []
                if batch:
                    c.executemany(
                        f"INSERT INTO {table_name} VALUES ({','.join('?' * len(PRECINCT_COLS))})",
                        batch,
                    )
                    total += len(batch)
                print(f"    -> {total} rows so far")

            c.execute(f"CREATE INDEX idx_{table_name}_year ON {table_name}(year)")
            c.execute(f"CREATE INDEX idx_{table_name}_state ON {table_name}(state)")
            c.execute(f"CREATE INDEX idx_{table_name}_county ON {table_name}(county_fips)")
            print(f"  {table_name}: {total} total rows")

    conn.commit()

    # Summary
    print("\n=== U.S. Tables Summary ===")
    for tbl in ["us_president_county", "us_president_precinct",
                "us_house_precinct", "us_senate_precinct"]:
        try:
            cnt = c.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            print(f"  {tbl}: {cnt:,} rows")
        except Exception:
            print(f"  {tbl}: not loaded")

    conn.close()
    print(f"\nDatabase updated: {DB_PATH}")


if __name__ == "__main__":
    # Pass --county-only to skip the large precinct files
    county_only = "--county-only" in sys.argv
    build(load_precinct=not county_only)
