"""
Migrate election data from SQLite to PostgreSQL.
Run once: python migrate_to_postgres.py

Reads from: elections.db (SQLite)
Writes to: PostgreSQL election_agent database
"""
import sqlite3
import os
import sys
import time

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    print("Installing psycopg2-binary...")
    os.system(f"{sys.executable} -m pip install psycopg2-binary")
    import psycopg2
    from psycopg2.extras import execute_values

from dotenv import load_dotenv
load_dotenv()

SQLITE_PATH = os.path.join(os.path.dirname(__file__), "elections.db")
PG_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:Donlemon2020;@localhost:5432/election_agent"
)


def migrate():
    t0 = time.time()

    # Connect to both databases
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_conn.row_factory = sqlite3.Row
    pg_conn = psycopg2.connect(PG_URL)
    pg_cur = pg_conn.cursor()

    # ── Israeli Tables ──

    print("Creating Israeli tables...")

    pg_cur.execute("""
        DROP TABLE IF EXISTS party_locality CASCADE;
        DROP TABLE IF EXISTS localities CASCADE;
        DROP TABLE IF EXISTS parties CASCADE;
        DROP TABLE IF EXISTS elections CASCADE;
        DROP TABLE IF EXISTS socioeconomic CASCADE;
    """)

    pg_cur.execute("""
        CREATE TABLE elections (
            knesset INTEGER PRIMARY KEY,
            year INTEGER,
            total_eligible INTEGER,
            localities_count INTEGER,
            turnout_pct REAL,
            right_pct REAL, haredi_pct REAL, center_pct REAL,
            left_pct REAL, arab_pct REAL, opposition_right_pct REAL,
            right_haredi_pct REAL, center_left_arab_pct REAL
        )
    """)

    pg_cur.execute("""
        CREATE TABLE parties (
            knesset INTEGER, code TEXT, name TEXT, bloc TEXT,
            vote_pct REAL, votes INTEGER, seats INTEGER,
            PRIMARY KEY (knesset, code)
        )
    """)

    pg_cur.execute("""
        CREATE TABLE localities (
            name TEXT, knesset INTEGER,
            eligible INTEGER, turnout_pct REAL,
            right_pct REAL, haredi_pct REAL, center_pct REAL,
            left_pct REAL, arab_pct REAL,
            right_haredi_pct REAL, center_left_arab_pct REAL,
            PRIMARY KEY (name, knesset)
        )
    """)

    pg_cur.execute("""
        CREATE TABLE party_locality (
            knesset INTEGER, locality TEXT, party_code TEXT, vote_pct REAL,
            PRIMARY KEY (knesset, locality, party_code)
        )
    """)

    pg_cur.execute("""
        CREATE TABLE socioeconomic (
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
        )
    """)

    # ── U.S. Tables ──

    print("Creating U.S. tables...")

    pg_cur.execute("""
        DROP TABLE IF EXISTS us_president_county CASCADE;
        DROP TABLE IF EXISTS us_president_precinct CASCADE;
        DROP TABLE IF EXISTS us_house_precinct CASCADE;
        DROP TABLE IF EXISTS us_senate_precinct CASCADE;
    """)

    pg_cur.execute("""
        CREATE TABLE us_president_county (
            year INTEGER, state TEXT, state_fips TEXT,
            county_name TEXT, county_fips TEXT,
            candidate TEXT, party TEXT, votes INTEGER,
            nchs_code INTEGER, nchs_label TEXT,
            urban_rural TEXT, cbsa_title TEXT
        )
    """)

    # Precinct tables share the same schema
    for table in ["us_president_precinct", "us_house_precinct", "us_senate_precinct"]:
        pg_cur.execute(f"""
            CREATE TABLE {table} (
                year INTEGER, state TEXT, state_fips TEXT,
                county_name TEXT, county_fips TEXT,
                precinct TEXT, district TEXT,
                candidate TEXT, party TEXT, votes INTEGER,
                nchs_code INTEGER, nchs_label TEXT,
                urban_rural TEXT, cbsa_title TEXT
            )
        """)

    pg_conn.commit()

    # ── Copy data ──

    def copy_table(table_name, batch_size=5000):
        rows = sqlite_conn.execute(f"SELECT * FROM {table_name}").fetchall()
        if not rows:
            print(f"  {table_name}: 0 rows (skipped)")
            return
        cols = rows[0].keys()
        n_cols = len(cols)
        col_list = ", ".join(cols)
        template = "(" + ", ".join(["%s"] * n_cols) + ")"

        total = len(rows)
        for i in range(0, total, batch_size):
            batch = [tuple(row) for row in rows[i:i + batch_size]]
            execute_values(
                pg_cur,
                f"INSERT INTO {table_name} ({col_list}) VALUES %s",
                batch,
                template=template,
            )
            done = min(i + batch_size, total)
            pct = done / total * 100
            sys.stderr.write(f"\r  {table_name}: {done:,}/{total:,} ({pct:.0f}%)")
            sys.stderr.flush()

        pg_conn.commit()
        sys.stderr.write("\n")
        print(f"  {table_name}: {total:,} rows")

    # Israeli tables (small, fast)
    print("\nCopying Israeli data...")
    for t in ["elections", "parties", "localities", "party_locality", "socioeconomic"]:
        copy_table(t)

    # U.S. tables (large)
    print("\nCopying U.S. data...")
    for t in ["us_president_county", "us_president_precinct", "us_house_precinct", "us_senate_precinct"]:
        copy_table(t, batch_size=10000)

    # ── Indexes ──
    print("\nCreating indexes...")
    pg_cur.execute("CREATE INDEX IF NOT EXISTS idx_parties_knesset ON parties(knesset)")
    pg_cur.execute("CREATE INDEX IF NOT EXISTS idx_localities_knesset ON localities(knesset)")
    pg_cur.execute("CREATE INDEX IF NOT EXISTS idx_localities_name ON localities(name)")
    pg_cur.execute("CREATE INDEX IF NOT EXISTS idx_pl_knesset ON party_locality(knesset)")
    pg_cur.execute("CREATE INDEX IF NOT EXISTS idx_pl_locality ON party_locality(locality)")
    pg_cur.execute("CREATE INDEX IF NOT EXISTS idx_us_county_year ON us_president_county(year)")
    pg_cur.execute("CREATE INDEX IF NOT EXISTS idx_us_county_state ON us_president_county(state)")
    pg_conn.commit()

    sqlite_conn.close()
    pg_cur.close()
    pg_conn.close()

    elapsed = time.time() - t0
    print(f"\nMigration complete! ({elapsed:.1f}s)")


if __name__ == "__main__":
    migrate()
