"""
Build ChromaDB vector store from election data.
Run once (after build_us_db.py): python build_vectorstore.py

Creates: ./chroma_db/ directory with embedded election summaries.
Uses local sentence-transformers model (no API key needed).
"""
import os, sys, sqlite3, shutil, time
from dotenv import load_dotenv
load_dotenv()

from embeddings import LocalEmbeddings
from langchain_community.vectorstores import Chroma

DB_PATH = os.path.join(os.path.dirname(__file__), "elections.db")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def build_us_county_chunks(conn):
    """A. County-level election summaries."""
    chunks, metadatas = [], []

    # Get all (year, state, county) groups with vote totals
    rows = conn.execute("""
        SELECT year, state, county_name, county_fips, urban_rural, nchs_label,
               candidate, party, votes
        FROM us_president_county
        ORDER BY year, state, county_fips, votes DESC
    """).fetchall()

    # Group by (year, county_fips)
    from itertools import groupby
    keyfunc = lambda r: (r["year"], r["state"], r["county_name"], r["county_fips"],
                         r["urban_rural"], r["nchs_label"])

    for key, group in groupby(rows, key=keyfunc):
        year, state, county, fips, urban_rural, nchs_label = key
        candidates = list(group)
        total = sum(c["votes"] or 0 for c in candidates)
        if total == 0:
            continue

        top = []
        for c in candidates[:4]:  # top 4 candidates
            pct = (c["votes"] / total * 100) if total else 0
            top.append(f"{c['candidate']} ({c['party']}) received {c['votes']:,} votes ({pct:.1f}%)")

        urban_str = f" ({nchs_label or urban_rural})" if (nchs_label or urban_rural) else ""
        text = (f"In {year}, {county}, {state}{urban_str}: "
                f"{'; '.join(top)}. Total votes: {total:,}.")

        chunks.append(text)
        metadatas.append({
            "source": "us_county",
            "year": year,
            "state": state,
            "country": "US",
        })

    print(f"  County chunks: {len(chunks)}")
    return chunks, metadatas


def build_us_state_chunks(conn):
    """B. State-level aggregated summaries."""
    chunks, metadatas = [], []

    rows = conn.execute("""
        SELECT year, state, party, urban_rural, SUM(votes) as total_votes
        FROM us_president_county
        GROUP BY year, state, party, urban_rural
        ORDER BY year, state, party
    """).fetchall()

    from itertools import groupby

    # Group by (year, state)
    keyfunc = lambda r: (r["year"], r["state"])
    for key, group in groupby(rows, key=keyfunc):
        year, state = key
        records = list(group)

        # Aggregate by party
        party_totals = {}
        urban_party = {}  # {urban_rural: {party: votes}}
        for r in records:
            party = r["party"]
            votes = r["total_votes"] or 0
            party_totals[party] = party_totals.get(party, 0) + votes
            ur = r["urban_rural"] or "Unknown"
            if ur not in urban_party:
                urban_party[ur] = {}
            urban_party[ur][party] = urban_party[ur].get(party, 0) + votes

        grand_total = sum(party_totals.values())
        if grand_total == 0:
            continue

        # Winner
        winner_party = max(party_totals, key=party_totals.get)
        winner_votes = party_totals[winner_party]
        winner_pct = winner_votes / grand_total * 100

        parts = [f"In {year}, {state}: {winner_party} won with {winner_votes:,} votes ({winner_pct:.1f}%)."]

        # Top parties
        for party in ["DEMOCRAT", "REPUBLICAN"]:
            if party in party_totals and party != winner_party:
                v = party_totals[party]
                parts.append(f"{party}: {v:,} ({v/grand_total*100:.1f}%).")

        # Urban breakdown for two-party
        for ur in ["Urban", "Suburban", "Rural"]:
            if ur in urban_party:
                dem = urban_party[ur].get("DEMOCRAT", 0)
                rep = urban_party[ur].get("REPUBLICAN", 0)
                two_party = dem + rep
                if two_party > 0:
                    parts.append(f"{ur}: {dem/two_party*100:.0f}% Dem.")

        text = " ".join(parts)
        chunks.append(text)
        metadatas.append({
            "source": "us_state_summary",
            "year": year,
            "state": state,
            "country": "US",
        })

    print(f"  State summary chunks: {len(chunks)}")
    return chunks, metadatas


def build_nchs_chunks():
    """C. NCHS/urbanization context chunks."""
    chunks = [
        ("The NCHS Urban-Rural Classification divides U.S. counties into 6 categories: "
         "Large central metro (code 1), Large fringe metro (code 2), Medium metro (code 3), "
         "Small metro (code 4), Micropolitan (code 5), and Noncore (code 6). "
         "Codes 1 and 3 are classified as Urban, codes 2 and 4 as Suburban, "
         "and codes 5 and 6 as Rural."),
        ("The urban_rural column in U.S. election data is a simplified 3-bucket classification: "
         "Urban (large central + medium metro), Suburban (large fringe + small metro), "
         "and Rural (micropolitan + noncore). This enables quick urban/suburban/rural analysis."),
        ("U.S. presidential election data is available at county level from 2000 to 2024 "
         "(7 elections). Precinct-level data is available for 2016, 2020, and 2024. "
         "House and Senate precinct data covers 2016-2020."),
    ]
    metadatas = [{"source": "nchs_context", "country": "US"} for _ in chunks]
    print(f"  NCHS context chunks: {len(chunks)}")
    return chunks, metadatas


def build_israeli_chunks(conn):
    """E. Israeli election summaries (migrated from agent.py _build_rag_chunks)."""
    chunks, metadatas = [], []

    # Election summaries
    for row in conn.execute("SELECT * FROM elections ORDER BY knesset"):
        text = (
            f"Knesset {row['knesset']} ({row['year']}): "
            f"Eligible voters: {row['total_eligible']:,}. Turnout: {row['turnout_pct']}%. "
            f"Right: {row['right_pct']}%, Haredi: {row['haredi_pct']}%, "
            f"Center: {row['center_pct']}%, Left: {row['left_pct']}%, "
            f"Arab: {row['arab_pct']}%. "
            f"Right+Haredi bloc: {row['right_haredi_pct']}%, "
            f"Center+Left+Arab bloc: {row['center_left_arab_pct']}%."
        )
        chunks.append(text)
        metadatas.append({
            "source": "israeli_election",
            "year": row["year"],
            "country": "IL",
        })

    # Party results
    for row in conn.execute(
        "SELECT knesset, name, bloc, vote_pct, seats FROM parties WHERE seats>0 ORDER BY knesset, seats DESC"
    ):
        text = (f"K{row['knesset']}: {row['name']} ({row['bloc']}) - "
                f"{row['vote_pct']}% of votes, {row['seats']} seats.")
        chunks.append(text)
        metadatas.append({
            "source": "israeli_election",
            "year": None,
            "country": "IL",
        })

    # Socioeconomic
    for row in conn.execute("SELECT * FROM socioeconomic LIMIT 201"):
        text = (
            f"Socioeconomic - {row['name']}: pop {row['population']:.0f}, "
            f"median age {row['median_age']}, "
            f"academic degree {row['pct_academic_degree']:.1f}%, "
            f"income/capita {row['avg_monthly_income_per_capita']:.0f} NIS."
        )
        chunks.append(text)
        metadatas.append({
            "source": "israeli_election",
            "country": "IL",
        })

    print(f"  Israeli chunks: {len(chunks)}")
    return chunks, metadatas


def build_documentation_chunks():
    """D. Dataset documentation chunks."""
    docs = [
        ("The election database contains both U.S. federal election data and Israeli Knesset "
         "election data. U.S. data includes presidential results at county level (2000-2024) "
         "and precinct level (2016, 2020, 2024), plus House and Senate precinct data (2016-2020). "
         "Israeli data covers Knesset elections 14-25 (1996-2022)."),
        ("U.S. county-level presidential data (us_president_county) is the primary table for "
         "aggregated analysis. It contains ~75,000 rows across 7 elections and ~3,100 counties. "
         "Each row includes the candidate name, party, vote count, and NCHS urban-rural classification."),
        ("U.S. party labels: DEMOCRAT, REPUBLICAN, LIBERTARIAN, and OTHER. "
         "For two-party vote share calculations, filter to DEMOCRAT and REPUBLICAN only. "
         "Candidate names are uppercase (e.g., JOSEPH R BIDEN, DONALD J TRUMP, BARACK OBAMA)."),
        ("Israeli Knesset data includes: elections table (national stats per knesset), "
         "parties table (party results with seats and bloc), localities table (per-locality "
         "bloc breakdowns for 1,384 localities), and socioeconomic table (201 municipalities "
         "with income, education, and demographic indicators)."),
        ("Israeli political bloc definitions: "
         "'right' bloc includes Likud and Religious Zionism (Jewish Home). "
         "'haredi' (ultra-Orthodox) bloc includes Shas and United Torah Judaism (UTJ). "
         "'center' bloc includes Yesh Atid, National Unity (formerly Blue & White), and historically Kadima. "
         "'left' bloc includes Labor and Meretz. "
         "'arab' bloc includes Ra'am, Hadash-Ta'al, Balad, and the Joint List (when running together). "
         "'opposition_right' refers to Yisrael Beiteinu, a right-leaning secular party that has historically sat in opposition."),
        ("Israeli aggregate bloc groupings: "
         "'Right+Haredi bloc' (right_haredi_pct) combines the right and haredi blocs. "
         "This typically represents the natural coalition partners: Likud + Religious Zionism + Shas + UTJ. "
         "In K25, the Right+Haredi bloc won 64 seats and formed the governing coalition. "
         "'Center+Left+Arab bloc' (center_left_arab_pct) combines center, left, and arab blocs. "
         "This represents the potential opposition or alternative coalition: Yesh Atid + National Unity + Labor + Meretz + Arab parties. "
         "These two blocs together account for nearly all votes, with opposition_right (Yisrael Beiteinu) as the swing factor."),
        ("Israeli coalition formation: After each Knesset election, the president tasks a party leader "
         "(usually the largest party) with forming a coalition of 61+ seats (out of 120). "
         "The right_haredi_pct and center_left_arab_pct fields show which bloc grouping has the majority. "
         "Notable coalitions: K25 (2022) Likud+RZ+Shas+UTJ = 64 seats. "
         "K24 (2021) 'change government' coalition spanning Yesh Atid, National Unity, Labor, Meretz, Ra'am, and Yisrael Beiteinu. "
         "K20 (2015) Likud+Kulanu+Jewish Home+Shas+UTJ = 61 seats (narrowest possible)."),
    ]
    metadatas = [{"source": "documentation", "country": "both"} for _ in docs]
    print(f"  Documentation chunks: {len(docs)}")
    return docs, metadatas


def _progress_bar(current, total, bar_len=40, label=""):
    """Print a progress bar to stderr (flushes in-place)."""
    frac = current / total if total else 1
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    pct = frac * 100
    sys.stderr.write(f"\r  {label} [{bar}] {pct:5.1f}%  ({current}/{total})")
    sys.stderr.flush()
    if current >= total:
        sys.stderr.write("\n")


def build():
    t0 = time.time()

    # Clean existing ChromaDB
    if os.path.exists(CHROMA_DIR):
        try:
            shutil.rmtree(CHROMA_DIR)
            print("Removed existing chroma_db/")
        except PermissionError:
            # On Windows, files may be locked by another process
            backup = CHROMA_DIR + "_old"
            if os.path.exists(backup):
                shutil.rmtree(backup, ignore_errors=True)
            os.rename(CHROMA_DIR, backup)
            print(f"Renamed existing chroma_db/ to chroma_db_old/ (locked files)")

    conn = _get_conn()

    # ── Step 1: Build chunks ──
    print("\n[1/2] Building text chunks from database...")
    all_chunks, all_metadatas = [], []

    builders_db = [
        ("County summaries", build_us_county_chunks),
        ("State summaries", build_us_state_chunks),
    ]
    builders_static = [
        ("NCHS context", build_nchs_chunks),
        ("Documentation", build_documentation_chunks),
    ]
    builders_il = [
        ("Israeli data", build_israeli_chunks),
    ]

    total_builders = len(builders_db) + len(builders_static) + len(builders_il)
    done = 0

    for label, builder in builders_db:
        try:
            c, m = builder(conn)
            all_chunks.extend(c)
            all_metadatas.extend(m)
        except Exception as e:
            print(f"  WARNING: {label} failed: {e}")
        done += 1
        _progress_bar(done, total_builders, label="Chunks")

    for label, builder in builders_static:
        c, m = builder()
        all_chunks.extend(c)
        all_metadatas.extend(m)
        done += 1
        _progress_bar(done, total_builders, label="Chunks")

    for label, builder in builders_il:
        try:
            c, m = builder(conn)
            all_chunks.extend(c)
            all_metadatas.extend(m)
        except Exception as e:
            print(f"  WARNING: {label} failed: {e}")
        done += 1
        _progress_bar(done, total_builders, label="Chunks")

    conn.close()
    print(f"  Total chunks: {len(all_chunks)}")

    # ── Step 2: Embed and store ──
    print(f"\n[2/2] Embedding {len(all_chunks)} chunks into ChromaDB...")

    embedding_fn = LocalEmbeddings()

    vectorstore = Chroma(
        collection_name="election_data",
        embedding_function=embedding_fn,
        persist_directory=CHROMA_DIR,
    )

    batch_size = 500
    total_chunks = len(all_chunks)
    for i in range(0, total_chunks, batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_meta = all_metadatas[i:i + batch_size]
        vectorstore.add_texts(texts=batch_chunks, metadatas=batch_meta)
        _progress_bar(min(i + batch_size, total_chunks), total_chunks, label="Embed")

    elapsed = time.time() - t0
    print(f"\nChromaDB saved to {CHROMA_DIR}")
    print(f"Done! ({elapsed:.1f}s)")


if __name__ == "__main__":
    build()
