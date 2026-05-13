"""
Database connection manager — supports PostgreSQL (primary) and SQLite (fallback).

Uses DATABASE_URL from .env if set, otherwise falls back to SQLite.
All connections are read-only to prevent LLM-generated SQL from modifying data.
"""
import os
import sqlite3
from dotenv import load_dotenv
load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "elections.db")
DATABASE_URL = os.environ.get("DATABASE_URL")

# Forbidden SQL keywords — blocks destructive queries from LLM-generated SQL
_FORBIDDEN = ("DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "ATTACH", "DETACH", "CREATE", "REPLACE")


def get_connection():
    """Return a database connection (PostgreSQL if DATABASE_URL is set, else SQLite read-only)."""
    if DATABASE_URL:
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(DATABASE_URL, options="-c default_transaction_read_only=on")
        return conn
    else:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn


def validate_sql(sql: str) -> str | None:
    """Validate SQL is safe to execute. Returns error message if blocked, None if OK."""
    sql_upper = sql.strip().upper()
    for forbidden in _FORBIDDEN:
        if forbidden in sql_upper and not sql_upper.startswith("SELECT"):
            return f"Blocked: SQL contains forbidden keyword '{forbidden}'."
    return None


def enforce_limit(sql: str, limit: int = 50) -> str:
    """Append LIMIT clause if not already present."""
    if "LIMIT" not in sql.upper():
        sql = sql.rstrip().rstrip(";") + f" LIMIT {limit}"
    return sql


def execute_query(sql: str, params: tuple = ()) -> tuple[list[dict], list[str]]:
    """Execute a read-only SQL query and return (rows_as_dicts, column_names).

    Raises ValueError on SQL errors or blocked queries.
    """
    error = validate_sql(sql)
    if error:
        raise ValueError(error)

    sql = enforce_limit(sql)
    conn = get_connection()

    try:
        cur = conn.cursor()
        # Adapt placeholder style: SQLite uses ?, PostgreSQL uses %s
        if DATABASE_URL and '?' in sql:
            sql = sql.replace('?', '%s')
        # Passing params=() to psycopg2 still triggers %-formatting, which
        # breaks any literal `%` (e.g. LIKE '%word%'). Only pass params when
        # there's something to bind.
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        cols = [desc[0] for desc in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        return rows, cols
    except Exception as e:
        raise ValueError(f"SQL Error: {e}\nQuery: {sql}")
    finally:
        conn.close()


def fetch_all(sql: str, params: tuple = ()) -> list[dict]:
    """Execute a read-only query and return rows as dicts. No SQL validation (for internal use)."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        if DATABASE_URL and '?' in sql:
            sql = sql.replace('?', '%s')
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()
