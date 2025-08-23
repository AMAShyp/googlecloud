# pages/2_Edit_Database.py
from __future__ import annotations

import time
from typing import List, Tuple, Optional, Iterable, Mapping, Any

import pandas as pd
import sqlalchemy as sa
import streamlit as st
from db_handler import DatabaseManager  # uses the revised manager above


st.set_page_config(page_title="Edit / Inspect Database", layout="wide")
st.title("Edit / Inspect Database")


# ───────────────────────────
# Connection management
# ───────────────────────────

@st.cache_resource(show_spinner=False)
def get_db() -> DatabaseManager:
    """
    Cache the DB manager as a resource so reruns don't create new connections.
    IMPORTANT: We don't connect here; engine connects lazily on first use.
    """
    # If you need to pass a config dict, do it here:
    # cfg = st.secrets["cloudsql"]  # or however you load it
    # return DatabaseManager(cfg)
    return DatabaseManager(cfg=st.secrets["cloudsql"])  # adjust to your secrets layout


def _pg8000_err_dict(exc: Exception) -> Optional[dict]:
    """
    Extract the pg8000 error dict from a wrapped SQLAlchemy/DBAPI exception.
    """
    # SQLAlchemy DBAPIError typically exposes .orig
    err = getattr(exc, "orig", exc)
    try:
        first = err.args[0]
        if isinstance(first, dict):
            return first
    except Exception:
        pass
    return None


def _is_conn_slots_exhausted(exc: Exception) -> bool:
    d = _pg8000_err_dict(exc)
    return bool(d and d.get("C") == "53300")


def _friendly_error(exc: Exception) -> str:
    d = _pg8000_err_dict(exc)
    if not d:
        return str(exc)
    code = d.get("C", "")
    msg = d.get("M", str(exc))
    detail = d.get("D", "")
    rel = d.get("t", "")
    name = d.get("n", "")
    extra = []
    if detail:
        extra.append(detail)
    if rel:
        extra.append(f"table={rel}")
    if name:
        extra.append(f"constraint={name}")
    suffix = (" • " + " • ".join(extra)) if extra else ""
    return f"[{code}] {msg}{suffix}"


def _reset_db_resource():
    try:
        db = get_db()
        db.close()
    except Exception:
        pass
    get_db.clear()


with st.sidebar:
    st.subheader("Connection")
    if st.button("Disconnect / reset DB", type="secondary", use_container_width=True):
        _reset_db_resource()
        st.success("Connection pool disposed. Your slot has been released.")


# ───────────────────────────
# Schema helpers (cached)
# ───────────────────────────

@st.cache_data(show_spinner=False, ttl=60)
def get_available_schemas(db_key: str) -> List[str]:
    q = """
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name NOT IN ('pg_catalog','information_schema')
        ORDER BY schema_name;
    """
    db = get_db()
    df = db.fetch_data(q)
    return df["schema_name"].tolist() if not df.empty else ["public"]


@st.cache_data(show_spinner=False, ttl=60)
def get_schema_columns(db_key: str, schema: str) -> pd.DataFrame:
    q = """
        SELECT table_name, column_name, data_type
        FROM   information_schema.columns
        WHERE  table_schema = :schema
        ORDER  BY table_name, ordinal_position;
    """
    db = get_db()
    return db.fetch_data(q, {"schema": schema})


# We key caches by a short "db_key" so if you switch DBs/users, caches separate.
# Here we derive it from secrets minimally.
_db_key = f'{st.secrets["cloudsql"].get("instance_connection_name","")}/{st.secrets["cloudsql"].get("db","")}/{st.secrets["cloudsql"].get("user","")}'


# ───────────────────────────
# Safe SQL splitter
# ───────────────────────────

def split_sql_statements(sql_text: str) -> List[str]:
    """Split SQL into statements (uses sqlparse if available; fallback otherwise)."""
    try:
        import sqlparse  # type: ignore
        stmts = [
            str(stmt).strip()
            for stmt in sqlparse.parse(sql_text or "")
            if str(stmt).strip().strip(";")
        ]
        return [s.rstrip(";").strip() for s in stmts if s.rstrip(";").strip()]
    except Exception:
        s = sql_text or ""
        out, buf = [], []
        in_single = in_double = in_dollar = False
        dollar_tag = ""
        i = 0
        while i < len(s):
            ch = s[i]
            buf.append(ch)

            if not in_single and not in_double:
                if not in_dollar and ch == "$":
                    j = i + 1
                    while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                        j += 1
                    if j < len(s) and s[j] == "$":
                        in_dollar = True
                        dollar_tag = s[i : j + 1]  # like $tag$
                        i = j
                elif in_dollar and ch == "$":
                    tag_len = len(dollar_tag)
                    if i - tag_len + 1 >= 0 and s[i - tag_len + 1 : i + 1] == dollar_tag:
                        in_dollar = False
                        dollar_tag = ""

            if not in_dollar:
                if ch == "'" and not in_double and (i == 0 or s[i - 1] != "\\"):
                    in_single = not in_single
                elif ch == '"' and not in_single and (i == 0 or s[i - 1] != "\\"):
                    in_double = not in_double
                elif ch == ";" and not in_single and not in_double:
                    stmt = "".join(buf).strip()[:-1].strip()
                    if stmt:
                        out.append(stmt)
                    buf = []
            i += 1

        tail = "".join(buf).strip()
        if tail:
            out.append(tail)
        return [x for x in out if x]


# ───────────────────────────
# Timeouts & execution
# ───────────────────────────

def _set_timeouts(conn: sa.Connection, lock_timeout_ms: int, statement_timeout_ms: int, in_txn: bool) -> None:
    """
    If inside a transaction, use SET LOCAL so values reset on COMMIT/ROLLBACK.
    Otherwise use session-level SET and explicitly reset afterwards.
    """
    if in_txn:
        conn.exec_driver_sql(f"SET LOCAL lock_timeout = '{int(lock_timeout_ms)}ms';")
        conn.exec_driver_sql(f"SET LOCAL statement_timeout = '{int(statement_timeout_ms)}ms';")
        conn.exec_driver_sql("SET LOCAL idle_in_transaction_session_timeout = '15000ms';")
    else:
        conn.exec_driver_sql(f"SET lock_timeout = '{int(lock_timeout_ms)}ms';")
        conn.exec_driver_sql(f"SET statement_timeout = '{int(statement_timeout_ms)}ms';")


def _reset_timeouts(conn: sa.Connection) -> None:
    conn.exec_driver_sql("RESET lock_timeout;")
    conn.exec_driver_sql("RESET statement_timeout;")


def run_one_statement(
    engine: sa.Engine,
    sql: str,
    lock_timeout_ms: int,
    statement_timeout_ms: int,
    max_rows: Optional[int],
    explain: bool,
    in_txn: bool,
) -> Tuple[str, object, float]:
    """
    Returns: (kind, payload, elapsed_sec)
      kind="result" -> payload is DataFrame (or list of EXPLAIN rows)
      kind="ok"     -> payload is status string (rowcount)
    """
    to_exec = f"EXPLAIN (ANALYZE, BUFFERS, VERBOSE) {sql}" if explain else sql
    start = time.perf_counter()

    if in_txn:
        # Use a single transaction context supplied by the caller.
        # Here we assume we're already in a transaction; just run the statement.
        with engine.connect() as conn:
            with conn.begin():  # explicit txn
                _set_timeouts(conn, lock_timeout_ms, statement_timeout_ms, in_txn=True)
                result = conn.exec_driver_sql(to_exec)
                payload, kind = _consume_result(result, max_rows)
    else:
        # Per-statement transaction to mimic "autocommit per statement" semantics.
        with engine.connect() as conn:
            with conn.begin():
                _set_timeouts(conn, lock_timeout_ms, statement_timeout_ms, in_txn=True)
                result = conn.exec_driver_sql(to_exec)
                payload, kind = _consume_result(result, max_rows)
            # Timeouts were LOCAL; automatically reset on commit.

    elapsed = time.perf_counter() - start
    return (kind, payload, elapsed)


def _consume_result(result: sa.CursorResult, max_rows: Optional[int]) -> Tuple[object, str]:
    if result.returns_rows:
        rows = result.fetchall()
        cols = result.keys()
        if max_rows is not None and max_rows > 0 and len(rows) > max_rows:
            rows = rows[:max_rows]
        # EXPLAIN typically returns one column "QUERY PLAN"; keep DataFrame for consistency
        df = pd.DataFrame(rows, columns=cols)
        return df, "result"
    else:
        rc = result.rowcount
        msg = "Command executed." if (rc in (-1, None)) else f"Command executed • rowcount={rc}"
        return msg, "ok"


# ───────────────────────────
# UI – Schema overview
# ───────────────────────────

st.subheader("Schema overview")

try:
    schemas = get_available_schemas(_db_key)
    schema = st.selectbox(
        "Schema",
        schemas or ["public"],
        index=(schemas or ["public"]).index("public") if "public" in (schemas or []) else 0,
    )

    schema_rows = get_schema_columns(_db_key, schema)
    if not schema_rows.empty:
        by_table: dict[str, list[str]] = {}
        for _, r in schema_rows.iterrows():
            t, c, d = r["table_name"], r["column_name"], r["data_type"]
            by_table.setdefault(t, []).append(f"{c} ({d})")
        for t, cols in by_table.items():
            st.markdown(f"**{t}**")
            st.write(", ".join(cols))
    else:
        st.info("No tables found in this schema.")
except Exception as e:
    if _is_conn_slots_exhausted(e):
        st.error("Database is out of available connection slots (code 53300).")
        with st.expander("What you can do", expanded=True):
            st.markdown(
                "- Close extra app tabs/sessions, then click **Disconnect / reset DB** in the sidebar.\n"
                "- Keep this app’s pool small (already limited to 1 connection).\n"
                "- Consider adding PgBouncer or increasing your Cloud SQL instance size if this recurs."
            )
    else:
        st.error(_friendly_error(e))
    st.stop()

st.caption(
    "Tip: Use single-transaction mode to run several statements atomically, "
    "or per-statement transactions to run each statement independently."
)

# ───────────────────────────
# UI – SQL Runner
# ───────────────────────────

st.subheader("Run arbitrary SQL")

with st.expander("Execution settings", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        single_txn = st.checkbox(
            "Run all statements in a single transaction (rollback on first error)",
            value=False,
            help="If unchecked, each statement runs in its own short transaction.",
        )
    with col2:
        lock_timeout_ms = st.number_input(
            "Lock timeout (ms)", min_value=0, max_value=60000, value=2000, step=250,
            help="Fail fast if another session holds a conflicting lock.",
        )
    with col3:
        statement_timeout_ms = st.number_input(
            "Statement timeout (ms)", min_value=0, max_value=10_000_000, value=30_000, step=500,
            help="Abort statements that run too long.",
        )
    col4, col5 = st.columns(2)
    with col4:
        max_rows = st.number_input(
            "Max rows to display", min_value=100, max_value=1_000_000, value=5000, step=100,
            help="Limits result-set size shown in the UI.",
        )
    with col5:
        explain = st.checkbox(
            "EXPLAIN ANALYZE (executes the query)",
            value=False,
            help="Prepends EXPLAIN (ANALYZE, BUFFERS, VERBOSE). Note: EXPLAIN ANALYZE actually runs the query.",
        )

sql_txt = st.text_area(
    "SQL to execute inside this DB (multiple statements supported)",
    height=220,
    placeholder=(
        "Paste one or more SQL statements separated by semicolons…\n\n"
        "Example:\n"
        "CREATE TABLE public.shelf_map_locations (\n"
        "  locid varchar NOT NULL,\n"
        "  label varchar,\n"
        "  x_pct numeric,\n"
        "  y_pct numeric,\n"
        "  w_pct numeric,\n"
        "  h_pct numeric,\n"
        "  rotation_deg real,\n"
        "  PRIMARY KEY (locid)\n"
        ");\n"
        "SELECT * FROM information_schema.tables WHERE table_schema='public';"
    ),
)

if st.button("Run SQL", type="primary"):
    stmts = split_sql_statements(sql_txt or "")
    if not stmts:
        st.warning("No SQL statements detected.")
        st.stop()

    engine = get_db().engine

    try:
        if single_txn:
            # Run all in one transaction; on first error, rollback and stop.
            with engine.connect() as conn:
                with conn.begin():
                    for idx, stmt in enumerate(stmts, start=1):
                        with st.container(border=True):
                            st.markdown(f"**Statement {idx}**")
                            st.code(stmt, language="sql")
                            _set_timeouts(conn, lock_timeout_ms, statement_timeout_ms, in_txn=True)
                            try:
                                result = conn.exec_driver_sql(
                                    f"EXPLAIN (ANALYZE, BUFFERS, VERBOSE) {stmt}" if explain else stmt
                                )
                                payload, kind = _consume_result(result, int(max_rows))
                            except Exception as e:
                                st.error(_friendly_error(e))
                                st.warning("Transaction rolled back due to the error above.")
                                raise  # triggers rollback

                            if kind == "result":
                                st.dataframe(payload, use_container_width=True)
                                st.caption(f"Returned {len(payload)} row(s)")
                            else:
                                st.success(payload)
        else:
            # Each statement gets its own short transaction (mimics autocommit per statement).
            for idx, stmt in enumerate(stmts, start=1):
                with st.container(border=True):
                    st.markdown(f"**Statement {idx}**")
                    st.code(stmt, language="sql")
                    try:
                        kind, payload, _elapsed = run_one_statement(
                            engine,
                            stmt,
                            int(lock_timeout_ms),
                            int(statement_timeout_ms),
                            int(max_rows),
                            bool(explain),
                            in_txn=False,
                        )
                    except Exception as e:
                        st.error(_friendly_error(e))
                        continue

                    if kind == "result":
                        st.dataframe(payload, use_container_width=True)
                        st.caption(f"Returned {len(payload)} row(s)")
                    else:
                        st.success(payload)
    except Exception as outer:
        # Handle batch-level connection exhaustion or other fatal errors once.
        if _is_conn_slots_exhausted(outer):
            st.error("Database is out of available connection slots (code 53300).")
            with st.expander("What you can do", expanded=True):
                st.markdown(
                    "- Close extra app tabs/sessions, then click **Disconnect / reset DB** in the sidebar.\n"
                    "- This page keeps only one connection in its pool. If the error persists, other clients are likely consuming the slots.\n"
                    "- Consider PgBouncer or increasing `max_connections` by sizing up your instance."
                )
        else:
            st.error(_friendly_error(outer))
