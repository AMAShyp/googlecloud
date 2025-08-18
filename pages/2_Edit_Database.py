# pages/2_Edit_Database.py
import re
import json
import time
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

from db_handler import DatabaseManager

st.set_page_config(page_title="Edit / Inspect Database", layout="wide")
st.title("🛠️ Edit / Inspect Database")

db = DatabaseManager()

# ───────────────────────────────────────────────────────────────
# Helpers: catalog + quoting
# ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=60)
def list_schemas() -> List[str]:
    q = """
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name NOT IN ('pg_catalog','information_schema')
        ORDER BY schema_name;
    """
    df = db.fetch_data(q)
    return df["schema_name"].tolist() if not df.empty else ["public"]

@st.cache_data(show_spinner=False, ttl=60)
def list_tables(schema: str) -> pd.DataFrame:
    q = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = %s AND table_type='BASE TABLE'
        ORDER BY table_name;
    """
    return db.fetch_data(q, (schema,))

@st.cache_data(show_spinner=False, ttl=60)
def list_columns(schema: str) -> pd.DataFrame:
    q = """
        SELECT table_name, column_name, data_type, is_nullable, column_default
        FROM   information_schema.columns
        WHERE  table_schema = %s
        ORDER  BY table_name, ordinal_position;
    """
    return db.fetch_data(q, (schema,))

def qident(name: str) -> str:
    """Quote an identifier safely."""
    return '"' + name.replace('"', '""') + '"'

def is_valid_ident(name: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name))

def sql_literal(val: Any) -> str:
    """SQL literal for default values when not using 'expression' mode."""
    if val is None:
        return "NULL"
    if isinstance(val, (int, float)):
        return str(val)
    # string → single-quote, escape single quotes
    return "'" + str(val).replace("'", "''") + "'"

# ───────────────────────────────────────────────────────────────
# Inspect schema (left)
# ───────────────────────────────────────────────────────────────
st.subheader("📚 Schema overview")

schemas = list_schemas()
schema = st.selectbox(
    "Schema",
    options=schemas,
    index=(schemas.index("public") if "public" in schemas else 0),
)

cols_df = list_columns(schema)
if cols_df.empty:
    st.info("No tables found in this schema.")
else:
    with st.expander("Tables and columns", expanded=True):
        by_table: Dict[str, pd.DataFrame] = {}
        for tname, group in cols_df.groupby("table_name", sort=True):
            by_table[tname] = group[["column_name", "data_type", "is_nullable", "column_default"]]
        for tname, g in by_table.items():
            st.markdown(f"**{tname}**")
            st.dataframe(g.reset_index(drop=True), use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────────────────────
# Table builder (right)
# ───────────────────────────────────────────────────────────────
st.subheader("🧱 Create a new table")

table_name = st.text_input("Table name", placeholder="e.g., products")
if table_name and not is_valid_ident(table_name):
    st.error("Table name must match ^[A-Za-z_][A-Za-z0-9_]*$")

# Basic recommended Postgres types
PG_TYPES = [
    # numeric
    "smallint", "integer", "bigint", "decimal", "numeric", "real", "double precision",
    "serial", "bigserial",
    # string
    "text", "varchar(255)", "varchar(100)", "varchar(50)", "char(10)",
    # date/time
    "date", "timestamp", "timestamp with time zone", "time", "time with time zone",
    # bool/json/etc
    "boolean", "uuid", "jsonb",
]

st.caption("Define columns (add/remove rows, mark PKs/Unique, set defaults).")
default_rows = [
    {"name": "id", "type": "bigint",  "nullable": False, "primary_key": True,  "unique": False, "default": "", "default_is_expr": False},
    {"name": "created_at", "type": "timestamp", "nullable": False, "primary_key": False, "unique": False, "default": "now()", "default_is_expr": True},
    {"name": "name", "type": "text", "nullable": False, "primary_key": False, "unique": False, "default": "", "default_is_expr": False},
]

# Data editor for columns
col_config = {
    "name": {"editable": True},
    "type": st.column_config.SelectboxColumn("type", options=PG_TYPES, width="medium"),
    "nullable": st.column_config.CheckboxColumn("nullable"),
    "primary_key": st.column_config.CheckboxColumn("primary_key"),
    "unique": st.column_config.CheckboxColumn("unique"),
    "default": st.column_config.TextColumn("default (literal or expression)"),
    "default_is_expr": st.column_config.CheckboxColumn("default_is_expr", help="If checked, 'default' will NOT be quoted (treated as SQL expression)."),
}
columns_df = st.data_editor(
    pd.DataFrame(default_rows),
    num_rows="dynamic",
    use_container_width=True,
    column_config=col_config,
    key="table_columns_editor",
)

# Additional options: composite unique constraints, indexes, foreign keys
st.markdown("### 🔧 Constraints & Indexes (optional)")
c1, c2 = st.columns(2)

with c1:
    unique_groups_raw = st.text_area(
        "Composite UNIQUE constraints (one per line, comma-separated column names)",
        placeholder="e.g.\nname, department_id\nbarcode",
        height=100,
    )
with c2:
    indexes_raw = st.text_area(
        "Indexes (B-Tree). One per line: index_name: col1,col2",
        placeholder="e.g.\nidx_products_name: name\nidx_products_dept_created: department_id, created_at",
        height=100,
    )

st.markdown("### 🔗 Foreign keys (optional)")
fk_rows_default = pd.DataFrame([{
    "constraint_name": "",
    "columns": "",
    "ref_table": "",
    "ref_columns": "",
    "on_delete": "NO ACTION",
    "on_update": "NO ACTION",
}])
fk_editor = st.data_editor(
    fk_rows_default,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "constraint_name": st.column_config.TextColumn("constraint_name", help="Optional; leave empty for auto-generated."),
        "columns": st.column_config.TextColumn("columns", help="Comma-separated local column(s)"),
        "ref_table": st.column_config.TextColumn("ref_table", help="Fully qualified or same schema (e.g., categories)"),
        "ref_columns": st.column_config.TextColumn("ref_columns", help="Comma-separated referenced column(s)"),
        "on_delete": st.column_config.SelectboxColumn("on_delete", options=["NO ACTION","RESTRICT","CASCADE","SET NULL","SET DEFAULT"]),
        "on_update": st.column_config.SelectboxColumn("on_update", options=["NO ACTION","RESTRICT","CASCADE","SET NULL","SET DEFAULT"]),
    },
    key="fk_editor",
)

# ───────────────────────────────────────────────────────────────
# Build CREATE TABLE SQL
# ───────────────────────────────────────────────────────────────
def build_create_table_sql(schema: str, table: str, cols_df: pd.DataFrame,
                           unique_groups_txt: str, indexes_txt: str, fk_df: pd.DataFrame) -> Dict[str, Any]:
    errors = []
    warnings = []

    # validate table name
    if not table:
        errors.append("Table name is required.")
    elif not is_valid_ident(table):
        errors.append("Invalid table name (must match ^[A-Za-z_][A-Za-z0-9_]*$).")

    # clean columns
    cols_df = cols_df.fillna({"name": "", "type": "", "default": ""})
    cols = []
    col_names_seen = set()
    pk_cols = []
    uniques_inline = []

    for i, row in cols_df.iterrows():
        name = str(row["name"]).strip()
        typ  = str(row["type"]).strip().lower()
        nullable = bool(row["nullable"])
        pk = bool(row["primary_key"])
        uniq = bool(row["unique"])
        default = str(row["default"]).strip()
        default_is_expr = bool(row.get("default_is_expr", False))

        if not name:
            continue
        if not is_valid_ident(name):
            errors.append(f"Invalid column name: {name}")
            continue
        if not typ:
            errors.append(f"Missing type for column: {name}")
            continue
        if name in col_names_seen:
            errors.append(f"Duplicate column name: {name}")
            continue
        col_names_seen.add(name)

        col_parts = [qident(name), typ]
        if default:
            # expression vs literal
            if default_is_expr:
                col_parts.append(f"DEFAULT {default}")
            else:
                col_parts.append(f"DEFAULT {sql_literal(default)}")
        if pk:
            pk_cols.append(name)
        if uniq:
            uniques_inline.append([name])

        if not nullable and not pk:
            # PK implies NOT NULL implicitly in PG; no need to add again
            col_parts.append("NOT NULL")

        cols.append(" ".join(col_parts))

    if not cols:
        errors.append("At least one column is required.")

    # primary key
    constraints = []
    if pk_cols:
        constraints.append(f"PRIMARY KEY ({', '.join(qident(c) for c in pk_cols)})")

    # composite UNIQUE groups
    unique_groups = []
    for line in (unique_groups_txt or "").splitlines():
        line = line.strip()
        if not line:
            continue
        cols_grp = [c.strip() for c in line.split(",") if c.strip()]
        if not cols_grp:
            continue
        # validate identifiers
        bad = [c for c in cols_grp if not is_valid_ident(c)]
        if bad:
            errors.append(f"Invalid column name(s) in UNIQUE: {', '.join(bad)}")
            continue
        unique_groups.append(cols_grp)

    # Inline uniques (single columns)
    for u in uniques_inline:
        if u not in unique_groups:
            unique_groups.append(u)

    for grp in unique_groups:
        constraints.append(f"UNIQUE ({', '.join(qident(c) for c in grp)})")

    # foreign keys
    fk_clauses = []
    for _, r in fk_df.fillna("").iterrows():
        cols_local = [c.strip() for c in str(r["columns"]).split(",") if c.strip()]
        ref_tbl = str(r["ref_table"]).strip()
        ref_cols = [c.strip() for c in str(r["ref_columns"]).split(",") if c.strip()]
        if not cols_local and not ref_tbl:
            continue  # empty row
        if not cols_local or not ref_tbl or not ref_cols:
            errors.append("Foreign key row incomplete (columns, ref_table, ref_columns are required).")
            continue
        # constraint name optional
        cname = str(r.get("constraint_name", "") or "").strip()
        od = str(r.get("on_delete", "NO ACTION"))
        ou = str(r.get("on_update", "NO ACTION"))

        # quote ref table if not schema-qualified
        if "." in ref_tbl:
            ref_tbl_q = ".".join(qident(p) for p in ref_tbl.split(".", 1))
        else:
            ref_tbl_q = f"{qident(schema)}.{qident(ref_tbl)}"

        fk = (f"CONSTRAINT {qident(cname)} " if cname else "") + \
             f"FOREIGN KEY ({', '.join(qident(c) for c in cols_local)}) " \
             f"REFERENCES {ref_tbl_q} ({', '.join(qident(c) for c in ref_cols)}) " \
             f"ON DELETE {od} ON UPDATE {ou}"
        fk_clauses.append(fk)

    all_defs = cols + constraints + fk_clauses
    if not all_defs and not errors:
        errors.append("Nothing to create. Add columns or constraints.")

    create_sql = f"CREATE TABLE {qident(schema)}.{qident(table)} (\n  " + ",\n  ".join(all_defs) + "\n);"

    # indexes (separate DDLs)
    index_ddls: List[str] = []
    for line in (indexes_txt or "").splitlines():
        line = line.strip()
        if not line:
            continue
        # format: idx_name: col1,col2
        if ":" not in line:
            warnings.append(f"Index line ignored (missing ':'): {line}")
            continue
        idx_name, cols_str = [p.strip() for p in line.split(":", 1)]
        if not is_valid_ident(idx_name):
            errors.append(f"Invalid index name: {idx_name}")
            continue
        idx_cols = [c.strip() for c in cols_str.split(",") if c.strip()]
        bad = [c for c in idx_cols if not is_valid_ident(c)]
        if bad:
            errors.append(f"Invalid column(s) in index {idx_name}: {', '.join(bad)}")
            continue
        idx_sql = f"CREATE INDEX {qident(idx_name)} ON {qident(schema)}.{qident(table)} " \
                  f"USING btree ({', '.join(qident(c) for c in idx_cols)});"
        index_ddls.append(idx_sql)

    return {"errors": errors, "warnings": warnings, "create_sql": create_sql, "index_ddls": index_ddls}

# ───────────────────────────────────────────────────────────────
# Build + preview
# ───────────────────────────────────────────────────────────────
build = st.button("🧱 Build CREATE TABLE SQL")
result = None
if build:
    result = build_create_table_sql(schema, table_name.strip(), columns_df, unique_groups_raw, indexes_raw, fk_editor)

    if result["errors"]:
        st.error("Please fix the following issues before running:")
        for e in result["errors"]:
            st.write(f"- {e}")
    else:
        if result["warnings"]:
            st.warning("Warnings:")
            for w in result["warnings"]:
                st.write(f"- {w}")

        st.markdown("#### Preview DDL")
        st.code(result["create_sql"], language="sql")

        if result["index_ddls"]:
            with st.expander("Additional index DDLs", expanded=False):
                for ddl in result["index_ddls"]:
                    st.code(ddl, language="sql")

# ───────────────────────────────────────────────────────────────
# Execute
# ───────────────────────────────────────────────────────────────
if result and not result["errors"]:
    st.markdown("---")
    st.subheader("Execute")

    colx, coly = st.columns([1, 2])
    with colx:
        single_txn = st.checkbox("Single transaction (recommended)", value=True,
                                 help="CREATE TABLE + indexes in one transaction.")
        lock_timeout_ms = st.number_input("Lock timeout (ms)", min_value=0, max_value=60000, value=2000, step=250)
        stmt_timeout_ms = st.number_input("Statement timeout (ms)", min_value=0, max_value=10_000_000, value=30000, step=500)
    with coly:
        st.info("You can re-run safely: if the table already exists, rerun will fail unless you drop it first.")

    run = st.button("🚀 Run DDL now")
    if run:
        start = time.perf_counter()
        try:
            cur = db.conn.cursor()
            try:
                if single_txn:
                    cur.execute("BEGIN;")
                    cur.execute(f"SET LOCAL lock_timeout = '{int(lock_timeout_ms)}ms';")
                    cur.execute(f"SET LOCAL statement_timeout = '{int(stmt_timeout_ms)}ms';")
                else:
                    cur.execute(f"SET lock_timeout = '{int(lock_timeout_ms)}ms';")
                    cur.execute(f"SET statement_timeout = '{int(stmt_timeout_ms)}ms';")

                # CREATE TABLE
                cur.execute(result["create_sql"])

                # Indexes
                for ddl in result["index_ddls"]:
                    cur.execute(ddl)

                if single_txn:
                    db.conn.commit()
                else:
                    # If autocommit-like, still ensure commit
                    db.conn.commit()

                elapsed = (time.perf_counter() - start) * 1000
                st.success(f"Done. DDL executed in {elapsed:.0f} ms.")
            except Exception as e:
                db.conn.rollback()
                raise
            finally:
                cur.close()
        except Exception as e:
            st.error(f"Execution failed: {e}")

# ───────────────────────────────────────────────────────────────
# Quick viewer for one table (optional)
# ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔎 Quick table viewer")

tbls_df = list_tables(schema)
table_pick = st.selectbox("Pick a table to preview", options=tbls_df["table_name"].tolist() if not tbls_df.empty else [])
limit = st.number_input("Rows to show", min_value=10, max_value=10000, value=50, step=10)

if table_pick:
    try:
        df = db.fetch_data(f'SELECT * FROM {qident(schema)}.{qident(table_pick)} LIMIT %s;', (int(limit),))
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
