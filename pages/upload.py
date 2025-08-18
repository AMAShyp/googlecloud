# pages/upload.py
import io
import math
import time
import json
import unicodedata
import pandas as pd
import streamlit as st
from db_handler import DatabaseManager

st.set_page_config(page_title="CSV → Table Upload", layout="wide")
st.title("⬆️ CSV → Table Upload (templates + chunked ingest + debug)")

db = DatabaseManager()

# ───────────────────────────────────────────────────────────────
# Debug logger kept in session
# ───────────────────────────────────────────────────────────────
if "upload_logs" not in st.session_state:
    st.session_state.upload_logs = []

def log(msg: str, data: dict | None = None):
    entry = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "msg": msg}
    if data is not None:
        entry["data"] = data
    st.session_state.upload_logs.append(entry)

def dump_logs_text() -> str:
    lines = []
    for e in st.session_state.upload_logs:
        if "data" in e:
            lines.append(f'{e["t"]} | {e["msg"]} | {json.dumps(e["data"], ensure_ascii=False)}')
        else:
            lines.append(f'{e["t"]} | {e["msg"]}')
    return "\n".join(lines)

# ───────────────────────────────────────────────────────────────
# Catalog helpers (safe for empty returns)
# ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=60)
def list_tables() -> pd.DataFrame:
    q = """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog','information_schema')
          AND table_type = 'BASE TABLE'
        ORDER BY table_schema, table_name
    """
    df = db.fetch_data(q)
    if df.empty:
        return pd.DataFrame(columns=["table_schema", "table_name"])
    return df

@st.cache_data(show_spinner=False, ttl=60)
def table_columns(schema: str, table: str) -> pd.DataFrame:
    q = """
        SELECT
            c.ordinal_position,
            c.column_name,
            c.data_type,
            c.is_nullable = 'YES' AS is_nullable,
            c.column_default,
            c.character_maximum_length,
            c.numeric_precision,
            c.numeric_scale
        FROM information_schema.columns c
        WHERE c.table_schema = %s AND c.table_name = %s
        ORDER BY c.ordinal_position
    """
    df = db.fetch_data(q, (schema, table))
    if df.empty:
        return pd.DataFrame(columns=[
            "ordinal_position", "column_name", "data_type", "is_nullable",
            "column_default", "character_maximum_length",
            "numeric_precision", "numeric_scale"
        ])
    return df

@st.cache_data(show_spinner=False, ttl=60)
def table_primary_keys(schema: str, table: str) -> list[str]:
    q = """
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        WHERE tc.constraint_type = 'PRIMARY KEY'
          AND tc.table_schema = %s
          AND tc.table_name = %s
        ORDER BY kcu.ordinal_position
    """
    df = db.fetch_data(q, (schema, table))
    if df.empty or "column_name" not in df.columns:
        return []
    return df["column_name"].tolist()

# ───────────────────────────────────────────────────────────────
# CSV + mapping helpers
# ───────────────────────────────────────────────────────────────
def normalize_name(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower().replace(" ", "_")

def automap(csv_cols, table_cols):
    norm_csv = {normalize_name(c): c for c in csv_cols}
    return {t: norm_csv.get(normalize_name(t)) for t in table_cols}

def csv_to_df(uploaded_file, delimiter, encoding, has_header, quotechar):
    raw = uploaded_file.read()
    opts = dict(sep=delimiter or ",", encoding=encoding or "utf-8", dtype=str,
                keep_default_na=False, na_values=[""], quotechar=quotechar or '"')
    if has_header:
        return pd.read_csv(io.BytesIO(raw), **opts)
    df = pd.read_csv(io.BytesIO(raw), header=None, **opts)
    df.columns = [f"column_{i+1}" for i in range(df.shape[1])]
    return df

def coerce_cell(val: str):
    if val is None:
        return None
    if isinstance(val, str):
        v = val.strip()
        if v == "":
            return None
        return v
    return val

def build_insert_sql(schema: str, table: str, cols: list[str], on_conflict_do_nothing: bool):
    cols_quoted = ', '.join([f'"{c}"' for c in cols])
    placeholders = ', '.join(['%s'] * len(cols))
    sql = f'INSERT INTO "{schema}"."{table}" ({cols_quoted}) VALUES ({placeholders})'
    if on_conflict_do_nothing:
        sql += " ON CONFLICT DO NOTHING"
    return sql

# Template + required inference
def example_for_type(row) -> str:
    dt = (row.get("data_type") or "").lower()
    if "int" in dt: return "123"
    if "numeric" in dt or "decimal" in dt: return "9.99"
    if "double" in dt or "real" in dt or "float" in dt: return "3.14"
    if "bool" in dt: return "true"
    if dt == "date": return "2025-01-01"
    if "timestamp" in dt: return "2025-01-01 12:34:56"
    if dt == "time": return "12:34:56"
    if "uuid" in dt: return "00000000-0000-0000-0000-000000000000"
    if "json" in dt: return '{"key":"value"}'
    if "char" in dt or "text" in dt: return "example"
    return "value"

def build_csv_template(cols_df: pd.DataFrame) -> pd.DataFrame:
    if cols_df.empty or "column_name" not in cols_df.columns:
        return pd.DataFrame([{"example_column": "value"}, {"example_column": "value"}])
    example = {}
    for _, r in cols_df.iterrows():
        c = r["column_name"]
        example[c] = example_for_type(r)
    return pd.DataFrame([example, example])

def required_columns(cols_df: pd.DataFrame, pks: list[str]) -> list[str]:
    req = []
    if cols_df.empty:
        return req
    for _, r in cols_df.iterrows():
        c = r["column_name"]
        not_null = not bool(r.get("is_nullable", False))
        has_default = str(r.get("column_default", "") or "") != ""
        if c in pks and not has_default:
            req.append(c)
        elif not_null and not has_default:
            req.append(c)
    return req

# ───────────────────────────────────────────────────────────────
# UI controls
# ───────────────────────────────────────────────────────────────
colA, colB, colC = st.columns([1.4, 1, 1])
with colA:
    delim = st.text_input("Delimiter", value=",", help="e.g. , ; | \\t")
    enc = st.text_input("Encoding", value="utf-8")
with colB:
    quotechar = st.text_input("Quote char", value='"')
    has_header = st.checkbox("CSV has header row", value=True)
with colC:
    default_chunk = 2000   # ~13 chunks for 25k rows → good balance
    chunk_size = st.number_input("Insert chunk size", min_value=200, max_value=20000,
                                 value=default_chunk, step=200,
                                 help="2,000 is a solid default for ~25k rows.")

tables = list_tables()
schemas = sorted(tables["table_schema"].unique().tolist())
schema = st.selectbox(
    "Schema",
    options=schemas,
    index=0 if "public" not in schemas else schemas.index("public"),
)
subset = tables[tables["table_schema"] == schema]
table = st.selectbox("Table", options=subset["table_name"].tolist() or ["— none —"])

with st.expander("🐞 Debug panel", expanded=False):
    st.write("Live upload logs will appear here.")
    st.code(dump_logs_text() or "No logs yet.", language="text")
    st.download_button("⬇️ Download logs", data=(dump_logs_text() or "No logs."),
                       file_name="upload_debug_logs.txt", mime="text/plain")

if not table or table == "— none —":
    st.info("Pick a schema and table to continue.")
    st.stop()

# ───────────────────────────────────────────────────────────────
# Metadata + template
# ───────────────────────────────────────────────────────────────
cols_df = table_columns(schema, table)
pks = table_primary_keys(schema, table)
req_cols = required_columns(cols_df, pks)
template_df = build_csv_template(cols_df)

st.subheader("📘 Data dictionary")
dd = cols_df.copy()
if not dd.empty:
    dd.insert(1, "is_primary_key", dd["column_name"].isin(pks))
st.dataframe(dd if not dd.empty else pd.DataFrame(
    [{"info": "No columns found (permissions or table missing?)"}]),
    use_container_width=True, hide_index=True
)

with st.expander("📄 CSV Template & Examples", expanded=True):
    st.markdown(
        f"""
**Prepare your CSV for `{schema}.{table}`**

- Columns can be in **any order** — you will map them before upload.
- **Required columns**: `{', '.join(req_cols) if req_cols else '— none —'}`.
- Optional columns can be blank (NULL).
- Formats: DATE `YYYY-MM-DD`, TIMESTAMP `YYYY-MM-DD HH:MM:SS`, BOOLEAN `true/false`, NUMERIC `9.99`.
        """
    )
    st.write("**Template preview (example values):**")
    st.dataframe(template_df, use_container_width=True)
    st.download_button("⬇️ Download CSV template",
                       data=template_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{schema}.{table}.template.csv", mime="text/csv")

# ───────────────────────────────────────────────────────────────
# Upload CSV
# ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Choose a CSV file to upload into this table", type=["csv"])
if not uploaded:
    st.info("Pick a CSV to continue.")
    st.stop()

try:
    df_csv = csv_to_df(uploaded, delimiter=delim.replace("\\t", "\t"),
                       encoding=enc, has_header=has_header, quotechar=quotechar)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    log("read_csv_failed", {"error": str(e)})
    st.stop()

st.write("**CSV Preview**")
st.dataframe(df_csv.head(50), use_container_width=True)

# ───────────────────────────────────────────────────────────────
# Column mapping
# ───────────────────────────────────────────────────────────────
st.subheader("Map CSV columns → table columns")
table_cols = cols_df["column_name"].tolist() if "column_name" in cols_df.columns else []
if not table_cols:
    st.error("This table has no columns or metadata couldn’t be read.")
    st.stop()

csv_cols = df_csv.columns.tolist()
default_map = automap(csv_cols, table_cols)

mapping = {}
m1, m2 = st.columns([2, 2])
with m1:
    st.caption("Table column → CSV column")
    for tcol in table_cols:
        default_choice = default_map.get(tcol)
        mapping[tcol] = st.selectbox(
            f'↪ {tcol}',
            options=["— skip —"] + csv_cols,
            index=(["— skip —"] + csv_cols).index(default_choice) if default_choice in csv_cols else 0,
            key=f"map_{tcol}",
        )
with m2:
    st.caption("Options")
    truncate = st.checkbox("TRUNCATE table before load (danger!)", value=False)
    on_conflict = st.checkbox("ON CONFLICT DO NOTHING (skip duplicates)", value=True)
    show_sql = st.checkbox("Show generated INSERT SQL", value=False)
    row_debug = st.checkbox("Row-by-row debug (first N rows)", value=False,
                            help="Executes the first N rows individually with RETURNING to show inserted/skipped. Slower, use for diagnosis.")
    row_debug_n = st.number_input("N", min_value=1, max_value=200, value=25, step=1, disabled=not row_debug)

# Effective columns + required check
target_cols = [c for c in table_cols if mapping.get(c) and mapping[c] != "— skip —"]
missing_required = [c for c in req_cols if c not in target_cols]
if missing_required:
    st.error(f"These required columns are not mapped: {', '.join(missing_required)}")
    st.stop()
if not target_cols:
    st.warning("No columns mapped. Map at least one table column to a CSV column.")
    st.stop()

# Build mapped DataFrame
mapped = pd.DataFrame()
try:
    for tcol in target_cols:
        mapped[tcol] = df_csv[mapping[tcol]].map(coerce_cell)
except KeyError as e:
    st.error(f"Mapping refers to a missing CSV column: {e}")
    log("mapping_missing_column", {"error": str(e)})
    st.stop()

st.write("**Mapped Preview**")
st.dataframe(mapped.head(50), use_container_width=True)
st.caption(f"{len(mapped):,} rows will be inserted into {schema}.{table} with columns {target_cols}.")

insert_sql = build_insert_sql(schema, table, target_cols, on_conflict_do_nothing=on_conflict)
if show_sql:
    st.code(insert_sql, language="sql")

# ───────────────────────────────────────────────────────────────
# Optional Row-by-row debug (first N)
# ───────────────────────────────────────────────────────────────
if row_debug:
    st.info("Row-by-row debug for the first N rows (using RETURNING).")
    sample = mapped.head(int(row_debug_n))
    results = []
    for idx, row in sample.iterrows():
        params = tuple(row[c] for c in target_cols)
        # Add RETURNING to detect actual insertion vs. conflict-skip
        sql = insert_sql + " RETURNING true"
        cur = db.conn.cursor()
        try:
            cur.execute("SET LOCAL statement_timeout = 15000;")
            cur.execute(sql, params)
            inserted = cur.fetchone() is not None
            db.conn.commit()
        except Exception as e:
            db.conn.rollback()
            inserted = False
            results.append({"row": int(idx + 1), "inserted": False, "error": str(e)})
        else:
            results.append({"row": int(idx + 1), "inserted": bool(inserted)})
        finally:
            cur.close()
    st.dataframe(pd.DataFrame(results), use_container_width=True)
    log("row_debug_results", {"results": results})

# ───────────────────────────────────────────────────────────────
# Execute chunked upload (fast path)
# ───────────────────────────────────────────────────────────────
go = st.button("🚀 Start upload")
if not go:
    st.stop()

if truncate:
    st.error("You chose to TRUNCATE the table before load. This will DELETE all existing rows.")
    if not st.checkbox("I understand, proceed with TRUNCATE"):
        st.stop()

total_rows = len(mapped)
chunks = int(math.ceil(total_rows / chunk_size)) if total_rows else 0
log("upload_start", {
    "schema": schema, "table": table,
    "rows": total_rows, "chunk_size": int(chunk_size),
    "chunks": int(chunks), "on_conflict_do_nothing": on_conflict,
    "truncate": truncate
})

# Optional truncate first
try:
    if truncate:
        db.execute_command(f'TRUNCATE TABLE "{schema}"."{table}" RESTART IDENTITY CASCADE;')
        log("truncate_done", {"schema": schema, "table": table})
except Exception as e:
    st.error(f"TRUNCATE failed: {e}")
    log("truncate_failed", {"error": str(e)})
    st.stop()

# If table has a single-column PK, we can estimate conflicts per chunk
single_pk = pks[0] if len(pks) == 1 else None

prog = st.progress(0.0)
status = st.empty()
inserted_total = 0
overall_start = time.time()

try:
    for i in range(chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)
        batch = mapped.iloc[start_idx:end_idx]

        # Build params using a robust method (prevents weird slicing bugs)
        params = list(batch[target_cols].itertuples(index=False, name=None))

        # Optional conflict estimate using single-column PK (fast)
        conflict_estimate = None
        if single_pk and single_pk in target_cols:
            # select existing keys among this batch
            keys = tuple(batch[single_pk].dropna().unique().tolist())
            if len(keys) > 0:
                placeholders = ", ".join(["%s"] * len(keys))
                q = f'SELECT "{single_pk}" FROM "{schema}"."{table}" WHERE "{single_pk}" IN ({placeholders})'
                try:
                    existing = db.fetch_data(q, keys)
                    conflict_estimate = int(existing.shape[0])
                except Exception as e:
                    conflict_estimate = None
                    log("pk_conflict_estimate_failed", {"chunk": i+1, "error": str(e)})

        cur = db.conn.cursor()
        t0 = time.time()
        try:
            cur.execute("SET LOCAL statement_timeout = 120000;")  # 120s per chunk
            cur.executemany(insert_sql, params)
            rowcount = cur.rowcount  # driver-reported; may be total or last stmt depending on driver
        except Exception as e:
            db.conn.rollback()
            cur.close()
            log("chunk_failed", {
                "chunk_index": i + 1,
                "rows_in_chunk": len(batch),
                "error": str(e),
                "range": [int(start_idx + 1), int(end_idx)]
            })
            st.error(f"Failed on chunk {i+1}/{chunks} (rows {start_idx+1}-{end_idx}). Error: {e}")
            st.stop()
        else:
            db.conn.commit()
            cur.close()
            elapsed = time.time() - t0
            rps = len(batch) / elapsed if elapsed > 0 else None
            inserted_total += len(batch)  # optimistic count (all attempted)
            log("chunk_ok", {
                "chunk_index": i + 1,
                "rows_in_chunk": len(batch),
                "elapsed_sec": round(elapsed, 3),
                "rows_per_sec": round(rps, 1) if rps else None,
                "rowcount_reported": rowcount,
                "pk_conflict_estimate": conflict_estimate,
                "range": [int(start_idx + 1), int(end_idx)]
            })

        prog.progress((i + 1) / max(chunks, 1))
        msg = (f"Chunk {i+1}/{chunks}: attempted {len(batch):,} rows "
               f"(total attempted {inserted_total:,}/{total_rows:,}).")
        if conflict_estimate is not None:
            msg += f" ~{conflict_estimate:,} existing (by PK)."
        status.info(msg)

    total_elapsed = time.time() - overall_start
    rps_total = inserted_total / total_elapsed if total_elapsed > 0 else None
    log("upload_done", {
        "attempted_total": inserted_total,
        "total_rows": total_rows,
        "elapsed_sec": round(total_elapsed, 2),
        "rows_per_sec": round(rps_total, 1) if rps_total else None
    })

    prog.progress(1.0)
    st.success(f"Done! Attempted {inserted_total:,} rows into {schema}.{table} "
               f"in {total_elapsed:.2f}s "
               f"({(rps_total or 0):.0f} rows/sec). "
               f"See debug panel for per-chunk details.")
    st.balloons()

except Exception as e:
    log("upload_exception", {"error": str(e)})
    st.error(f"Upload failed: {e}")

# Refresh debug panel
with st.expander("🐞 Debug panel", expanded=False):
    st.code(dump_logs_text() or "No logs yet.", language="text")
    st.download_button("⬇️ Download logs", data=(dump_logs_text() or "No logs."),
                       file_name="upload_debug_logs.txt", mime="text/plain")
