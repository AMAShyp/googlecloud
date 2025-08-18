# pages/upload.py
import io
import time
import json
import math
import uuid
import unicodedata
import pandas as pd
import streamlit as st

# Your connector-based db manager (for catalog queries)
from db_handler import DatabaseManager

# For COPY we use psycopg2 through the Cloud SQL Connector (separate path)
from google.cloud.sql.connector import Connector
from google.oauth2 import service_account
import psycopg2

st.set_page_config(page_title="CSV → Table Upload (COPY)", layout="wide")
st.title("⬆️ CSV → Table Upload (staging + COPY)")

###############################################################################
# 0) Build a psycopg2 connection via Cloud SQL Connector (separate from pg8000)
###############################################################################
def open_psycopg2_conn():
    # Load secrets / env consistent with your db_handler
    cfg = {
        "instance_connection_name": st.secrets["cloudsql"]["instance_connection_name"],
        "user": st.secrets["cloudsql"]["user"],
        "password": st.secrets["cloudsql"]["password"],
        "db": st.secrets["cloudsql"]["db"],
    }
    creds = None
    if "gcp_service_account" in st.secrets:
        creds = service_account.Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"])
        )
    connector = Connector(credentials=creds) if creds else Connector()
    # IMPORTANT: use driver "psycopg2"
    conn = connector.connect(
        cfg["instance_connection_name"],
        "psycopg2",
        user=cfg["user"],
        password=cfg["password"],
        db=cfg["db"],
        connect_timeout=10,
    )
    # Keep handles for cleanup
    conn._cloudsql_connector = connector  # noqa: SLF001
    return conn

def close_psycopg2_conn(conn):
    try:
        conn.close()
    except Exception:
        pass
    try:
        connector = getattr(conn, "_cloudsql_connector", None)
        if connector:
            connector.close()
    except Exception:
        pass

###############################################################################
# 1) Small logging helpers
###############################################################################
if "upload_logs" not in st.session_state:
    st.session_state.upload_logs = []

def log(msg, data=None):
    st.session_state.upload_logs.append({
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "msg": msg,
        **({"data": data} if data is not None else {})
    })

def dump_logs_text():
    lines = []
    for e in st.session_state.upload_logs:
        d = e.get("data")
        if d is None:
            lines.append(f'{e["t"]} | {e["msg"]}')
        else:
            lines.append(f'{e["t"]} | {e["msg"]} | {json.dumps(d, ensure_ascii=False)}')
    return "\n".join(lines)

###############################################################################
# 2) Catalog helpers (using your pg8000-backed DatabaseManager)
###############################################################################
db = DatabaseManager()

@st.cache_data(show_spinner=False, ttl=60)
def list_tables():
    q = """
      SELECT table_schema, table_name
      FROM information_schema.tables
      WHERE table_schema NOT IN ('pg_catalog','information_schema')
        AND table_type='BASE TABLE'
      ORDER BY table_schema, table_name
    """
    df = db.fetch_data(q)
    return df if not df.empty else pd.DataFrame(columns=["table_schema","table_name"])

@st.cache_data(show_spinner=False, ttl=60)
def table_columns(schema, table):
    q = """
      SELECT
        c.ordinal_position,
        c.column_name,
        c.data_type,
        c.is_nullable = 'YES' AS is_nullable,
        c.column_default
      FROM information_schema.columns c
      WHERE c.table_schema=%s AND c.table_name=%s
      ORDER BY c.ordinal_position
    """
    df = db.fetch_data(q, (schema, table))
    if df.empty:
        return pd.DataFrame(columns=["ordinal_position","column_name","data_type","is_nullable","column_default"])
    return df

@st.cache_data(show_spinner=False, ttl=60)
def primary_keys(schema, table):
    q = """
      SELECT kcu.column_name
      FROM information_schema.table_constraints tc
      JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name=kcu.constraint_name
       AND tc.table_schema=kcu.table_schema
      WHERE tc.constraint_type='PRIMARY KEY'
        AND tc.table_schema=%s AND tc.table_name=%s
      ORDER BY kcu.ordinal_position
    """
    df = db.fetch_data(q, (schema, table))
    return df["column_name"].tolist() if not df.empty and "column_name" in df.columns else []

###############################################################################
# 3) CSV + mapping helpers
###############################################################################
def normalize(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    return s.lower().replace(" ","_")

def automap(csv_cols, table_cols):
    norm = {normalize(c): c for c in csv_cols}
    return {t: norm.get(normalize(t)) for t in table_cols}

def read_csv(uploaded, delimiter, encoding, has_header, quotechar):
    raw = uploaded.read()
    opts = dict(sep=delimiter, encoding=encoding, dtype=str, keep_default_na=False, na_values=[""], quotechar=quotechar)
    if has_header:
        return pd.read_csv(io.BytesIO(raw), **opts)
    df = pd.read_csv(io.BytesIO(raw), header=None, **opts)
    df.columns = [f"column_{i+1}" for i in range(df.shape[1])]
    return df

def coerce_cell(v):
    if v is None: return None
    if isinstance(v,str):
        v = v.strip()
        return None if v=="" else v
    return v

def required_columns(cols_df, pks):
    req=[]
    for _,r in cols_df.iterrows():
        c=r["column_name"]; not_null=not bool(r["is_nullable"]); has_def=str(r.get("column_default") or "")!=""
        if c in pks and not has_def: req.append(c)
        elif not_null and not has_def: req.append(c)
    return req

def example_for_type(dt: str) -> str:
    dt=(dt or "").lower()
    if "int" in dt: return "123"
    if "numeric" in dt or "decimal" in dt: return "9.99"
    if "double" in dt or "real" in dt or "float" in dt: return "3.14"
    if "bool" in dt: return "true"
    if dt=="date": return "2025-01-01"
    if "timestamp" in dt: return "2025-01-01 12:34:56"
    if dt=="time": return "12:34:56"
    if "uuid" in dt: return "00000000-0000-0000-0000-000000000000"
    if "json" in dt: return '{"key":"value"}'
    if "char" in dt or "text" in dt: return "example"
    return "value"

def csv_template(cols_df):
    if cols_df.empty: return pd.DataFrame([{"example_column":"value"},{"example_column":"value"}])
    row={}
    for _,r in cols_df.iterrows():
        row[r["column_name"]] = example_for_type(r["data_type"])
    return pd.DataFrame([row,row])

###############################################################################
# 4) UI controls
###############################################################################
c1,c2,c3 = st.columns([1.5,1,1])
with c1:
    delim = st.text_input("Delimiter", value=",", help="e.g. , ; | \\t").replace("\\t","\t")
    enc = st.text_input("Encoding", value="utf-8")
with c2:
    quotechar = st.text_input("Quote char", value='"')
    has_header = st.checkbox("CSV has header row", value=True)
with c3:
    chunk_rows = st.number_input("COPY batch rows (lines per stream)", min_value=500, max_value=100000, value=20000, step=500,
                                 help="COPY streams the whole CSV; this value only affects progress jumps. 20,000 works well.")

tables = list_tables()
schemas = sorted(tables["table_schema"].unique().tolist())
schema = st.selectbox("Schema", options=schemas, index=schemas.index("public") if "public" in schemas else 0)
t_subset = tables[tables["table_schema"]==schema]
table = st.selectbox("Table", options=t_subset["table_name"].tolist() or ["— none —"])

with st.expander("🐞 Debug panel", expanded=False):
    st.code(dump_logs_text() or "No logs yet.", language="text")
    st.download_button("⬇️ Download logs", data=(dump_logs_text() or "No logs."),
                       file_name="upload_debug_logs.txt", mime="text/plain")

if not table or table=="— none —":
    st.info("Pick a schema & table.")
    st.stop()

cols_df = table_columns(schema, table)
pks = primary_keys(schema, table)
req_cols = required_columns(cols_df, pks)
tmpl = csv_template(cols_df)

st.subheader("📘 Data dictionary")
dd = cols_df.copy()
if not dd.empty:
    dd.insert(1,"is_primary_key", dd["column_name"].isin(pks))
st.dataframe(dd if not dd.empty else pd.DataFrame([{"info":"No columns found."}]), use_container_width=True, hide_index=True)

with st.expander("📄 CSV Template & Examples", expanded=True):
    st.write("**Template preview (example values):**")
    st.dataframe(tmpl, use_container_width=True)
    st.download_button("⬇️ Download CSV template",
                       data=tmpl.to_csv(index=False).encode("utf-8"),
                       file_name=f"{schema}.{table}.template.csv", mime="text/csv")

uploaded = st.file_uploader("Choose CSV", type=["csv"])
if not uploaded:
    st.info("Upload a CSV to continue.")
    st.stop()

try:
    df_csv = read_csv(uploaded, delim, enc, has_header, quotechar)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    log("read_csv_failed", {"error": str(e)})
    st.stop()

st.write("**CSV Preview**")
st.dataframe(df_csv.head(50), use_container_width=True)

# Mapping UI
st.subheader("Map CSV columns → table columns")
table_cols = cols_df["column_name"].tolist()
default_map = automap(df_csv.columns.tolist(), table_cols)

mapping={}
m1,m2 = st.columns([2,2])
with m1:
    st.caption("Table column → CSV column")
    for tcol in table_cols:
        default_choice = default_map.get(tcol)
        mapping[tcol] = st.selectbox(
            f'↪ {tcol}',
            options=["— skip —"] + df_csv.columns.tolist(),
            index=(["— skip —"] + df_csv.columns.tolist()).index(default_choice) if default_choice in df_csv.columns else 0,
            key=f"map_{tcol}"
        )
with m2:
    truncate = st.checkbox("TRUNCATE table before load (danger!)", value=False)
    on_conflict = st.checkbox("ON CONFLICT DO NOTHING (skip duplicates)", value=True)
    show_sql = st.checkbox("Show INSERT SQL", value=False)

target_cols = [c for c in table_cols if mapping.get(c) and mapping[c]!="— skip —"]
missing_req = [c for c in req_cols if c not in target_cols]
if missing_req:
    st.error(f"Required columns not mapped: {', '.join(missing_req)}")
    st.stop()
if not target_cols:
    st.warning("Map at least one column.")
    st.stop()

# Build a mapped DataFrame in the target column order
mapped = pd.DataFrame()
for tcol in target_cols:
    mapped[tcol] = df_csv[mapping[tcol]].map(coerce_cell)

st.write("**Mapped Preview**")
st.dataframe(mapped.head(50), use_container_width=True)
st.caption(f"{len(mapped):,} rows → {schema}.{table} columns {target_cols}")

# INSERT SQL (staging -> target)
cols_quoted = ", ".join([f'"{c}"' for c in target_cols])
insert_sql = f'INSERT INTO "{schema}"."{table}" ({cols_quoted}) SELECT {cols_quoted} FROM {{staging}}'
if on_conflict:
    insert_sql += " ON CONFLICT DO NOTHING"
if show_sql:
    st.code(insert_sql.replace("{staging}", "<staging_table>"), language="sql")

go = st.button("🚀 Start upload (COPY)")
if not go:
    st.stop()

if truncate:
    st.error("You chose TRUNCATE (will delete ALL rows first).")
    if not st.checkbox("I understand, proceed with TRUNCATE"):
        st.stop()

###############################################################################
# 5) BULK LOAD VIA STAGING + COPY
###############################################################################
start_total = time.time()
staging = f'__staging_{table}_{uuid.uuid4().hex[:8]}'
log("upload_start", {"table": f"{schema}.{table}", "staging": staging, "rows": len(mapped)})

# Generate a CSV string from the mapped DF (header included for COPY)
csv_buf = io.StringIO()
mapped.to_csv(csv_buf, index=False)
csv_buf.seek(0)

prog = st.progress(0.0)
status = st.empty()

conn = None
try:
    conn = open_psycopg2_conn()
    cur = conn.cursor()

    # Optional truncate
    if truncate:
        t0 = time.time()
        cur.execute(f'TRUNCATE TABLE "{schema}"."{table}" RESTART IDENTITY CASCADE;')
        conn.commit()
        log("truncate_done", {"elapsed_sec": round(time.time()-t0, 3)})

    # 1) Create staging table with TEXT columns
    t0 = time.time()
    cols_def = ", ".join([f'"{c}" TEXT' for c in target_cols])
    cur.execute(f'CREATE TEMP TABLE "{staging}" ({cols_def}) ON COMMIT DROP;')
    conn.commit()
    log("staging_created", {"staging": staging, "elapsed_sec": round(time.time()-t0, 3)})

    # 2) COPY CSV (header) into staging
    t0 = time.time()
    copy_sql = f'COPY "{staging}" ({cols_quoted}) FROM STDIN WITH (FORMAT csv, HEADER true, DELIMITER \'{delim if delim!="\t" else "\\t"}\', QUOTE \'{quotechar}\')'
    cur.copy_expert(copy_sql, csv_buf)
    conn.commit()
    log("copy_done", {"elapsed_sec": round(time.time()-t0, 3)})

    # 3) Insert from staging into real table
    t0 = time.time()
    cur.execute(insert_sql.format(staging=f'"{staging}"'))
    inserted = cur.rowcount  # rows inserted (skipped conflicts aren't counted)
    conn.commit()
    log("insert_done", {"inserted": inserted, "elapsed_sec": round(time.time()-t0, 3)})

    prog.progress(1.0)
    status.info("Finalizing…")

    total_elapsed = time.time() - start_total
    st.success(f"Completed. Inserted ~{inserted:,} rows into {schema}.{table} in {total_elapsed:.2f}s.")
    st.balloons()

except Exception as e:
    if conn:
        try: conn.rollback()
        except Exception: pass
    st.error(f"Upload failed: {e}")
    log("upload_failed", {"error": str(e)})
finally:
    if conn:
        try:
            cur.close()
        except Exception:
            pass
        close_psycopg2_conn(conn)

# Debug panel refresh
with st.expander("🐞 Debug panel", expanded=False):
    st.code(dump_logs_text() or "No logs yet.", language="text")
    st.download_button("⬇️ Download logs", data=(dump_logs_text() or "No logs."),
                       file_name="upload_debug_logs.txt", mime="text/plain")
