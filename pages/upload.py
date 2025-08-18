# pages/5_Upload_CSV.py
import io
import math
import unicodedata
import pandas as pd
import streamlit as st
from db_handler import DatabaseManager

st.set_page_config(page_title="CSV → Table Upload", layout="wide")
st.title("⬆️ Upload CSV Data to a Table")

db = DatabaseManager()

# ───────────────────────────────────────────────────────────────
# Helpers: metadata
# ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=60)
def list_tables():
    q = """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog','information_schema')
          AND table_type = 'BASE TABLE'
        ORDER BY table_schema, table_name
    """
    return db.fetch_data(q)

@st.cache_data(show_spinner=False, ttl=60)
def table_columns(schema: str, table: str) -> pd.DataFrame:
    q = """
        SELECT
            c.ordinal_position,
            c.column_name,
            c.data_type,
            c.is_nullable = 'YES' AS is_nullable,
            c.column_default
        FROM information_schema.columns c
        WHERE c.table_schema = %s AND c.table_name = %s
        ORDER BY c.ordinal_position
    """
    return db.fetch_data(q, (schema, table))

# ───────────────────────────────────────────────────────────────
# Helpers: CSV & mapping
# ───────────────────────────────────────────────────────────────
def normalize_name(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower().replace(" ", "_")

def automap(csv_cols, table_cols):
    norm_csv = {normalize_name(c): c for c in csv_cols}
    mapping = {}
    for tcol in table_cols:
        mapping[tcol] = norm_csv.get(normalize_name(tcol))  # may be None
    return mapping

def csv_to_df(uploaded_file, delimiter, encoding, has_header, quotechar):
    raw = uploaded_file.read()
    opts = dict(sep=delimiter or ",", encoding=encoding or "utf-8", dtype=str,
                keep_default_na=False, na_values=[""], quotechar=quotechar or '"')
    if has_header:
        return pd.read_csv(io.BytesIO(raw), **opts)
    df = pd.read_csv(io.BytesIO(raw), header=None, **opts)
    df.columns = [f"column_{i+1}" for i in range(df.shape[1])]
    return df

def build_insert_sql(schema: str, table: str, cols: list[str], on_conflict_do_nothing: bool):
    cols_quoted = ', '.join([f'"{c}"' for c in cols])
    placeholders = ', '.join(['%s'] * len(cols))
    sql = f'INSERT INTO "{schema}"."{table}" ({cols_quoted}) VALUES ({placeholders})'
    if on_conflict_do_nothing:
        sql += " ON CONFLICT DO NOTHING"
    return sql

def coerce_cell(val: str):
    if val is None:
        return None
    if isinstance(val, str):
        v = val.strip()
        if v == "":
            return None
        return v
    return val

# ───────────────────────────────────────────────────────────────
# NEW: Sample CSV generation
# ───────────────────────────────────────────────────────────────
def example_value(pg_type: str, i: int) -> str:
    """Return a simple example value string for a given Postgres data_type."""
    t = (pg_type or "").lower()
    # common scalar types
    if t in ("integer", "int", "int4", "smallint", "int2", "bigint", "int8", "serial", "bigserial"):
        return str(1000 + i)
    if "numeric" in t or "decimal" in t or "double" in t or "real" in t or "float" in t:
        return f"{i + 0.5}"
    if t in ("boolean", "bool"):
        return "true" if i % 2 == 0 else "false"
    if "timestamp" in t:
        return "2025-01-01 12:00:00"
    if t == "date":
        return "2025-01-01"
    if t == "time":
        return "12:34:56"
    if "uuid" in t:
        # looks like uuid but string okay; DB or app can cast
        return f"00000000-0000-0000-0000-0000000000{i%10}"
    if "json" in t:
        return '{"key":"value"}'
    if "char" in t or "text" in t or "name" in t:
        return f"sample_{i}"
    if "bytea" in t:
        return "\\x"  # empty bytea
    # fallback
    return f"sample_{i}"

def build_sample_dataframe(cols_df: pd.DataFrame, required_only: bool, sample_rows: int = 5) -> pd.DataFrame:
    """
    Create a sample DataFrame with headers matching the table columns.
    If required_only=True, include only columns that are NOT NULL and have NO default.
    """
    df = cols_df.copy()
    if required_only:
        # required if NOT nullable AND has no default
        req_mask = (~df["is_nullable"]) & (df["column_default"].isna() | (df["column_default"] == ""))
        df = df[req_mask]

    headers = df["column_name"].tolist()
    types = df["data_type"].tolist()

    rows = []
    for i in range(sample_rows):
        rows.append([example_value(types[j], i) for j in range(len(headers))])

    return pd.DataFrame(rows, columns=headers)

# ───────────────────────────────────────────────────────────────
# UI — pick target table + CSV settings
# ───────────────────────────────────────────────────────────────
optA, optB, optC, optD = st.columns([1, 1, 1, 1])
with optA:
    delim = st.text_input("Delimiter", value=",", help="e.g. , ; | \\t")
with optB:
    enc = st.text_input("Encoding", value="utf-8")
with optC:
    quotechar = st.text_input("Quote char", value='"')
with optD:
    has_header = st.checkbox("CSV has header", value=True)

tables = list_tables()
schemas = sorted(tables["table_schema"].unique().tolist())
schema = st.selectbox("Schema", options=schemas, index=schemas.index("public") if "public" in schemas else 0)
subset = tables[tables["table_schema"] == schema]
table = st.selectbox("Table", options=subset["table_name"].tolist() or ["— none —"])

cols_df = pd.DataFrame()
if table and table != "— none —":
    cols_df = table_columns(schema, table)
    with st.expander("Table columns (from information_schema)"):
        st.dataframe(cols_df, use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────────────────────
# NEW: Sample CSV generator (downloadable)
# ───────────────────────────────────────────────────────────────
st.subheader("📄 Sample CSV for this table")
if cols_df.empty:
    st.info("Pick a table to generate a sample CSV.")
else:
    sleft, sright = st.columns([1, 3])
    with sleft:
        required_only = st.checkbox("Only required columns", value=True,
                                    help="Include non-nullable columns without default.")
        sample_rows = st.number_input("Sample rows", 1, 100, 5, 1)
    sample_df = build_sample_dataframe(cols_df, required_only=required_only, sample_rows=int(sample_rows))
    st.dataframe(sample_df, use_container_width=True)
    st.download_button(
        "⬇️ Download sample CSV",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{schema}.{table}.sample.csv",
        mime="text/csv",
    )
    st.caption("Tip: start from this CSV, then add/remove columns/rows as needed.")

st.divider()
st.subheader("📥 Upload your CSV")

uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
if not uploaded:
    st.info("Pick a CSV to continue.")
    st.stop()

# ───────────────────────────────────────────────────────────────
# Read CSV + show preview
# ───────────────────────────────────────────────────────────────
try:
    df_csv = csv_to_df(uploaded, delimiter=delim.replace("\\t", "\t"), encoding=enc,
                       has_header=has_header, quotechar=quotechar)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.write("**CSV Preview**")
st.dataframe(df_csv.head(50), use_container_width=True)

if cols_df.empty:
    st.warning("No columns found for the selected table.")
    st.stop()

# ───────────────────────────────────────────────────────────────
# Column mapping (auto + override)
# ───────────────────────────────────────────────────────────────
st.subheader("Map CSV columns to table columns")
table_cols = cols_df["column_name"].tolist()
csv_cols = df_csv.columns.tolist()
default_map = automap(csv_cols, table_cols)

mapping = {}
map_cols = st.container()
with map_cols:
    m1, m2 = st.columns([2.5, 1.5])
    with m1:
        st.caption("Table column → CSV column")
        for tcol in table_cols:
            default_choice = default_map.get(tcol)
            options = ["— skip —"] + csv_cols
            mapping[tcol] = st.selectbox(
                f'↪ {tcol}',
                options=options,
                index=(options.index(default_choice) if default_choice in csv_cols else 0),
                key=f"map_{tcol}",
            )
    with m2:
        st.caption("Load options")
        truncate = st.checkbox("TRUNCATE table before load (danger!)", value=False)
        on_conflict = st.checkbox("ON CONFLICT DO NOTHING (skip duplicates)", value=True)
        chunk_size = st.number_input("Insert chunk size", min_value=100, max_value=20000, value=1000, step=100)

# Effective columns to load
target_cols = [c for c in table_cols if mapping.get(c) and mapping[c] != "— skip —"]
if not target_cols:
    st.warning("No columns mapped. Map at least one table column to a CSV column.")
    st.stop()

# Build mapped frame
mapped = pd.DataFrame()
try:
    for tcol in target_cols:
        mapped[tcol] = df_csv[mapping[tcol]].map(coerce_cell)
except KeyError as e:
    st.error(f"Mapping refers to a missing CSV column: {e}")
    st.stop()

st.write("**Mapped Preview**")
st.dataframe(mapped.head(50), use_container_width=True)
st.caption(f"{len(mapped):,} rows will be attempted to insert into {schema}.{table} with columns {target_cols}.")

# ───────────────────────────────────────────────────────────────
# Upload action (chunked executemany + per-chunk transaction)
# ───────────────────────────────────────────────────────────────
do_upload = st.button("🚀 Start upload")
if not do_upload:
    st.stop()

if truncate:
    st.error("You chose to TRUNCATE the table before load. This will DELETE all existing rows.")
    if not st.checkbox("I understand, proceed with TRUNCATE"):
        st.stop()

try:
    if truncate:
        db.execute_command(f'TRUNCATE TABLE "{schema}"."{table}" RESTART IDENTITY CASCADE;')

    insert_sql = build_insert_sql(schema, table, target_cols, on_conflict_do_nothing=on_conflict)
    total_rows = len(mapped)
    chunks = int(math.ceil(total_rows / chunk_size)) if total_rows else 0

    prog = st.progress(0.0)
    status = st.empty()
    inserted_total = 0

    for i in range(chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_rows)
        batch = mapped.iloc[start:end]

        # Prepare params for executemany
        vals = batch.apply(lambda r: tuple(r[c] for c in target_cols), axis=1).tolist()

        cur = db.conn.cursor()
        try:
            # generous per-chunk timeout (120s)
            cur.execute("SET LOCAL statement_timeout = 120000;")
            cur.executemany(insert_sql, vals)
        except Exception as e:
            db.conn.rollback()
            cur.close()
            st.error(f"Failed on chunk {i+1}/{chunks} (rows {start+1}-{end}). Error: {e}")
            st.stop()
        else:
            db.conn.commit()
            cur.close()
            inserted_total += len(batch)

        prog.progress((i + 1) / chunks if chunks else 1.0)
        status.info(f"Inserted {inserted_total:,}/{total_rows:,} rows...")

    prog.progress(1.0)
    st.success(f"Done! Inserted {inserted_total:,} rows into {schema}.{table}.")
    st.balloons()

except Exception as e:
    st.error(f"Upload failed: {e}")
