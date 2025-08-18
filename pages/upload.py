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
# Helpers
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

def normalize_name(s: str) -> str:
    """Looser case-insensitive, diacritic-insensitive name for auto-mapping."""
    s = (s or "").strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower().replace(" ", "_")

def automap(csv_cols, table_cols):
    norm_csv = {normalize_name(c): c for c in csv_cols}
    mapping = {}
    for tcol in table_cols:
        key = normalize_name(tcol)
        mapping[tcol] = norm_csv.get(key)  # may be None (unmapped)
    return mapping

def csv_to_df(uploaded_file, delimiter, encoding, has_header, quotechar):
    # read once to bytes so we can detect header behavior consistently
    raw = uploaded_file.read()
    opts = dict(sep=delimiter or ",", encoding=encoding or "utf-8", dtype=str,
                keep_default_na=False, na_values=[""], quotechar=quotechar or '"')
    if has_header:
        return pd.read_csv(io.BytesIO(raw), **opts)
    # No header: create generic column names
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
    """Basic coercion: empty string → None; trim; pass through otherwise."""
    if val is None:
        return None
    if isinstance(val, str):
        v = val.strip()
        if v == "":
            return None
        return v
    return val

# ───────────────────────────────────────────────────────────────
# UI — pick target table
# ───────────────────────────────────────────────────────────────
left, right = st.columns([1, 1])
with left:
    delim = st.text_input("Delimiter", value=",", help="e.g. , ; | \\t")
    enc = st.text_input("Encoding", value="utf-8")
with right:
    quotechar = st.text_input("Quote char", value='"')
    has_header = st.checkbox("CSV has header row", value=True)

tables = list_tables()
schemas = sorted(tables["table_schema"].unique().tolist())
schema = st.selectbox("Schema", options=schemas, index=0 if "public" not in schemas else schemas.index("public"))
subset = tables[tables["table_schema"] == schema]
table = st.selectbox("Table", options=subset["table_name"].tolist() or ["— none —"])

cols_df = pd.DataFrame()
if table and table != "— none —":
    cols_df = table_columns(schema, table)
    with st.expander("Table columns"):
        st.dataframe(cols_df, use_container_width=True, hide_index=True)

uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
if not uploaded:
    st.info("Pick a CSV to continue.")
    st.stop()

# ───────────────────────────────────────────────────────────────
# Read CSV + show preview
# ───────────────────────────────────────────────────────────────
try:
    df_csv = csv_to_df(uploaded, delimiter=delim.replace("\\t", "\t"), encoding=enc, has_header=has_header, quotechar=quotechar)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.write("**CSV Preview**")
st.dataframe(df_csv.head(50), use_container_width=True)

if cols_df.empty:
    st.warning("No columns found for the selected table.")
    st.stop()

# ───────────────────────────────────────────────────────────────
# Column mapping
# ───────────────────────────────────────────────────────────────
st.subheader("Map CSV columns to table columns")
table_cols = cols_df["column_name"].tolist()
csv_cols = df_csv.columns.tolist()
default_map = automap(csv_cols, table_cols)

mapping = {}
map_cols = st.container()
with map_cols:
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
        chunk_size = st.number_input("Insert chunk size", min_value=100, max_value=10000, value=1000, step=100)

# Effective columns to load
target_cols = [c for c in table_cols if mapping.get(c) and mapping[c] != "— skip —"]
if not target_cols:
    st.warning("No columns mapped. Map at least one table column to a CSV column.")
    st.stop()

# Build a preview of the mapped frame
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
# Upload action
# ───────────────────────────────────────────────────────────────
do_upload = st.button("🚀 Start upload")
if not do_upload:
    st.stop()

# Confirm destructive option
if truncate:
    st.error("You chose to TRUNCATE the table before load. This will DELETE all existing rows.")
    if not st.checkbox("I understand, proceed with TRUNCATE"):
        st.stop()

# Execute
try:
    # Optional truncate first (outside of main transaction for clarity)
    if truncate:
        db.execute_command(f'TRUNCATE TABLE "{schema}"."{table}" RESTART IDENTITY CASCADE;')

    insert_sql = build_insert_sql(schema, table, target_cols, on_conflict_do_nothing=on_conflict)

    # Chunked insert with a manual transaction (one per chunk)
    total_rows = len(mapped)
    chunks = int(math.ceil(total_rows / chunk_size))
    prog = st.progress(0.0)
    status = st.empty()

    # Use low-level execute to control commit per chunk
    inserted_total = 0
    for i in range(chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_rows)
        batch = mapped.iloc[start:end]
        params = [tuple(batch[c].tolist()[j] for c in target_cols) for j in range(len(batch))]

        # open cursor, executemany, commit
        cur = db.conn.cursor()
        try:
            cur.execute("SET LOCAL statement_timeout = 120000;")  # 120s per chunk
            cur.executemany(insert_sql, params)
        except Exception as e:
            db.conn.rollback()
            cur.close()
            st.error(f"Failed on chunk {i+1}/{chunks} (rows {start+1}-{end}). Error: {e}")
            st.stop()
        else:
            db.conn.commit()
            cur.close()
            inserted_total += len(batch)

        prog.progress((i + 1) / chunks)
        status.info(f"Inserted {inserted_total:,}/{total_rows:,} rows...")

    prog.progress(1.0)
    st.success(f"Done! Inserted {inserted_total:,} rows into {schema}.{table}.")
    st.balloons()

except Exception as e:
    st.error(f"Upload failed: {e}")
