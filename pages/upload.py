# pages/5_Upload_CSV.py
import io
import math
import unicodedata
import pandas as pd
import streamlit as st
from db_handler import DatabaseManager

st.set_page_config(page_title="CSV → Table Upload", layout="wide")
st.title("⬆️ CSV → Table Upload (with table-specific templates)")

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
            c.column_default,
            c.character_maximum_length,
            c.numeric_precision,
            c.numeric_scale
        FROM information_schema.columns c
        WHERE c.table_schema = %s AND c.table_name = %s
        ORDER BY c.ordinal_position
    """
    return db.fetch_data(q, (schema, table))

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
    return df["column_name"].tolist()

def normalize_name(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower().replace(" ", "_")

def automap(csv_cols, table_cols):
    norm_csv = {normalize_name(c): c for c in csv_cols}
    mapping = {}
    for tcol in table_cols:
        key = normalize_name(tcol)
        mapping[tcol] = norm_csv.get(key)  # may be None
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
# CSV template generator (per data type)
# ───────────────────────────────────────────────────────────────
def example_for_type(row) -> str:
    dt = (row["data_type"] or "").lower()
    # quick examples per type
    if "int" in dt:
        return "123"
    if "numeric" in dt or "decimal" in dt:
        return "9.99"
    if "double" in dt or "real" in dt or "float" in dt:
        return "3.14"
    if "bool" in dt:
        return "true"
    if "date" == dt:
        return "2025-01-01"
    if "timestamp" in dt:
        return "2025-01-01 12:34:56"
    if "time" == dt:
        return "12:34:56"
    if "uuid" in dt:
        return "00000000-0000-0000-0000-000000000000"
    if "json" in dt:
        return '{"key":"value"}'
    if "char" in dt or "text" in dt:
        return "example"
    return "value"

def build_csv_template(cols_df: pd.DataFrame, pks: list[str]) -> pd.DataFrame:
    # Column order = table order; include all columns.
    # Example row = one best-effort example based on data type.
    example = {}
    for _, r in cols_df.iterrows():
        c = r["column_name"]
        example[c] = example_for_type(r)
    # two rows for illustration
    df = pd.DataFrame([example, example])
    return df

def required_columns(cols_df: pd.DataFrame, pks: list[str]) -> list[str]:
    req = []
    for _, r in cols_df.iterrows():
        c = r["column_name"]
        not_null = not bool(r["is_nullable"])
        has_default = r["column_default"] not in (None, "")
        # PKs are effectively required unless default (e.g. serial)
        if c in pks and not has_default:
            req.append(c)
        elif not_null and not has_default:
            req.append(c)
    return req

# ───────────────────────────────────────────────────────────────
# UI — target table selection + template
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
schema = st.selectbox(
    "Schema",
    options=schemas,
    index=0 if "public" not in schemas else schemas.index("public"),
)
subset = tables[tables["table_schema"] == schema]
table = st.selectbox("Table", options=subset["table_name"].tolist() or ["— none —"])

if not table or table == "— none —":
    st.info("Pick a schema and table to continue.")
    st.stop()

cols_df = table_columns(schema, table)
pks = table_primary_keys(schema, table)
req_cols = required_columns(cols_df, pks)
template_df = build_csv_template(cols_df, pks)

st.subheader("📘 Data dictionary")
dd = cols_df.copy()
dd.insert(1, "is_primary_key", dd["column_name"].isin(pks))
st.dataframe(dd, use_container_width=True, hide_index=True)

with st.expander("📄 CSV Template & Examples", expanded=True):
    st.markdown(
        f"""
**How to prepare your CSV for `{schema}.{table}`**

- Columns may be in **any order** — you will map them before upload.
- **Required columns** (no default & NOT NULL or part of PK): `{', '.join(req_cols) if req_cols else '— none —'}`.
- **Optional columns** can be left out or blank (they’ll be inserted as NULL).
- Suggested formats:
  - **DATE**: `YYYY-MM-DD` (e.g., `2025-01-01`)
  - **TIMESTAMP**: `YYYY-MM-DD HH:MM:SS` (e.g., `2025-01-01 12:34:56`)
  - **BOOLEAN**: `true` / `false`
  - **NUMERIC**: plain digits like `9.99`
- If your file has no header row, uncheck “CSV has header row” and we’ll generate generic names.
        """
    )
    st.write("**Template preview (example values):**")
    st.dataframe(template_df, use_container_width=True)
    st.download_button(
        "⬇️ Download CSV template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{schema}.{table}.template.csv",
        mime="text/csv",
    )

# ───────────────────────────────────────────────────────────────
# Upload the CSV
# ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Choose a CSV file to upload into this table", type=["csv"])
if not uploaded:
    st.info("Pick a CSV to continue.")
    st.stop()

# Read CSV + show preview
try:
    df_csv = csv_to_df(uploaded, delimiter=delim.replace("\\t", "\t"), encoding=enc, has_header=has_header, quotechar=quotechar)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.write("**CSV Preview**")
st.dataframe(df_csv.head(50), use_container_width=True)

# ───────────────────────────────────────────────────────────────
# Column mapping
# ───────────────────────────────────────────────────────────────
st.subheader("Map CSV columns → table columns")
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

# Validate required columns mapped
missing_required = [c for c in req_cols if c not in target_cols]
if missing_required:
    st.error(f"These required columns are not mapped: {', '.join(missing_required)}")
    st.stop()

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
st.caption(f"{len(mapped):,} rows will be inserted into {schema}.{table} with columns {target_cols}.")

# ───────────────────────────────────────────────────────────────
# Upload action (chunked insert)
# ───────────────────────────────────────────────────────────────
do_upload = st.button("🚀 Start upload")
if not do_upload:
    st.stop()

# Confirm destructive option
if truncate:
    st.error("You chose to TRUNCATE the table before load. This will DELETE all existing rows.")
    if not st.checkbox("I understand, proceed with TRUNCATE"):
        st.stop()

try:
    # Optional truncate first
    if truncate:
        db.execute_command(f'TRUNCATE TABLE "{schema}"."{table}" RESTART IDENTITY CASCADE;')

    insert_sql = build_insert_sql(schema, table, target_cols, on_conflict_do_nothing=on_conflict)

    total_rows = len(mapped)
    chunks = int(math.ceil(total_rows / chunk_size))
    prog = st.progress(0.0)
    status = st.empty()

    inserted_total = 0
    for i in range(chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_rows)
        batch = mapped.iloc[start:end]
        params = [tuple(batch[c].tolist()[j] for c in target_cols) for j in range(len(batch))]

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
