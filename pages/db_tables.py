# pages/4_DB_Tables.py
import streamlit as st
import pandas as pd
from db_handler import DatabaseManager

st.set_page_config(page_title="DB Tables Browser", layout="wide")
st.title("📚 Database Tables Browser")

db = DatabaseManager()

@st.cache_data(show_spinner=False, ttl=60)
def load_objects(include_views: bool) -> pd.DataFrame:
    """List tables (and optionally views) from information_schema."""
    q = """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
    """
    if not include_views:
        q += " AND table_type = 'BASE TABLE'"
    q += " ORDER BY table_schema, table_name"
    return db.fetch_data(q)

@st.cache_data(show_spinner=False, ttl=60)
def get_columns(schema: str, table: str) -> pd.DataFrame:
    q = """
        SELECT
            ordinal_position,
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """
    return db.fetch_data(q, (schema, table))

@st.cache_data(show_spinner=True, ttl=30)
def get_row_count(schema: str, table: str) -> int:
    # identifiers come from catalog (safe); quote with ""
    df = db.fetch_data(f'SELECT COUNT(*) AS c FROM "{schema}"."{table}"')
    return int(df["c"].iat[0]) if not df.empty else 0

@st.cache_data(show_spinner=True, ttl=30)
def preview_table(schema: str, table: str, limit: int) -> pd.DataFrame:
    return db.fetch_data(f'SELECT * FROM "{schema}"."{table}" LIMIT {int(limit)}')

# Controls
colA, colB, colC, colD = st.columns([1, 1, 1, 1.2])
with colA:
    include_views = st.toggle("Include views", value=False)
with colB:
    with_counts = st.toggle("Show row counts (slow)", value=False,
                            help="Runs COUNT(*) per table.")
with colC:
    default_limit = st.number_input("Preview rows", 5, 1000, 50, 5)
with colD:
    name_filter = st.text_input("Filter name contains", "", label_visibility="collapsed")

# Load + filter
objects_df = load_objects(include_views)
schemas = sorted(objects_df["table_schema"].unique().tolist())
schema_pick = st.multiselect("Schemas", options=schemas, default=schemas)

filtered = objects_df[objects_df["table_schema"].isin(schema_pick)].copy()
if name_filter:
    nf = name_filter.lower()
    filtered = filtered[filtered["table_name"].str.lower().str.contains(nf)]

# Optional counts
if with_counts and not filtered.empty:
    st.info("Counting rows…", icon="⏳")
    counts = []
    for _, r in filtered.iterrows():
        try:
            counts.append(get_row_count(r["table_schema"], r["table_name"]))
        except Exception:
            counts.append(None)
    filtered.insert(2, "row_count", counts)

st.subheader("Objects")
st.dataframe(filtered.reset_index(drop=True), use_container_width=True, hide_index=True)

# Inspect
st.divider()
st.subheader("Inspect a table / view")

choices = [f'{s}.{t}' for s, t in zip(filtered["table_schema"], filtered["table_name"])]
selection = st.selectbox("Pick an object", options=choices if choices else ["— none —"])

if choices:
    schema, table = selection.split(".", 1)

    left, right = st.columns([1, 3])
    with left:
        if st.button("🔄 Refresh", use_container_width=True):
            load_objects.clear(); get_columns.clear()
            get_row_count.clear(); preview_table.clear()
            st.experimental_rerun()

        show_cols = st.checkbox("Show columns", True)
        do_preview = st.checkbox("Preview data", True)
        limit = st.number_input("Limit", 1, 5000, int(default_limit), 10)

        if with_counts:
            try:
                st.metric("Row count", f"{get_row_count(schema, table):,}")
            except Exception as e:
                st.warning(f"Row count failed: {e}")

    with right:
        tabs = st.tabs(["Columns", "Preview", "SQL"])
        with tabs[0]:
            if show_cols:
                try:
                    st.dataframe(get_columns(schema, table),
                                 use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Failed to load columns: {e}")

        with tabs[1]:
            if do_preview:
                try:
                    df = preview_table(schema, table, limit)
                    st.dataframe(df, use_container_width=True)
                    st.download_button("⬇️ Download CSV",
                                       data=df.to_csv(index=False).encode("utf-8"),
                                       file_name=f"{schema}.{table}.csv",
                                       mime="text/csv")
                except Exception as e:
                    st.error(f"Preview failed: {e}")

        with tabs[2]:
            st.code(f'SELECT * FROM "{schema}"."{table}" LIMIT {int(limit)};', language="sql")
else:
    st.info("No objects to show with the current filters.")
