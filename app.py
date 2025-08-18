# pages/5_Item_Transfer.py
import streamlit as st
import pandas as pd
from db_handler import DatabaseManager

st.set_page_config(page_title="Item Transfer (app.item → public.item)", layout="wide")
st.title("🛠️ Transfer Items: app.item → public.item")

db = DatabaseManager()

SRC = ('app', 'item')
DST = ('public', 'item')

ALL_COLS = [
    "itemid","itemnameenglish","itemnamekurdish","classcat","departmentcat",
    "sectioncat","familycat","subfamilycat","shelflife","origincountry",
    "manufacturer","brand","barcode","unittype","packaging","itempicture",
    "createdat","updatedat","threshold","averagerequired","shelfthreshold",
    "shelfaverage","sellingprice","packetbarcode","cartonbarcode","packetsize",
    "cartonsize","carton_price","packet_price",
]

@st.cache_data(show_spinner=False, ttl=30)
def table_exists(schema: str, table: str) -> bool:
    q = """
      SELECT 1 FROM information_schema.tables
      WHERE table_schema=%s AND table_name=%s
      LIMIT 1
    """
    return not db.fetch_data(q, (schema, table)).empty

@st.cache_data(show_spinner=False, ttl=30)
def count_rows(schema: str, table: str) -> int:
    df = db.fetch_data(f'SELECT COUNT(*) AS c FROM "{schema}"."{table}"')
    return int(df["c"].iat[0]) if not df.empty else 0

@st.cache_data(show_spinner=False, ttl=30)
def max_itemid(schema: str, table: str) -> int | None:
    df = db.fetch_data(f'SELECT MAX(itemid) AS mx FROM "{schema}"."{table}"')
    if df.empty or pd.isna(df["mx"].iat[0]):
        return None
    return int(df["mx"].iat[0])

def dry_run_upsert(src_schema, src_table, dst_schema, dst_table):
    """Return (would_insert, would_update) using diff against dst by itemid."""
    q = f'''
    WITH src AS (
        SELECT itemid FROM "{src_schema}"."{src_table}"
    ),
    dst AS (
        SELECT itemid FROM "{dst_schema}"."{dst_table}"
    )
    SELECT
      (SELECT COUNT(*) FROM src s LEFT JOIN dst d USING(itemid) WHERE d.itemid IS NULL) AS would_insert,
      (SELECT COUNT(*) FROM src s JOIN dst d USING(itemid)) AS would_update;
    '''
    df = db.fetch_data(q)
    wi = int(df["would_insert"].iat[0])
    wu = int(df["would_update"].iat[0])
    return wi, wu

def realign_sequence(dst_schema, dst_table, seq_name_guess: str | None = None):
    """
    Align the destination sequence to MAX(itemid). If seq name not provided,
    try to discover it via pg_get_serial_sequence.
    """
    if seq_name_guess:
        seq_expr = f"'{seq_name_guess}'"
    else:
        seq_expr = f"pg_get_serial_sequence('\"{dst_schema}\".\"{dst_table}\"','itemid')"

    q = f'''
    DO $$
    DECLARE
      seq text;
      mx  bigint;
    BEGIN
      SELECT {seq_expr} INTO seq;
      IF seq IS NULL THEN
        -- no serial sequence attached; nothing to do
        RETURN;
      END IF;

      EXECUTE format('SELECT MAX(itemid) FROM "{dst_schema}"."{dst_table}"') INTO mx;
      IF mx IS NULL THEN
        -- empty table → set to 1 (nextval returns 2), but - better set 1
        EXECUTE format('SELECT setval(%L, 1, false)', seq);
      ELSE
        EXECUTE format('SELECT setval(%L, %s, true)', seq, mx);
      END IF;
    END$$;
    '''
    db.execute_command(q)

# UI – status card
src_ok = table_exists(*SRC)
dst_ok = table_exists(*DST)

cols = st.columns(3)
with cols[0]:
    st.metric("Source", f"{SRC[0]}.{SRC[1]}", "OK" if src_ok else "Missing")
with cols[1]:
    st.metric("Target", f"{DST[0]}.{DST[1]}", "OK" if dst_ok else "Missing")
with cols[2]:
    if src_ok and dst_ok:
        st.success("Both tables exist.")
    else:
        st.error("One or both tables are missing. Create them before transferring.")

if not (src_ok and dst_ok):
    st.stop()

# Row counts
src_cnt = count_rows(*SRC)
dst_cnt = count_rows(*DST)

st.write("### Current row counts")
st.write(f"- Source **{SRC[0]}.{SRC[1]}**: **{src_cnt:,}** rows")
st.write(f"- Target **{DST[0]}.{DST[1]}**: **{dst_cnt:,}** rows")

# Transfer options
st.write("### Transfer options")
mode = st.radio(
    "Mode",
    ["Upsert (merge by itemid)", "Replace (truncate then copy)"],
    index=0,
    help="Upsert keeps existing rows and updates them. Replace wipes target first.",
)

with st.expander("Advanced (usually leave as-is)"):
    seq_hint = st.text_input(
        "Target sequence name (optional)",
        value="", placeholder="e.g. item_itemid_seq"
    )

# Dry run (for upsert)
if mode.startswith("Upsert"):
    with st.spinner("Computing dry-run diff…"):
        ins, upd = dry_run_upsert(*SRC, *DST)
    st.info(f"Dry run → would **insert {ins:,}** rows and **update {upd:,}** rows.", icon="ℹ️")

st.divider()

# Action buttons
col1, col2 = st.columns([1, 1])
go_upsert = col1.button("🚀 Start transfer (Upsert)" if mode.startswith("Upsert") else "🚀 Start transfer (Replace)", type="primary")
refresh = col2.button("🔄 Refresh counts")

if refresh:
    count_rows.clear()
    st.rerun()

if go_upsert:
    # Build column lists quoted
    cols_csv = ", ".join([f'"{c}"' for c in ALL_COLS])
    set_csv = ", ".join([f'"{c}" = EXCLUDED."{c}"' for c in ALL_COLS if c != "itemid"])

    try:
        if mode.startswith("Upsert"):
            st.warning("Running UPSERT… please wait.", icon="⏳")
            # One transaction: upsert then re-align sequence
            db.execute_command(f"BEGIN;")
            sql = f'''
            INSERT INTO "{DST[0]}"."{DST[1]}" ({cols_csv})
            SELECT {cols_csv}
            FROM "{SRC[0]}"."{SRC[1]}"
            ON CONFLICT ("itemid")
            DO UPDATE SET {set_csv};
            '''
            db.execute_command(sql)
            # Sequence align
            seqname = seq_hint.strip() or None
            realign_sequence(DST[0], DST[1], seqname)
            db.execute_command("COMMIT;")

            # Post-stats
            count_rows.clear()
            new_cnt = count_rows(*DST)
            st.success(f"Upsert done ✅ Target now has **{new_cnt:,}** rows.")
        else:
            st.warning("Running REPLACE (truncate → copy)… please wait.", icon="⏳")
            db.execute_command("BEGIN;")
            # Wipe target first
            db.execute_command(f'TRUNCATE TABLE "{DST[0]}"."{DST[1]}" RESTART IDENTITY;')
            # Copy all
            sql = f'''
            INSERT INTO "{DST[0]}"."{DST[1]}" ({cols_csv})
            SELECT {cols_csv}
            FROM "{SRC[0]}"."{SRC[1]}";
            '''
            db.execute_command(sql)
            # Ensure sequence aligns (in case identity isn’t attached)
            seqname = seq_hint.strip() or None
            realign_sequence(DST[0], DST[1], seqname)
            db.execute_command("COMMIT;")

            count_rows.clear()
            new_cnt = count_rows(*DST)
            st.success(f"Replace done ✅ Target now has **{new_cnt:,}** rows.")
    except Exception as e:
        try:
            db.execute_command("ROLLBACK;")
        except Exception:
            pass
        st.error(f"Transfer failed: {e}")

# Preview (optional)
st.divider()
st.write("### Quick preview")
c1, c2 = st.columns(2)
with c1:
    if st.button("Preview source (top 20)"):
        df = db.fetch_data(f'SELECT * FROM "{SRC[0]}"."{SRC[1]}" ORDER BY itemid LIMIT 20')
        st.dataframe(df, use_container_width=True)
with c2:
    if st.button("Preview target (top 20)"):
        df = db.fetch_data(f'SELECT * FROM "{DST[0]}"."{DST[1]}" ORDER BY itemid LIMIT 20')
        st.dataframe(df, use_container_width=True)
