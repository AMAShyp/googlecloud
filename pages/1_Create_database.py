# pages/1_Create_Database.py
import streamlit as st
import pandas as pd
import psycopg2
from db_utils import get_conn, list_databases, valid_db

st.title("Create Database")

db_name = st.text_input("Database name", max_chars=32, help="Letters, numbers, underscores; must start with a letter.")
sql_extra = st.text_area("Optional SQL to run in the new database (e.g. CREATE TABLE …)", height=140)

if st.button("Create Database and Run SQL"):
    if not valid_db(db_name or ""):
        st.error("Invalid name.")
    else:
        try:
            conn = get_conn()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(f'CREATE DATABASE "{db_name}";')
            conn.close()
            st.success(f"Database **{db_name}** created.")

            if sql_extra.strip():
                with get_conn(db_name) as new_conn, new_conn.cursor() as cur:
                    cur.execute(sql_extra)
                    if cur.description:
                        rows = cur.fetchall()
                        cols = [c[0] for c in cur.description]
                        st.dataframe(pd.DataFrame(rows, columns=cols))
                    else:
                        new_conn.commit()
                st.success("Extra SQL executed.")
        except psycopg2.errors.DuplicateDatabase:
            st.warning("Database already exists.")
        except Exception as e:
            st.error(e)

if st.button("List existing databases"):
    st.dataframe(pd.DataFrame({"Database": list_databases()}))
