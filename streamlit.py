# streamlit_app.py
import os
import pandas as pd
import streamlit as st
from van import get_connection, process_nl_query, download_sql_file_with_curl, load_sql_and_train_once, SCHEMA_PATH, VANNA_FLAG_PATH

st.set_page_config(page_title="NL → SQL → CSV Exporter", layout="wide")

st.title("🧠 Natural Language → SQL → CSV Exporter")

# Setup database and Vanna on first load
if not os.path.exists(SCHEMA_PATH):
    download_sql_file_with_curl("https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql")

if not os.path.exists(VANNA_FLAG_PATH):
    conn = get_connection()
    load_sql_and_train_once(SCHEMA_PATH)
    with open(VANNA_FLAG_PATH, "w") as f:
        f.write("trained")
else:
    conn = get_connection()

# Streamlit UI
nl_query = st.text_input("Enter your natural language query:", placeholder="e.g. Show top 5 albums")
submit = st.button("Run Query")

if submit and nl_query.strip():
    df = process_nl_query(conn, nl_query)
    if not df.empty:
        st.success("✅ Query executed successfully!")
        st.dataframe(df)

        # Download buttons
        csv = df.to_csv(index=False).encode('utf-8')
        tsv = df.to_csv(index=False, sep='\t').encode('utf-8')

        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="query_results.csv",
            mime="text/csv"
        )

        st.download_button(
            label="📥 Download as TSV",
            data=tsv,
            file_name="query_results.tsv",
            mime="text/tab-separated-values"
        )
    else:
        st.error("❌ No results found or query failed.")
