
import os
import re
import sqlite3
import subprocess
import requests
import json
import time
import pandas as pd
from hashlib import md5
from dotenv import load_dotenv
from vanna.remote import VannaDefault

# ─── Configuration ────────────────────────────────────────────────────────────
GITHUB_RAW_SQL_URL = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
FILES_DIR          = "files"
os.makedirs(FILES_DIR, exist_ok=True)

SCHEMA_PATH        = f"{FILES_DIR}/schema.sql"
DB_FILE            = f"{FILES_DIR}/mydb.sqlite"
VANNA_FLAG_PATH    = f"{FILES_DIR}/vanna_trained.flag"
DB_INITED_FLAG     = f"{FILES_DIR}/db_inited.flag"
CACHE_PATH         = f"{FILES_DIR}/query_cache.json"

DB_PATH            = DB_FILE
load_dotenv()
VANNA_API_KEY      = os.getenv("VANNA_API_KEY")
VANNA_MODEL        = "4base"
MISTRAL_API_KEY    = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL    = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL      = "mistral-small"

query_cache = {}

# ─── Helper Functions ──────────────────────────────────────────────────────────
def download_sql_file_with_curl(url, output_path=SCHEMA_PATH):
    subprocess.run(["curl", "-L", url, "-o", output_path], check=True)

def extract_ddl(sql_script: str) -> str:
    blocks = re.findall(r"(CREATE\s+TABLE.+?;)", sql_script, flags=re.IGNORECASE | re.DOTALL)
    return "\n\n".join(blocks)

def extract_first_table_name(sql_script: str) -> str:
    m = re.search(r"CREATE\s+TABLE\s+[`\"[]?(\w+)[`\"\]]?", sql_script, flags=re.IGNORECASE)
    return m.group(1) if m else None

def call_mistral_chat(prompt: str, model=MISTRAL_MODEL, temp=0.2) -> str:
    resp = requests.post(
        MISTRAL_API_URL,
        headers={
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp
        },
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def sanitize_sql(raw_sql: str) -> str:
    lines = raw_sql.strip().splitlines()
    lines = [line for line in lines if not line.lstrip().startswith("--")]
    return "\n".join(lines).strip()

def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_cache():
    with open(CACHE_PATH, "w") as f:
        json.dump(query_cache, f)

def get_connection():
    first_time = not os.path.exists(DB_INITED_FLAG)
    conn = sqlite3.connect(DB_PATH)
    if first_time:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        with open(DB_INITED_FLAG, "w") as f:
            f.write("ok")
    return conn

# ─── Vanna Setup ──────────────────────────────────────────────────────────────
if not VANNA_API_KEY:
    raise RuntimeError("Please set VANNA_API_KEY in your environment.")
vn = VannaDefault(model=VANNA_MODEL, api_key=VANNA_API_KEY)

def load_sql_and_train_once(schema_path: str):
    with open(schema_path, "r", encoding="utf-8") as f:
        full_sql = f.read()
    ddl = extract_ddl(full_sql)
    if not ddl:
        raise RuntimeError("No CREATE TABLE statements found!")
    vn.train(ddl=ddl)

    table = extract_first_table_name(ddl)
    if table:
        example_sql = f"SELECT * FROM {table} LIMIT 5;"
        vn.train(sql=example_sql)

# ─── Query Processor ──────────────────────────────────────────────────────────
query_cache = load_cache()

def process_nl_query(conn, nl_query: str) -> pd.DataFrame:
    key = md5(nl_query.encode()).hexdigest()
    start = time.time()
    if key in query_cache:
        sql = query_cache[key]
    else:
        sql = vn.generate_sql(nl_query, allow_llm_to_see_data=True) or ""
        sql = sanitize_sql(sql)
        if not sql.strip() or sql.lower().startswith("error"):
            cur = conn.cursor()
            cur.execute("SELECT sql FROM sqlite_master WHERE type='table';")
            schema_desc = "\n".join(r[0] for r in cur.fetchall() if r[0])
            sql = call_mistral_chat(
                f"You are an expert SQL assistant.\nSchema:\n{schema_desc}\n\nQuestion: {nl_query}\nSQL:"
            )
            sql = sanitize_sql(sql)
        query_cache[key] = sql
        save_cache()

    try:
        df = pd.read_sql_query(sql, conn)
    except Exception as e:
        print("❌ Execution error:", e)
        return pd.DataFrame()

    duration = time.time() - start
    print(f"⏱️ Took {duration:.3f}s total.")
    return df
