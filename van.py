import os
import re
import sqlite3
import subprocess
import requests
import json
import time
from hashlib import md5
from dotenv import load_dotenv
from vanna.remote import VannaDefault

# ─── Configuration ────────────────────────────────────────────────────────────
GITHUB_RAW_SQL_URL   = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
SCHEMA_PATH          = "files/schema.sql"
DB_FILE              = "files/mydb.sqlite"
DB_SCHEMA_FLAG       = "files/db_inited.flag"
VANNA_FLAG_PATH      = "files/vanna_trained.flag"
CACHE_PATH           = "files/query_cache.json"
DB_PATH              = DB_FILE

load_dotenv()
VANNA_API_KEY   = os.getenv("VANNA_API_KEY")
VANNA_MODEL     = "4base"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL   = "mistral-large"

if not VANNA_API_KEY:
    raise RuntimeError("Please set VANNA_API_KEY in your environment.")

vn = VannaDefault(model=VANNA_MODEL, api_key=VANNA_API_KEY)

# ─── Helper Functions ─────────────────────────────────────────────────────────
def download_sql_file_with_curl(url, output_path=SCHEMA_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    subprocess.run(["curl", "-L", url, "-o", output_path], check=True)
    print(f"✅ Downloaded schema to {output_path}")
    return output_path


def extract_ddl(sql_script: str) -> str:
    blocks = re.findall(r"(CREATE\s+TABLE.+?;)", sql_script, flags=re.IGNORECASE|re.DOTALL)
    return "\n\n".join(blocks)


def extract_first_table_name(sql_script: str) -> str:
    m = re.search(r"CREATE\s+TABLE\s+[`\"\[]?(\w+)[`\"\]]?", sql_script, flags=re.IGNORECASE)
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


# ─── Database Setup ────────────────────────────────────────────────────────────
def get_connection():
    # ensure directories exist
    os.makedirs(os.path.dirname(SCHEMA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)

    first_time = not os.path.exists(DB_SCHEMA_FLAG)
    conn = sqlite3.connect(DB_PATH)
    if first_time:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        with open(DB_SCHEMA_FLAG, "w") as f:
            f.write("ok")
        print("✅ Schema loaded into new DB.")
    else:
        print("⏩ Schema already in DB, skipping DDL.")
    return conn


def load_and_train(conn, schema_path: str):
    with open(schema_path, "r", encoding="utf-8") as f:
        full_sql = f.read()

    ddl = extract_ddl(full_sql)
    if not ddl:
        raise RuntimeError("No CREATE TABLE statements found!")

    vn.train(ddl=ddl)

    table = extract_first_table_name(ddl)
    if not table:
        raise RuntimeError("Couldn't extract a table name!")

    vn.train(sql=f"SELECT * FROM {table} LIMIT 5;")
    # allow real-data access
    vn.run_sql = lambda q: conn.execute(q).fetchall()
    print("✅ Vanna trained on schema + sample query.")


# ─── Query Cache ───────────────────────────────────────────────────────────────
def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


query_cache = load_cache()


def process_nl_query(conn, nl_query: str):
    start = time.time()
    key = md5(nl_query.encode()).hexdigest()
    if key in query_cache:
        print("🔁 Using cached SQL.")
        sql = query_cache[key]
    else:
        print("🧠 Asking Vanna (with real data)…")
        sql = vn.generate_sql(nl_query, allow_llm_to_see_data=True) or ""
        if not sql.strip():
            print("⚠️ Vanna failed → falling back to Mistral.")
            cur = conn.cursor()
            cur.execute("SELECT sql FROM sqlite_master WHERE type='table';")
            schema_desc = "\n".join(r[0] for r in cur.fetchall() if r[0])
            sql = call_mistral_chat(
                f"You are an expert SQL assistant.\nSchema:\n{schema_desc}\n\nQuestion: {nl_query}\nSQL:"
            )
        query_cache[key] = sql
        save_cache(query_cache)

    print(f"📄 SQL: {sql}")
    try:
        rows = conn.execute(sql).fetchall()
        print("📊 Result:", rows)
    except Exception as e:
        print("❌ Execution error:", e)

    duration = time.time() - start
    print(f"⏱️  Took {duration:.3f}s.")


# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Download schema if missing
    if not os.path.exists(SCHEMA_PATH):
        download_sql_file_with_curl(GITHUB_RAW_SQL_URL)

    # 2) Initialize DB and train Vanna once
    if not os.path.exists(VANNA_FLAG_PATH):
        conn = get_connection()
        load_and_train(conn, SCHEMA_PATH)
        with open(VANNA_FLAG_PATH, "w") as f:
            f.write("trained")
    else:
        print("⚡ Vanna already trained; opening existing DB.")
        conn = get_connection()

    # 3) Process your natural-language query
    process_nl_query(conn, "top 5 countries on the basis of sales")
