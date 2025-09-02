import sqlite3

DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

print("== Tablas ==")
tabs = conn.execute("""
SELECT name FROM sqlite_master
WHERE type='table' OR type='view'
ORDER BY name
""").fetchall()
for t in tabs:
    name = t["name"]
    try:
        c = conn.execute(f"SELECT COUNT(*) AS n FROM {name}").fetchone()["n"]
    except Exception:
        c = "?"
    print(f"- {name}: {c}")

print("\n== Virtual tables (FTS) ==")
fts = conn.execute("""
SELECT name, sql FROM sqlite_master
WHERE sql LIKE '%fts5%' OR sql LIKE '%fts4%'
""").fetchall()
for r in fts:
    print(f"- {r['name']} (def): {r['sql'][:120]}...")

print("\n== Columnas por tabla clave (si existen) ==")
for candidate in ["documents","chunks","embeddings","fts_chunks",
                  "documents_v2","chunks_v2","embeddings_v2","fts_chunks_v2",
                  "documents_v3","chunks_v3","embeddings_v3","fts_chunks_v3"]:
    try:
        cols = conn.execute(f"PRAGMA table_info({candidate})").fetchall()
        if cols:
            print(f"[{candidate}] -> " + ", ".join([c["name"] for c in cols]))
    except Exception:
        pass
