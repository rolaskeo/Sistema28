# scripts/inspect_columns_plus.py
import sqlite3

DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"
conn = sqlite3.connect(DB); conn.row_factory = sqlite3.Row

print("== PRAGMA table_info(document_chunks) ==")
for c in conn.execute("PRAGMA table_info(document_chunks)"):
    print(dict(c))

print("\n== 1 fila de document_chunks ==")
row = conn.execute("SELECT * FROM document_chunks LIMIT 1").fetchone()
if row:
    print(list(row.keys()))
    print(dict(row))
