# scripts/list_documents.py
import sqlite3
DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"
conn = sqlite3.connect(DB); conn.row_factory = sqlite3.Row

rows = conn.execute("SELECT id, titulo, path FROM documents ORDER BY id").fetchall()
print("=== documents ===")
for r in rows:
    print(f"{r['id']:>3}  {r['titulo']}  |  {r['path']}")
