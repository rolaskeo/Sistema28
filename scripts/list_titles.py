# scripts/list_titles.py
import sqlite3
DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"
conn = sqlite3.connect(DB); conn.row_factory = sqlite3.Row

print("=== documents (viejo) ===")
for r in conn.execute("SELECT id, titulo, path FROM documents ORDER BY id"):
    print(r["id"], r["titulo"])

print("\n=== documents_v3 (nuevo) ===")
try:
    for r in conn.execute("SELECT id, title FROM documents_v3 ORDER BY id"):
        print(r["id"], r["title"])
except Exception:
    print("(Aún no existe documents_v3)")
