import sqlite3, json

DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"

con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row
cur = con.cursor()

# 1) Mostrar las columnas reales de documents_v3
cols = [r[1] for r in cur.execute("PRAGMA table_info(documents_v3)").fetchall()]
print("columns:", cols)

# 2) Inspeccionar el doc cuyo título empieza con 10ba1a3e191e_
row = cur.execute(
    "select id, title, tipo, tipo_documento, tipo_doc, metadata "
    "from documents_v3 where title like ? limit 1",
    ("10ba1a3e191e_%",)
).fetchone()
print(dict(row) if row else None)

con.close()
