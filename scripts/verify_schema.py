import sqlite3, pprint
DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"
con = sqlite3.connect(DB)
cur = con.cursor()

def cols(t):
    cur.execute(f"PRAGMA table_info({t});")
    return [r[1] for r in cur.fetchall()]

cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY 1;")
print("Tablas/Vistas:")
pprint.pp([r[0] for r in cur.fetchall()])

print("\ncolumns documents:")
print(cols("documents"))

print("\ncolumns document_chunks:")
print(cols("document_chunks"))

con.close()
print("\nOK ✅")
