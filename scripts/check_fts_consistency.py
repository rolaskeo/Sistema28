import sqlite3
DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"
con = sqlite3.connect(DB)
cur = con.cursor()

def cnt(q):
    cur.execute(q); return cur.fetchone()[0]

chunks = cnt("SELECT COUNT(*) FROM document_chunks")
fts    = cnt("SELECT COUNT(*) FROM fts_chunks")

print("document_chunks:", chunks)
print("fts_chunks:", fts)
print("OK ✅" if chunks == fts else "⚠️ FTS desalineado (reconstruir)")
con.close()
