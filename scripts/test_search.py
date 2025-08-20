import sqlite3

DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"
Q  = "concesion"   # cambialo por algo que seguro exista: 'contrato', 'vial', 'decreto', etc.

con = sqlite3.connect(DB)
cur = con.cursor()

sql = """
SELECT dc.id, substr(dc.chunk_text,1,180)
FROM fts_chunks f
JOIN document_chunks dc ON dc.id = f.rowid
WHERE f.text_norm MATCH ?
LIMIT 5;
"""
cur.execute(sql, (Q,))
rows = cur.fetchall()
con.close()

if not rows:
    print("(sin resultados, probá cambiando Q en el script)")
else:
    for rid, prev in rows:
        print(f"- chunk {rid}: {prev.replace('\\n',' ')}")
