import sqlite3
DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"
con = sqlite3.connect(DB)
cur = con.cursor()

def count(tbl):
    try:
        cur.execute(f"SELECT COUNT(*) FROM {tbl}")
        return cur.fetchone()[0]
    except Exception as e:
        return f"error: {e}"

for t in ["documents","document_chunks","document_chunks_backup","embeddings","relevance_feedback"]:
    print(t, count(t))

con.close()
