import sqlite3

DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"
con = sqlite3.connect(DB)
cur = con.cursor()

# 1) borrar triggers viejos
for trg in ("fts_chunks_ai","fts_chunks_ad","fts_chunks_au"):
    try:
        cur.execute(f"DROP TRIGGER IF EXISTS {trg};")
    except:
        pass

# 2) borrar FTS y sombras
cur.execute("""
SELECT name FROM sqlite_master
WHERE name IN ('fts_chunks','fts_chunks_data','fts_chunks_idx','fts_chunks_docsize','fts_chunks_config');
""")
for (name,) in cur.fetchall():
    cur.execute(f"DROP TABLE IF EXISTS {name};")
con.commit()

# 3) crear FTS correcto (solo text_norm, content apuntando a document_chunks.id)
cur.execute("""
CREATE VIRTUAL TABLE fts_chunks USING fts5(
  text_norm,
  content='document_chunks',
  content_rowid='id',
  tokenize='unicode61 remove_diacritics 2'
);
""")

# 4) triggers correctos
cur.execute("""
CREATE TRIGGER fts_chunks_ai AFTER INSERT ON document_chunks BEGIN
  INSERT INTO fts_chunks(rowid, text_norm) VALUES (new.id, new.text_norm);
END;""")
cur.execute("""
CREATE TRIGGER fts_chunks_ad AFTER DELETE ON document_chunks BEGIN
  INSERT INTO fts_chunks(fts_chunks, rowid, text_norm) VALUES('delete', old.id, old.text_norm);
END;""")
cur.execute("""
CREATE TRIGGER fts_chunks_au AFTER UPDATE ON document_chunks BEGIN
  INSERT INTO fts_chunks(fts_chunks, rowid, text_norm) VALUES('delete', old.id, old.text_norm);
  INSERT INTO fts_chunks(rowid, text_norm) VALUES (new.id, new.text_norm);
END;""")

# 5) repoblar FTS con lo existente
cur.execute("SELECT id, COALESCE(text_norm,'') FROM document_chunks;")
rows = cur.fetchall()
cur.executemany(
    "INSERT INTO fts_chunks(rowid, text_norm) VALUES (?, ?)",
    rows
)

con.commit()
con.close()
print(f"FTS corregida e indexada ✅  Filas: {len(rows)}")
