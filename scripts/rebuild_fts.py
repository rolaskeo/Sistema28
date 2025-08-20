import sqlite3

DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"

con = sqlite3.connect(DB)
cur = con.cursor()

# 1) Borrar triggers viejos si existen
for trg in ("fts_chunks_ai", "fts_chunks_ad", "fts_chunks_au"):
    try:
        cur.execute(f"DROP TRIGGER IF EXISTS {trg};")
    except Exception:
        pass

# 2) Borrar tabla FTS y sus tablas sombra si existen
cur.execute("""
SELECT name FROM sqlite_master
WHERE name IN ('fts_chunks','fts_chunks_data','fts_chunks_idx','fts_chunks_docsize','fts_chunks_config');
""")
for (name,) in cur.fetchall():
    try:
        cur.execute(f"DROP TABLE IF EXISTS {name};")
        print(f"- Drop {name}")
    except Exception as e:
        print(f"! Error drop {name}: {e}")

con.commit()

# 3) Crear FTS5 nuevamente (content ligado a document_chunks.id)
cur.execute("""
CREATE VIRTUAL TABLE fts_chunks
USING fts5(
  text_norm,
  chunk_id UNINDEXED,
  content='document_chunks',
  content_rowid='id',
  tokenize='unicode61 remove_diacritics 2'
);
""")

# 4) Recrear triggers de sincronización
cur.execute("""
CREATE TRIGGER IF NOT EXISTS fts_chunks_ai AFTER INSERT ON document_chunks BEGIN
  INSERT INTO fts_chunks(rowid, text_norm, chunk_id) VALUES (new.id, new.text_norm, new.id);
END;""")
cur.execute("""
CREATE TRIGGER IF NOT EXISTS fts_chunks_ad AFTER DELETE ON document_chunks BEGIN
  INSERT INTO fts_chunks(fts_chunks, rowid, text_norm, chunk_id) VALUES('delete', old.id, old.text_norm, old.id);
END;""")
cur.execute("""
CREATE TRIGGER IF NOT EXISTS fts_chunks_au AFTER UPDATE ON document_chunks BEGIN
  INSERT INTO fts_chunks(fts_chunks, rowid, text_norm, chunk_id) VALUES('delete', old.id, old.text_norm, old.id);
  INSERT INTO fts_chunks(rowid, text_norm, chunk_id) VALUES (new.id, new.text_norm, new.id);
END;""")

# 5) Repoblar FTS con lo que haya hoy en document_chunks
cur.execute("SELECT id, COALESCE(text_norm,'') FROM document_chunks;")
rows = cur.fetchall()
cur.executemany(
    "INSERT INTO fts_chunks(rowid, text_norm, chunk_id) VALUES (?, ?, ?)",
    [(rid, txt, rid) for (rid, txt) in rows]
)

con.commit()
con.close()
print(f"FTS reconstruido ✅  Filas indexadas: {len(rows)}")
