# scripts/init_v3_schema.py
import sqlite3, datetime

DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"
now = datetime.datetime.utcnow().isoformat()

conn = sqlite3.connect(DB)
cur = conn.cursor()

# documents_v3
cur.execute("""
CREATE TABLE IF NOT EXISTS documents_v3 (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE,
  title TEXT,
  author TEXT,
  issuer TEXT,
  year INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""")

# chunks_v3
cur.execute("""
CREATE TABLE IF NOT EXISTS chunks_v3 (
  chunk_id INTEGER PRIMARY KEY,
  doc_id INTEGER NOT NULL,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  text_norm TEXT,
  page INTEGER,
  offset_start INTEGER,
  offset_end INTEGER,
  law_refs TEXT,
  FOREIGN KEY(doc_id) REFERENCES documents_v3(id)
)
""")

# embeddings_v3
cur.execute("""
CREATE TABLE IF NOT EXISTS embeddings_v3 (
  chunk_id INTEGER PRIMARY KEY,
  model TEXT,
  dim INTEGER NOT NULL,
  vector BLOB NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(chunk_id) REFERENCES chunks_v3(chunk_id)
)
""")

# FTS para chunks_v3 (por text_norm)
cur.execute("""
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks_v3 USING fts5(
  text_norm,
  content='chunks_v3',
  content_rowid='chunk_id',
  tokenize='unicode61 remove_diacritics 2'
)
""")

# Triggers de sync FTS
cur.execute("""
CREATE TRIGGER IF NOT EXISTS chunks_v3_ai AFTER INSERT ON chunks_v3 BEGIN
  INSERT INTO fts_chunks_v3(rowid, text_norm) VALUES (new.chunk_id, new.text_norm);
END;
""")
cur.execute("""
CREATE TRIGGER IF NOT EXISTS chunks_v3_ad AFTER DELETE ON chunks_v3 BEGIN
  INSERT INTO fts_chunks_v3(fts_chunks_v3, rowid, text_norm) VALUES('delete', old.chunk_id, old.text_norm);
END;
""")
cur.execute("""
CREATE TRIGGER IF NOT EXISTS chunks_v3_au AFTER UPDATE ON chunks_v3 BEGIN
  INSERT INTO fts_chunks_v3(fts_chunks_v3, rowid, text_norm) VALUES('delete', old.chunk_id, old.text_norm);
  INSERT INTO fts_chunks_v3(rowid, text_norm) VALUES (new.chunk_id, new.text_norm);
END;
""")

conn.commit()
conn.close()
print("OK: esquema V3 asegurado (documents_v3, chunks_v3, embeddings_v3, fts_chunks_v3).")
