-- === Nuevas columnas en documents ===
ALTER TABLE documents ADD COLUMN jurisdiction TEXT;
ALTER TABLE documents ADD COLUMN norm_level TEXT;
ALTER TABLE documents ADD COLUMN date_issued TEXT;  -- ISO YYYY-MM-DD
ALTER TABLE documents ADD COLUMN issuer TEXT;       -- CSJN, ONC, etc.

-- === Nuevas columnas en document_chunks ===
ALTER TABLE document_chunks ADD COLUMN page_num INTEGER;
ALTER TABLE document_chunks ADD COLUMN section TEXT;      -- texto de sección
ALTER TABLE document_chunks ADD COLUMN chunk_type TEXT;   -- VISTOS/CONSIDERANDOS/RESUELVE/FALLO/ANEXO/OTROS
ALTER TABLE document_chunks ADD COLUMN keywords TEXT;     -- coma-separadas
ALTER TABLE document_chunks ADD COLUMN text_norm TEXT;    -- normalizado (lower, sin tildes, sin underscores)
ALTER TABLE document_chunks ADD COLUMN law_refs TEXT;     -- JSON con refs a normas
ALTER TABLE document_chunks ADD COLUMN offset_start INTEGER;
ALTER TABLE document_chunks ADD COLUMN offset_end INTEGER;

-- === Embeddings (si no existiera) ===
CREATE TABLE IF NOT EXISTS embeddings(
  chunk_id INTEGER PRIMARY KEY REFERENCES document_chunks(id),
  model TEXT,
  dim INTEGER,
  vector BLOB,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- === Feedback del usuario / aprendizaje ligero ===
CREATE TABLE IF NOT EXISTS relevance_feedback(
  id INTEGER PRIMARY KEY,
  query TEXT,
  chunk_id INTEGER REFERENCES document_chunks(id),
  label INTEGER,         -- -1, 0, +1
  reason TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- === Caché de rerank ===
CREATE TABLE IF NOT EXISTS rerank_cache(
  query_hash TEXT,
  chunk_id INTEGER,
  score REAL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(query_hash, chunk_id)
);

-- === Sinónimos / alias para expansión de consulta ===
CREATE TABLE IF NOT EXISTS synonyms(
  canonical TEXT,
  variant TEXT
);

-- === FTS5 para búsqueda léxica (sobre text_norm) ===
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks
USING fts5(
  text_norm,
  chunk_id UNINDEXED,
  content='document_chunks',
  content_rowid='id',
  tokenize='unicode61 remove_diacritics 2'
);

-- Triggers de sincronización FTS5
CREATE TRIGGER IF NOT EXISTS fts_chunks_ai AFTER INSERT ON document_chunks BEGIN
  INSERT INTO fts_chunks(rowid, text_norm, chunk_id) VALUES (new.id, new.text_norm, new.id);
END;
CREATE TRIGGER IF NOT EXISTS fts_chunks_ad AFTER DELETE ON document_chunks BEGIN
  INSERT INTO fts_chunks(fts_chunks, rowid, text_norm, chunk_id) VALUES('delete', old.id, old.text_norm, old.id);
END;
CREATE TRIGGER IF NOT EXISTS fts_chunks_au AFTER UPDATE ON document_chunks BEGIN
  INSERT INTO fts_chunks(fts_chunks, rowid, text_norm, chunk_id) VALUES('delete', old.id, old.text_norm, old.id);
  INSERT INTO fts_chunks(rowid, text_norm, chunk_id) VALUES (new.id, new.text_norm, new.id);
END;

-- === Índices útiles ===
CREATE INDEX IF NOT EXISTS idx_docs_date       ON documents(date_issued);
CREATE INDEX IF NOT EXISTS idx_docs_issuer     ON documents(issuer);
CREATE INDEX IF NOT EXISTS idx_chunks_docid    ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_type     ON document_chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_page     ON document_chunks(page_num);
