import sqlite3

DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"

def has_table(cur, name):
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;", (name,))
    return cur.fetchone() is not None

def has_column(cur, table, col):
    cur.execute(f"PRAGMA table_info({table});")
    return any(r[1] == col for r in cur.fetchall())

def exec_many(con, stmts):
    cur = con.cursor()
    for s in stmts:
        cur.execute(s)
    con.commit()

con = sqlite3.connect(DB)
cur = con.cursor()

# --- documents: columnas nuevas ---
if has_table(cur, "documents"):
    for col in ("jurisdiction","norm_level","date_issued","issuer"):
        if not has_column(cur, "documents", col):
            cur.execute(f"ALTER TABLE documents ADD COLUMN {col} TEXT;")
    con.commit()

# --- document_chunks: columnas nuevas ---
if has_table(cur, "document_chunks"):
    add_cols = [
        ("page_num","INTEGER"),
        ("section","TEXT"),
        ("chunk_type","TEXT"),
        ("keywords","TEXT"),
        ("text_norm","TEXT"),
        ("law_refs","TEXT"),
        ("offset_start","INTEGER"),
        ("offset_end","INTEGER"),
    ]
    for col, ctype in add_cols:
        if not has_column(cur, "document_chunks", col):
            cur.execute(f"ALTER TABLE document_chunks ADD COLUMN {col} {ctype};")
    con.commit()

# --- tablas auxiliares ---
exec_many(con, [
    """CREATE TABLE IF NOT EXISTS embeddings(
         chunk_id INTEGER PRIMARY KEY REFERENCES document_chunks(id),
         model TEXT, dim INTEGER, vector BLOB,
         created_at TEXT DEFAULT CURRENT_TIMESTAMP
       );""",
    """CREATE TABLE IF NOT EXISTS relevance_feedback(
         id INTEGER PRIMARY KEY,
         query TEXT,
         chunk_id INTEGER REFERENCES document_chunks(id),
         label INTEGER,
         reason TEXT,
         created_at TEXT DEFAULT CURRENT_TIMESTAMP
       );""",
    """CREATE TABLE IF NOT EXISTS rerank_cache(
         query_hash TEXT,
         chunk_id INTEGER,
         score REAL,
         created_at TEXT DEFAULT CURRENT_TIMESTAMP,
         PRIMARY KEY(query_hash, chunk_id)
       );""",
    """CREATE TABLE IF NOT EXISTS synonyms(
         canonical TEXT,
         variant TEXT
       );"""
])

# --- FTS5 (si no existe) ---
cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='fts_chunks';")
fts_exists = cur.fetchone() is not None
if not fts_exists:
    cur.execute("""
        CREATE VIRTUAL TABLE fts_chunks USING fts5(
            text_norm,
            content='document_chunks',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        );
    """)
    con.commit()

# --- triggers FTS5 ---
exec_many(con, [
    """CREATE TRIGGER IF NOT EXISTS fts_chunks_ai AFTER INSERT ON document_chunks BEGIN
           INSERT INTO fts_chunks(rowid, text_norm) VALUES (new.id, new.text_norm);
       END;""",
    """CREATE TRIGGER IF NOT EXISTS fts_chunks_ad AFTER DELETE ON document_chunks BEGIN
           INSERT INTO fts_chunks(fts_chunks, rowid, text_norm) VALUES('delete', old.id, old.text_norm);
       END;""",
    """CREATE TRIGGER IF NOT EXISTS fts_chunks_au AFTER UPDATE ON document_chunks BEGIN
           INSERT INTO fts_chunks(fts_chunks, rowid, text_norm) VALUES('delete', old.id, old.text_norm);
           INSERT INTO fts_chunks(rowid, text_norm) VALUES (new.id, new.text_norm);
       END;"""
])

# --- índices útiles ---
exec_many(con, [
    "CREATE INDEX IF NOT EXISTS idx_docs_date   ON documents(date_issued);",
    "CREATE INDEX IF NOT EXISTS idx_docs_issuer ON documents(issuer);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_docid ON document_chunks(document_id);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_type  ON document_chunks(chunk_type);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_page  ON document_chunks(page_num);",
])

con.close()
print("Migración segura aplicada ✅")
