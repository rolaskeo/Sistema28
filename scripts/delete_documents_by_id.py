# scripts/delete_documents_by_id.py
import sqlite3

DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"
IDS = [7, 8]

conn = sqlite3.connect(DB)
conn.execute("PRAGMA foreign_keys=OFF")  # por si no hay FKs definidos
conn.row_factory = sqlite3.Row

def get_paths(ids):
    q = f"SELECT id, path FROM documents WHERE id IN ({','.join(['?']*len(ids))})"
    return conn.execute(q, ids).fetchall()

def get_chunk_ids_by_paths(paths):
    q = f"SELECT id FROM document_chunks WHERE document_path IN ({','.join(['?']*len(paths))})"
    return [r["id"] for r in conn.execute(q, paths)]

ids = [int(x) for x in IDS]
rows = get_paths(ids)
paths = [r["path"] for r in rows]

# 1) embeddings “viejos” a nivel documento
if ids:
    conn.execute(f"DELETE FROM embeddings WHERE chunk_id IN ({','.join(['?']*len(ids))})", ids)

# 2) chunks viejos del/los documento(s)
if paths:
    conn.execute(f"DELETE FROM document_chunks WHERE document_path IN ({','.join(['?']*len(paths))})", paths)

# 3) fts (si está con content='document_chunks', se mantiene solo/auto; si no, limpiamos por las dudas)
try:
    conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild')")
except Exception:
    pass

# 4) documentos
if ids:
    conn.execute(f"DELETE FROM documents WHERE id IN ({','.join(['?']*len(ids))})", ids)

conn.commit()
print(f"OK: borrados documents {ids} y asociados (embeddings/doc_chunks/fts).")
