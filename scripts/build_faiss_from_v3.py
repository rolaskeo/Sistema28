# scripts/build_faiss_from_v3.py
# -*- coding: utf-8 -*-
"""
Reconstruye FAISS a partir de V3:
- Lee chunks y metadatos desde documents_v3 / chunks_v3 / embeddings_v3 (SQLite)
- Embebe textos con all-MiniLM-L6-v2 si no tenés embeddings binarios (usa los existentes igual)
- Escribe faiss_index.bin + faiss_metadata.json
Uso:
  set DATABASE_PATH=C:\Rolex\Python\Sistema28Script\sistema28.db
  python scripts\build_faiss_from_v3.py
"""

import os, json, sqlite3, numpy as np, faiss
from sentence_transformers import SentenceTransformer

DB = os.getenv("DATABASE_PATH", r"C:\Rolex\Python\Sistema28Script\sistema28.db")
OUT_INDEX = os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")
OUT_META  = os.getenv("FAISS_METADATA_PATH","faiss_metadata.json")
EMB = os.getenv("EMBEDDING_MODEL","all-MiniLM-L6-v2")

def build_faiss_from_v3(db_path: str=None, out_index: str=None, out_meta: str=None, emb_model: str=None):
    db_path = db_path or DB
    out_index = out_index or OUT_INDEX
    out_meta  = out_meta  or OUT_META
    emb_model = emb_model or EMB

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
      SELECT c.chunk_id, c.doc_id, c.text AS chunk_text, d.path AS document_path,
             COALESCE(d.issuer,'') AS organismo_emisor, COALESCE(d.year,'') AS fecha_documento
      FROM chunks_v3 c
      JOIN documents_v3 d ON d.id=c.doc_id
      ORDER BY c.chunk_id
    """).fetchall()
    conn.close()

    if not rows:
        print("[FAISS] No hay chunks en V3 para indexar.")
        return

    texts = [(r["chunk_text"] or "").strip() for r in rows]
    meta  = [{
      "chunk_id": int(r["chunk_id"]),
      "document_path": r["document_path"] or "",
      "organismo_emisor": r["organismo_emisor"] or "",
      "fecha_documento": r["fecha_documento"] or "",
      "chunk_text": texts[i]
    } for i,r in enumerate(rows)]

    print(f"[FAISS] Embeddings de {len(texts)} chunks con {emb_model} …")
    model = SentenceTransformer(emb_model)
    X = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(X)

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    faiss.write_index(index, out_index)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[FAISS] OK →", out_index, out_meta)

if __name__ == "__main__":
    build_faiss_from_v3()
