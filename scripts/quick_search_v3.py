# scripts/quick_search_v3.py
import sqlite3, numpy as np, textwrap
from s28.indexer import SentenceTFBackendMiniLM, _blob_to_np

DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"
TOPK = 5

def embed_q(q: str):
    be = SentenceTFBackendMiniLM("all-MiniLM-L6-v2")
    v = be.embed_one(q).astype(np.float32, copy=False)
    v /= (np.linalg.norm(v) + 1e-12)
    return v

conn = sqlite3.connect(DB); conn.row_factory = sqlite3.Row

meta = conn.execute("SELECT dim FROM embeddings_v3 LIMIT 1").fetchone()
assert meta, "No hay embeddings_v3"
dim = int(meta["dim"])

rows = conn.execute("""
  SELECT e.chunk_id, e.vector
  FROM embeddings_v3 e
  ORDER BY e.chunk_id
""").fetchall()

vecs = np.vstack([_blob_to_np(r["vector"], dim) for r in rows]).astype(np.float32)
vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
ids  = [int(r["chunk_id"]) for r in rows]

def title_page_snip(cid: int):
    r = conn.execute("""
      SELECT c.page, c.text, d.title, d.path
      FROM chunks_v3 c
      JOIN documents_v3 d ON d.id=c.doc_id
      WHERE c.chunk_id=?
    """, (cid,)).fetchone()
    return r

def run(q: str):
    qv = embed_q(q)
    sims = vecs @ qv
    order = np.argsort(-sims)[:TOPK]
    print(f"\nQ: {q}")
    for rank, i in enumerate(order, 1):
        cid = ids[i]
        r = title_page_snip(cid)
        title = r["title"] or r["path"].split("\\")[-1]
        page = r["page"]
        snip = textwrap.shorten((r["text"] or "").replace("\n"," "), width=300, placeholder="…")
        print(f"  {rank}. {title}  [p.{page if page else '-'}]  cid={cid}")
        print(f"     {snip}")

if __name__ == "__main__":
    # poné acá 2–3 consultas de prueba rápidas:
    run("redeterminación de precios en obra pública")
    run("impugnación de pliegos en contrataciones públicas")
    run("garantía de mantenimiento de oferta y su ejecución")
