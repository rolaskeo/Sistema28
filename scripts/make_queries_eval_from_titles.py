# scripts/make_queries_eval_from_titles.py
import json, sqlite3

DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"
IN  = r".\queries_by_title.jsonl"   # {"q":"...", "relevant_doc_title_old":"...", "relevant_doc_title_v3":"..."}
OUT = r".\queries_eval.jsonl"

conn = sqlite3.connect(DB); conn.row_factory = sqlite3.Row

def doc_id_by_title(table_docs: str, title: str, title_cols=("titulo","title")):
    # intenta columnas típicas
    for col in title_cols:
        try:
            r = conn.execute(f"SELECT id FROM {table_docs} WHERE {col} = ?", (title,)).fetchone()
            if r: return int(r["id"])
        except Exception:
            pass
    # fallback LIKE en ambas
    like = f"%{title}%"
    qparts = " OR ".join([f"{c} LIKE ?" for c in title_cols])
    try:
        r = conn.execute(f"SELECT id FROM {table_docs} WHERE {qparts} ORDER BY id LIMIT 1", (like,)*len(title_cols)).fetchone()
        return int(r["id"]) if r else None
    except Exception:
        return None

def chunk_ids_for_doc_old(doc_id: int):
    # viejo: document_chunks.id y FK document_id
    return [int(r["id"]) for r in conn.execute(
        "SELECT id FROM document_chunks WHERE document_id=? ORDER BY chunk_index, id", (doc_id,)
    )]

def chunk_ids_for_doc_v3(doc_id: int):
    # v3: chunks_v3.chunk_id y FK doc_id
    return [int(r["chunk_id"]) for r in conn.execute(
        "SELECT chunk_id FROM chunks_v3 WHERE doc_id=? ORDER BY chunk_index, chunk_id", (doc_id,)
    )]

items = [json.loads(l) for l in open(IN, "r", encoding="utf-8")]
out = []

for it in items:
    q = it["q"]
    row = {"q": q}

    t_old = it.get("relevant_doc_title_old")
    if t_old:
        did = doc_id_by_title("documents", t_old, title_cols=("titulo","title"))
        if did:
            row["relevant_chunk_ids_old"] = chunk_ids_for_doc_old(did)

    t_v3 = it.get("relevant_doc_title_v3")
    if t_v3:
        did = doc_id_by_title("documents_v3", t_v3, title_cols=("title","titulo"))
        if did:
            row["relevant_chunk_ids_v3"] = chunk_ids_for_doc_v3(did)

    out.append(row)

with open(OUT, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("OK ->", OUT, "(", len(out), "queries )")
