# scripts/build_faiss_v3.py
import sqlite3
import faiss
import numpy as np
import json

DB_PATH = "sistema28.db"              # ajustá si tu DB está en otra ruta
INDEX_PATH = "faiss_index_v3.bin"
META_PATH = "faiss_metadata_v3.json"

def fetch_embeddings():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT chunk_id, dim, vector
        FROM embeddings_v3
        ORDER BY chunk_id
    """).fetchall()
    conn.close()

    if not rows:
        raise SystemExit("No hay filas en embeddings_v3")

    # armamos matriz
    dim = int(rows[0]["dim"])
    ids = np.array([int(r["chunk_id"]) for r in rows], dtype=np.int64)
    vecs = [np.frombuffer(r["vector"], dtype=np.float32) for r in rows]
    X = np.vstack(vecs).astype(np.float32)

    # por si acaso, normalizamos (ya venían normalizados, pero no molesta)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms
    return ids, X, dim

if __name__ == "__main__":
    print(">> Cargando embeddings_v3...")
    ids, X, dim = fetch_embeddings()
    print(f"   {len(ids)} embeddings, dim={dim}")

    print(">> Construyendo índice FAISS (IndexFlatIP)...")
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    print(f">> Guardando índice en {INDEX_PATH} y metadata en {META_PATH} ...")
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"dim": int(dim), "count": int(len(ids)), "ids": ids.tolist()}, f, ensure_ascii=False)

    print("OK: índice FAISS v3 creado.")
