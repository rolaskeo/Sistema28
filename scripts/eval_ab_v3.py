# scripts/eval_ab_v3.py
# -*- coding: utf-8 -*-
import json, sqlite3, numpy as np, math, sys, os
from typing import List, Tuple, Optional

# === FIX DE IMPORTS ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from s28.indexer import SentenceTFBackendMiniLM, _blob_to_np

DB_PATH   = r"C:\Rolex\Python\Sistema28Script\sistema28.db"
QUERIES_JSONL = r".\queries_eval_onc.jsonl"

# Candidatos para OLD
EMB_OLD_CANDIDATES      = ["embeddings", "chunk_embeddings"]  # autodetect
DOCS_OLD                = "documents"                         # doc-level

# V3 (nuevo, chunk-level)
CHUNKS_V3               = "chunks_v3"
CHUNK_PK_V3             = "chunk_id"
EMB_V3                  = "embeddings_v3"
DOCS_V3                 = "documents_v3"

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

# --- Encoders por esquema ---
def _pick_model_from_name_or_dim(model_name: Optional[str], dim: int) -> str:
    """
    Elegimos SIEMPRE por dimensión para garantizar compatibilidad con los vectores.
    Ignoramos el nombre guardado si contradice el dim.
    """
    if dim == 384:
        return "all-MiniLM-L6-v2"
    if dim == 768:
        return "all-mpnet-base-v2"
    # Fallback razonable si aparece otra dim
    return "all-MiniLM-L6-v2"

def _make_encoder(model_name: str) -> SentenceTFBackendMiniLM:
    return SentenceTFBackendMiniLM(model_name)

def _table_has_columns(conn, table: str, required: list[str]) -> bool:
    try:
        cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]
        return all(c in cols for c in required)
    except sqlite3.OperationalError:
        return False

def _pick_old_embeddings_table(conn) -> Tuple[str, bool]:
    """
    Devuelve (tabla_embeddings, has_dim_col) para OLD.
    Debe tener columnnas: chunk_id, vector. dim es opcional.
    """
    for t in EMB_OLD_CANDIDATES:
        if _table_has_columns(conn, t, ["chunk_id", "vector"]):
            has_dim = _table_has_columns(conn, t, ["dim"])
            return t, has_dim
    raise RuntimeError("No encontré embeddings viejos válidos (ni 'embeddings' ni 'chunk_embeddings').")

# === MATRICES ===

def _matrix_docs_old(conn) -> Tuple[Optional[np.ndarray], List[int], str, int, Optional[str]]:
    """
    OLD (doc-level): JOIN embeddings e (chunk_id=documents.id) -> documents d
    Devuelve (mat_norm, doc_ids, emb_used, dim, model_name)
    """
    emb_used, has_dim = _pick_old_embeddings_table(conn)

    # Leer una muestra para obtener dim y (si existe) model
    meta = conn.execute(
        f"SELECT {'dim,' if has_dim else ''} COALESCE(model,'') AS model, LENGTH(vector) AS vlen FROM {emb_used} LIMIT 1"
    ).fetchone()
    if not meta:
        return None, [], emb_used, 0, None

    if has_dim:
        dim = int(meta["dim"])
    else:
        dim = int(meta["vlen"]) // 4  # float32

    model_name = meta["model"] if "model" in meta.keys() else None
    # Construir matriz
    rows = conn.execute(f"""
        SELECT e.chunk_id AS doc_id, e.vector
        FROM {emb_used} e
        JOIN {DOCS_OLD} d ON d.id = e.chunk_id
        ORDER BY e.chunk_id
    """).fetchall()
    if not rows:
        return None, [], emb_used, dim, model_name

    vecs = [_blob_to_np(r["vector"], dim) for r in rows]
    mat = np.vstack(vecs).astype(np.float32)
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    ids = [int(r["doc_id"]) for r in rows]
    return mat, ids, emb_used, dim, model_name

def _matrix_chunks_v3(conn) -> Tuple[Optional[np.ndarray], List[int], int, Optional[str]]:
    # Obtener meta de v3 (dim/model) si existe
    meta = conn.execute(
        f"SELECT dim, COALESCE(model,'') AS model FROM {EMB_V3} LIMIT 1"
    ).fetchone()
    if not meta:
        return None, [], 0, None
    dim = int(meta["dim"])
    model_name = meta["model"]

    rows = conn.execute(f"""
        SELECT e.chunk_id, e.vector
        FROM {EMB_V3} e
        JOIN {CHUNKS_V3} c ON c.{CHUNK_PK_V3} = e.chunk_id
        ORDER BY e.chunk_id
    """).fetchall()
    if not rows:
        return None, [], dim, model_name

    vecs = [_blob_to_np(r["vector"], dim) for r in rows]
    mat = np.vstack(vecs).astype(np.float32)
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    ids = [int(r["chunk_id"]) for r in rows]
    return mat, ids, dim, model_name

# === MÉTRICAS ===

def _topk_from_mat(qv: np.ndarray, mat: np.ndarray, ids: List[int], k: int):
    sims = mat @ qv
    order = np.argsort(-sims)[:k]
    return [ids[i] for i in order], [float(sims[i]) for i in order]

def _recall_at_k(pred: List[int], rel: List[int], k: int) -> float:
    srel = set(rel)
    return len([x for x in pred[:k] if x in srel]) / max(1, len(srel))

def _mrr_at_k(pred: List[int], rel: List[int], k: int) -> float:
    srel = set(rel)
    for i, x in enumerate(pred[:k], 1):
        if x in srel:
            return 1.0 / i
    return 0.0

def _ndcg_at_k(pred: List[int], rel: List[int], k: int) -> float:
    srel = set(rel)
    dcg = 0.0
    for i in range(1, k+1):
        if i-1 < len(pred) and pred[i-1] in srel:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(srel)) + 1))
    return (dcg / ideal) if ideal > 0 else 0.0

# === TITULOS ===

def _title_for_doc_old(conn, doc_id: int) -> str:
    r = conn.execute(f"SELECT titulo, path FROM {DOCS_OLD} WHERE id=?", (doc_id,)).fetchone()
    if not r:
        return "(desconocido)"
    return r["titulo"] or (r["path"] or "(s/título)")

def _title_for_chunk_v3(conn, chunk_id: int) -> str:
    r = conn.execute(f"""
        SELECT d.title
        FROM {CHUNKS_V3} c
        JOIN {DOCS_V3} d ON d.id = c.doc_id
        WHERE c.{CHUNK_PK_V3}=?
    """, (chunk_id,)).fetchone()
    return (r["title"] if r and r["title"] else "(s/título)")

def _first_hit_rank(pred: List[int], relevant: List[int]) -> int:
    srel = set(relevant)
    for i, x in enumerate(pred, 1):
        if x in srel:
            return i
    return 0

# === MAIN ===

def main():
    conn = _conn()
    K = 5
    items = [json.loads(l) for l in open(QUERIES_JSONL, "r", encoding="utf-8")]

    # OLD (doc-level): matriz + encoder acorde al modelo/dim OLD
    try:
        mat_old, doc_ids_old, emb_used, dim_old, model_old = _matrix_docs_old(conn)
        has_old = mat_old is not None and len(doc_ids_old) > 0
        model_name_old = _pick_model_from_name_or_dim(model_old, dim_old)
        enc_old = _make_encoder(model_name_old)
        print(f"[OLD] usando embeddings: {emb_used} (doc-level) | dim={dim_old} | model='{model_name_old}'")
    except Exception as e:
        mat_old, doc_ids_old, has_old = None, [], False
        enc_old = None
        print(f"[OLD] sin datos ({e})")

    # V3 (chunk-level): si existe, matriz + encoder por modelo/dim V3
    try:
        mat_v3, chunk_ids_v3, dim_v3, model_v3 = _matrix_chunks_v3(conn)
        has_v3 = mat_v3 is not None and len(chunk_ids_v3) > 0
        if has_v3:
            model_name_v3 = _pick_model_from_name_or_dim(model_v3, dim_v3)
        else:
            model_name_v3 = "all-MiniLM-L6-v2"
        enc_v3 = _make_encoder(model_name_v3)
        if has_v3:
            print(f"[V3 ] usando embeddings_v3 (chunk-level) | dim={dim_v3} | model='{model_name_v3}'")
        else:
            print("[V3 ] tablas v3 aún no existen")
    except sqlite3.OperationalError:
        mat_v3, chunk_ids_v3, has_v3 = None, [], False
        enc_v3 = None
        print("[V3 ] tablas v3 aún no existen")

    agg_old = {"R": [], "MRR": [], "nDCG": []}
    agg_v3  = {"R": [], "MRR": [], "nDCG": []}

    print("=== Evaluación por consulta (Top-1 y rank del primer relevante) ===")
    for it in items:
        q = it["q"]

        # --- OLD (doc-level) ---
        rel_doc_old = []
        if "relevant_doc_id_old" in it and it["relevant_doc_id_old"]:
            rel_doc_old = [int(it["relevant_doc_id_old"])]

        # --- V3 (chunk-level) ---
        rel_chunks_v3 = it.get("relevant_chunk_ids_v3", [])

        # Embeddings de consulta con encoder adecuado a cada esquema
        if has_old:
            qv_old = enc_old.embed_one(q).astype(np.float32, copy=False)
            qv_old /= (np.linalg.norm(qv_old) + 1e-12)
        if has_v3:
            qv_v3 = enc_v3.embed_one(q).astype(np.float32, copy=False)
            qv_v3 /= (np.linalg.norm(qv_v3) + 1e-12)

        # OLD (doc-level)
        top_old_docs, rank_old, top_old_title = ([], 0, "(sin datos)")
        if has_old:
            top_old_docs, _ = _topk_from_mat(qv_old, mat_old, doc_ids_old, K)
            rank_old = _first_hit_rank(top_old_docs, rel_doc_old)
            if top_old_docs:
                top_old_title = _title_for_doc_old(conn, top_old_docs[0])

            agg_old["R"].append(_recall_at_k(top_old_docs, rel_doc_old, K))
            agg_old["MRR"].append(_mrr_at_k(top_old_docs, rel_doc_old, K))
            agg_old["nDCG"].append(_ndcg_at_k(top_old_docs, rel_doc_old, K))

        # V3 (chunk-level)
        top_v3_chunks, rank_v3, top_v3_title = ([], 0, "(sin datos)")
        if has_v3:
            top_v3_chunks, _ = _topk_from_mat(qv_v3, mat_v3, chunk_ids_v3, K)
            rank_v3 = _first_hit_rank(top_v3_chunks, rel_chunks_v3)
            if top_v3_chunks:
                top_v3_title = _title_for_chunk_v3(conn, top_v3_chunks[0])

            agg_v3["R"].append(_recall_at_k(top_v3_chunks, rel_chunks_v3, K))
            agg_v3["MRR"].append(_mrr_at_k(top_v3_chunks, rel_chunks_v3, K))
            agg_v3["nDCG"].append(_ndcg_at_k(top_v3_chunks, rel_chunks_v3, K))

        print(f"- Q: {q}")
        if has_old:
            print(f"  OLD(doc): top1='{top_old_title}' | rank_rel@5={rank_old if rank_old>0 else '—'}")
        else:
            print("  OLD(doc): (sin embeddings)")
        if has_v3:
            print(f"  V3(chunk): top1='{top_v3_title}' | rank_rel@5={rank_v3 if rank_v3>0 else '—'}")
        else:
            print("  V3(chunk): (v3 aún no indexado)")

    def _avg(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    print("\n=== Promedios ===")
    if has_old:
        print("OLD(doc):", {k: round(_avg(v), 4) for k, v in agg_old.items()})
    else:
        print("OLD(doc): (sin embeddings)")

    if has_v3:
        print("V3(chunk):", {k: round(_avg(v), 4) for k, v in agg_v3.items()})
    else:
        print("V3(chunk): (v3 aún no indexado)")

if __name__ == "__main__":
    main()
