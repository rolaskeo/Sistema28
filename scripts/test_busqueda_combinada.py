# scripts/test_busqueda_combinada.py
# Verificador simple de búsqueda COMBINADA (FTS5 + FAISS) para Sistema 28
# Uso:
#   python scripts/test_busqueda_combinada.py "concesion vial ley 17.520" --k 10 --wsem 0.35
# Requisitos: sentence-transformers, faiss-cpu, python-dotenv (opcional)

from __future__ import annotations
import os, json, sqlite3, argparse, re
from pathlib import Path
from typing import List, Dict, Any, Tuple

# --- .env (opcional) ---
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k,v=line.split('=',1); os.environ.setdefault(k.strip(), v.strip())

DB_PATH   = os.environ.get("DATABASE_PATH", "sistema28.db")
FAISS_BIN = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_META= os.environ.get("FAISS_METADATA_PATH", "faiss_metadata.json")
EMB_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMB_DEVICE= os.environ.get("EMBEDDING_DEVICE", "cpu")

# --- Embeddings de consulta (local, CPU) ---
from sentence_transformers import SentenceTransformer
import numpy as np

_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMB_MODEL, device=EMB_DEVICE)
    return _model

def embed_query(q: str) -> np.ndarray:
    vec = get_model().encode([q], normalize_embeddings=True)
    return np.asarray(vec[0], dtype=np.float32)

# --- FAISS ---
import faiss

def load_faiss() -> Tuple[faiss.Index, List[int]]:
    if not (Path(FAISS_BIN).exists() and Path(FAISS_META).exists()):
        raise FileNotFoundError("Faltan faiss_index.bin o faiss_metadata.json — corré crear_indice_faiss.py")

    index = faiss.read_index(FAISS_BIN)
    meta_raw = Path(FAISS_META).read_text(encoding="utf-8")
    try:
        meta = json.loads(meta_raw)
    except Exception as e:
        raise ValueError(f"No pude leer {FAISS_META}: {e}")

    ids: List[int] = []

    if isinstance(meta, dict):
        # formatos típicos
        if "ids" in meta and isinstance(meta["ids"], list):
            ids = [int(x) for x in meta["ids"]]
        elif "index_to_chunk_id" in meta and isinstance(meta["index_to_chunk_id"], list):
            ids = [int(x) for x in meta["index_to_chunk_id"]]
        elif "mapping" in meta and isinstance(meta["mapping"], dict):
            # mapping: {"0": 123, "1": 456, ...}
            m = meta["mapping"]
            # construir lista ordenada por posición
            maxpos = index.ntotal
            ids = [0]*maxpos
            for k,v in m.items():
                if str(k).isdigit():
                    p = int(k)
                    if 0 <= p < maxpos:
                        ids[p] = int(v)
        elif all(k.isdigit() for k in meta.keys()):
            # dict con claves numéricas como posiciones
            maxpos = index.ntotal
            ids = [0]*maxpos
            for k,v in meta.items():
                p = int(k)
                if 0 <= p < maxpos:
                    ids[p] = int(v)
        else:
            raise ValueError("Estructura de metadata FAISS no reconocida en dict (esperaba 'ids', 'index_to_chunk_id' o mapping por posición).")

    elif isinstance(meta, list):
        if all(isinstance(x, (int, str)) for x in meta):
            ids = [int(x) for x in meta]
        elif meta and all(isinstance(x, dict) for x in meta):
            # lista de dicts -> intentar keys frecuentes
            cand_keys = ("chunk_id", "id", "doc_chunk_id")
            key = next((k for k in cand_keys if k in meta[0]), None)
            if key is None:
                raise ValueError(f"Lista de dicts en metadata, pero no encuentro ninguna de {cand_keys} como clave.")
            ids = [int(x[key]) for x in meta]
        else:
            raise ValueError("Estructura de metadata FAISS no reconocida en lista.")

    else:
        raise ValueError("Estructura de metadata FAISS no reconocida.")

    if index.ntotal != len(ids):
        print(f"[warn] index.ntotal={index.ntotal} != len(ids)={len(ids)} — metadata e índice pueden estar fuera de sync.")

    return index, ids

# --- FTS5 ---
def _sanitize_for_fts(query: str) -> str:
    # Tokenizar: palabras con acentos y números; quitar símbolos (como '.').
    terms = re.findall(r"[\wáéíóúüñÁÉÍÓÚÑ]+", query, flags=re.UNICODE)
    if not terms:
        return ""
    # Cada término entre comillas para evitar problemas con operadores
    return " OR ".join(f'"{t}"' for t in terms)

def fts_search(query: str, k: int = 10) -> List[Dict[str, Any]]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    fts_q = _sanitize_for_fts(query)
    if not fts_q:
        return []
    try:
        cur.execute(
            """
            SELECT dc.id, dc.document_id, dc.titulo, dc.chunk_text, dc.chunk_type, dc.page_num,
                   1.0/(1.0 + bm25(fts_chunks)) as score
            FROM fts_chunks
            JOIN document_chunks dc ON dc.id = fts_chunks.rowid
            WHERE fts_chunks MATCH ?
            ORDER BY score DESC
            LIMIT ?
            """,
            (fts_q, k),
        )
    except Exception:
        cur.execute(
            """
            SELECT dc.id, dc.document_id, dc.titulo, dc.chunk_text, dc.chunk_type, dc.page_num,
                   1.0 as score
            FROM fts_chunks
            JOIN document_chunks dc ON dc.id = fts_chunks.rowid
            WHERE fts_chunks MATCH ?
            LIMIT ?
            """,
            (fts_q, k),
        )
    rows = cur.fetchall(); con.close()

    out = []
    for (cid, did, titulo, text, ctype, page, score) in rows:
        out.append({
            "id": int(cid), "document_id": int(did) if did is not None else None,
            "titulo": titulo, "preview": (text or "")[:220].replace("\n"," "),
            "chunk_type": ctype, "page_num": page, "score": float(score),
            "src": "FTS",
        })
    return out

# --- FAISS search ---
def faiss_search(query: str, k: int = 10) -> List[Dict[str, Any]]:
    index, ids = load_faiss()
    q = embed_query(query)
    D, I = index.search(np.expand_dims(q, 0), k)
    pos = I[0].tolist(); sims = D[0].tolist()

    results = []
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    for rank, (p, s) in enumerate(zip(pos, sims), start=1):
        if p < 0 or p >= len(ids):
            continue
        cid = int(ids[p])
        cur.execute("SELECT document_id, titulo, chunk_text, chunk_type, page_num FROM document_chunks WHERE id=?", (cid,))
        r = cur.fetchone()
        if not r: continue
        did, titulo, text, ctype, page = r
        results.append({
            "id": cid, "document_id": int(did) if did is not None else None,
            "titulo": titulo, "preview": (text or "")[:220].replace("\n"," "),
            "chunk_type": ctype, "page_num": page, "score": float(s),
            "src": "FAISS",
        })
    con.close()
    return results

# --- Fusión simple ---
def fuse(fts: List[Dict[str, Any]], faiss: List[Dict[str, Any]], w_sem: float = 0.35, k: int = 10) -> List[Dict[str, Any]]:
    def norm(xs: List[float]) -> List[float]:
        if not xs: return []
        lo, hi = min(xs), max(xs)
        if hi <= lo: return [1.0 for _ in xs]
        return [(x-lo)/(hi-lo) for x in xs]

    by_id: Dict[int, Dict[str, Any]] = {}

    if fts:
        ns = norm([it["score"] for it in fts])
        for it, s in zip(fts, ns):
            row = dict(it)
            row["score_fts"] = s; row.setdefault("score_faiss", 0.0)
            by_id[row["id"]] = row

    if faiss:
        ns = norm([it["score"] for it in faiss])
        for it, s in zip(faiss, ns):
            prev = by_id.get(it["id"]) or dict(it)
            prev["score_faiss"] = s
            prev.setdefault("score_fts", 0.0)
            for k2 in ("titulo","preview","chunk_type","page_num","document_id"):
                prev.setdefault(k2, it.get(k2))
            by_id[prev["id"]] = prev

    merged = []
    for it in by_id.values():
        s = (1.0 - w_sem) * it.get("score_fts", 0.0) + w_sem * it.get("score_faiss", 0.0)
        it["score_combined"] = s
        merged.append(it)

    merged.sort(key=lambda x: x["score_combined"], reverse=True)
    return merged[:k]

# --- Pretty print ---
def show(results: List[Dict[str, Any]]):
    if not results:
        print("(sin resultados)"); return
    for i, it in enumerate(results, start=1):
        print(f"{i:02d}. [chunk {it['id']}] ({it.get('chunk_type')}, pág {it.get('page_num')}) score={it['score_combined']:.3f}")
        print(f"    {it.get('titulo')}")
        print(f"    {it.get('preview')}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prueba de búsqueda combinada (FTS + FAISS)")
    ap.add_argument("query", help="consulta")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--wsem", type=float, default=0.35, help="peso FAISS en score combinado [0..1]")
    args = ap.parse_args()

    print("[FTS] ...")
    fts = fts_search(args.query, k=args.k)
    print(f"  {len(fts)} items FTS")

    print("[FAISS] ...")
    fa = faiss_search(args.query, k=args.k)
    print(f"  {len(fa)} items FAISS")

    print("\n[FUSIÓN] ...")
    fused = fuse(fts, fa, w_sem=args.wsem, k=args.k)
    show(fused)
