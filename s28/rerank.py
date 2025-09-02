# s28/rerank.py
# -*- coding: utf-8 -*-
"""
Reranking híbrido para Sistema 28 (v3):
- Recall: FTS5 (fts_chunks_v3) -> top-N candidatos (tokens con AND).
- Relevancia semántica: coseno(query_embedding, chunk_embedding) desde embeddings_v3 (384d).
- Puntaje combinado: BM25 (proxy) + Coseno + Meta heurístico.
- Diversidad: MMR opcional sobre los vectores.
- Salida: resultados con (chunk_id, score, doc_id, title, page, snippet, path).

Esquema esperado:
  - documents_v3(id, path, title, author, issuer, year, ...)
  - chunks_v3(chunk_id, doc_id, chunk_index, text, text_norm, page, ...)
  - embeddings_v3(chunk_id, model, dim, vector BLOB)
  - fts_chunks_v3(text_norm, content='chunks_v3', content_rowid='chunk_id')
"""

from __future__ import annotations
import sqlite3, textwrap, re
from typing import List, Dict, Optional
import numpy as np

try:
    from unidecode import unidecode
except Exception:
    unidecode = None

from s28.indexer import SentenceTFBackendMiniLM
from functools import lru_cache

import os
from sentence_transformers import CrossEncoder

# --- Carga de .env y util de paths absolutos bajo la raíz del proyecto ---
from pathlib import Path
from dotenv import load_dotenv

# s28/ está dentro del proyecto; PROJ_ROOT es la carpeta raíz (donde vive tu .env)
PROJ_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJ_ROOT / ".env")

def _abs_under_root(p: str, default_name: str | None = None) -> str:
    """
    Si p es vacío/None, usa default_name (si se pasó).
    Devuelve ruta absoluta: si p es relativo, se interpreta bajo PROJ_ROOT.
    """
    if not p and default_name:
        p = default_name
    p = (p or "").strip().strip('"')
    pp = Path(p)
    return str(pp if pp.is_absolute() else (PROJ_ROOT / pp).resolve())

# --- FAISS opcional para recall adicional ---
import os
from functools import lru_cache

@lru_cache(maxsize=1)
def _load_faiss():
    try:
        import faiss, json
    except Exception:
        return None, None, None

    # Lee del .env (ya cargado arriba) y resuelve a rutas absolutas bajo PROJ_ROOT.
    idx_path  = _abs_under_root(os.getenv("FAISS_INDEX_PATH"),  "faiss_index.bin")
    meta_path = _abs_under_root(os.getenv("FAISS_METADATA_PATH"), "faiss_metadata.json")

    try:
        index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta, faiss
    except Exception:
        return None, None, None

#---FIN del FAISS Opcional-----------

def _faiss_candidates(query: str, k: int = 200):
    """
    Devuelve una lista de dicts con {"chunk_id": int, "score": float} usando FAISS.
    Requiere que el metadata JSON tenga 'chunk_id' por entrada.
    """
    index, meta, dim = _load_faiss()
    if index is None or meta is None or dim is None:
        return []

    # Embedding de la query con el backend cacheado (ya normalizado L2)
    qv = _embed_query(query).astype("float32")   # shape: (384,)
    qv = qv.reshape(1, -1)                       # FAISS espera (nq, dim)
    D, I = index.search(qv, min(k, index.ntotal))  # D: similitudes, I: índices
    I0 = I[0] if len(I.shape) > 1 else I
    D0 = D[0] if len(D.shape) > 1 else D

    out = []
    seen = set()
    for di, ii in zip(D0, I0):
        if ii < 0:
            continue
        m = meta[ii]
        cid = m.get("chunk_id") or m.get("id")
        if cid is None or cid in seen:
            continue
        seen.add(cid)
        out.append({"chunk_id": int(cid), "score": float(di)})
    return out

RERANK_XENCODER = int(os.getenv("RERANK_XENCODER", "0"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
RERANK_TRUST_REMOTE_CODE = (os.getenv("RERANK_TRUST_REMOTE_CODE", "1").lower() in ("1","true","yes","on"))
RERANK_TOPM     = int(os.getenv("RERANK_TOPM", "50"))
RERANK_WEIGHT   = float(os.getenv("RERANK_WEIGHT", "0.7"))

_XE = None

def _get_xe():
    global _XE
    if _XE is not None:
        return _XE
    model_name = os.getenv("RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
    trust_rc = (os.getenv("RERANK_TRUST_REMOTE_CODE", "1").lower() in ("1","true","yes","on"))
    max_len = int(os.getenv("RERANK_MAXLEN", "512"))

    # ⬇️ Microbloque de diagnóstico (exactamente aquí, antes del CrossEncoder)
    print(f"[S28] RERANK_MODEL={model_name} trust_remote_code={trust_rc}")

    # CrossEncoder reenvía kwargs a AutoTokenizer/AutoModel -> acepta trust_remote_code
    _XE = CrossEncoder(
        model_name,
        max_length=max_len,
        trust_remote_code=trust_rc,
    )
    return _XE

def _sigmoid(x):
    import math
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except Exception:
        return 0.5

# --------------------------- Config ---------------------------

EMB_MODEL_NAME = "all-MiniLM-L6-v2"  # 384d
# Cachear el backend para no reinstanciarlo en cada búsqueda
@lru_cache(maxsize=1)
def _get_stf_backend():
    return SentenceTFBackendMiniLM(EMB_MODEL_NAME)

DEFAULT_TOPK = 5
DEFAULT_CANDIDATES = 200
SNIPPET_W = 280

# Pesos del puntaje combinado
W_COS  = 0.65
W_BM25 = 0.25
W_META = 0.10

# Diversidad (MMR): poner None para desactivar
MMR_LAMBDA: Optional[float] = 0.70
PHRASE_BOOST = float(os.getenv("PHRASE_BOOST", "0.12"))

# --------------------------------------------------------------

def _norm_soft(s: str) -> str:
    if not s:
        return ""
    s2 = s.lower()
    if unidecode:
        s2 = unidecode(s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

def _fts_query_for(query: str):
    """
    Devuelve dos expresiones para FTS:
      - expr_and: token1 AND token2 AND ...
      - expr_or : token1 OR  token2 OR  ...
    """
    qn = _norm_soft(query)
    toks = [t for t in re.split(r"[^a-záéíóúñ0-9]+", qn) if len(t) >= 3]
    if not toks:
        return ("", "")
    expr_and = " AND ".join(toks)
    expr_or  = " OR ".join(toks)
    return (expr_and, expr_or)

def _bm25_to_proxy(bm25_value: float) -> float:
    """
    En FTS5, bm25(fts) más chico = mejor.
    Proxy [0..1]: 1/(1+bm25)
    """
    try:
        v = float(bm25_value)
        return 1.0 / (1.0 + v)
    except Exception:
        return 0.0

def _fetch_fts_candidates(conn: sqlite3.Connection, query: str, limit: int) -> List[Dict]:
    """
    Intenta FTS en 2 pasadas (AND luego OR). Si no hay nada, usa fallback LIKE
    que exige al menos 2 tokens presentes (recall razonable sin inundar ruido).
    """
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    expr_and, expr_or = _fts_query_for(query)

    # --- PASO 1: FTS (AND)
    if expr_and:
        try:
            rows = cur.execute("""
                SELECT c.chunk_id, c.doc_id, c.chunk_index, c.page,
                       bm25(fts) AS bm25,
                       c.text
                FROM fts_chunks_v3 AS fts
                JOIN chunks_v3 AS c ON c.chunk_id = fts.rowid
                WHERE fts.text_norm MATCH ?
                ORDER BY bm25(fts) ASC
                LIMIT ?
            """, (expr_and, limit)).fetchall()
            if rows:
                return [dict(r) for r in rows]
        except Exception:
            pass

    # --- PASO 2: FTS (OR)
    if expr_or:
        try:
            rows = cur.execute("""
                SELECT c.chunk_id, c.doc_id, c.chunk_index, c.page,
                       bm25(fts) AS bm25,
                       c.text
                FROM fts_chunks_v3 AS fts
                JOIN chunks_v3 AS c ON c.chunk_id = fts.rowid
                WHERE fts.text_norm MATCH ?
                ORDER BY bm25(fts) ASC
                LIMIT ?
            """, (expr_or, limit)).fetchall()
            if rows:
                return [dict(r) for r in rows]
        except Exception:
            pass

    # --- PASO 3: Fallback LIKE con “al menos 2 tokens”
    qn = _norm_soft(query)
    toks = [t for t in re.split(r"[^a-záéíóúñ0-9]+", qn) if len(t) >= 3]
    if not toks:
        return []

    # armamos: (LIKE t1) + (LIKE t2) + ... y exigimos >=2 hits
    likes = [f"(c.text_norm LIKE ?)" for _ in toks]
    cond  = " + ".join(likes) + " >= 2"   # al menos 2 tokens presentes
    params = [f"%{t}%" for t in toks] + [limit]
    rows = cur.execute(f"""
        SELECT c.chunk_id, c.doc_id, c.chunk_index, c.page,
               10.0 AS bm25,
               c.text
        FROM chunks_v3 c
        WHERE {cond}
        LIMIT ?
    """, params).fetchall()
    return [dict(r) for r in rows]

def _get_chunk_vec_v3(conn: sqlite3.Connection, chunk_id: int) -> Optional[np.ndarray]:
    """
    Lee vector desde embeddings_v3, normalizado L2 (np.float32).
    """
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    row = cur.execute("""
        SELECT dim, vector FROM embeddings_v3 WHERE chunk_id=?
    """, (chunk_id,)).fetchone()
    if not row:
        return None
    dim = int(row["dim"])
    vec = np.frombuffer(row["vector"], dtype=np.float32)
    if vec.shape[0] != dim:
        return None
    n = np.linalg.norm(vec)
    if n > 0:
        vec = (vec / n).astype(np.float32)
    else:
        vec = vec.astype(np.float32)
    return vec

def _embed_query(query: str) -> np.ndarray:
    be = _get_stf_backend()   # ← usa el cache
    qv = be.embed_one(query).astype(np.float32, copy=False)
    n = np.linalg.norm(qv)
    if n > 0:
        qv /= n
    return qv

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def _meta_score(chunk_text: str, issuer: Optional[str], year: Optional[int], query: str) -> float:
    """
    Heurística ligera de meta (0..1):
      + issuer aparece en query -> +0.15
      + año reciente (>=2020) -> +0.10
      + pocas coincidencias de tokens de query en el chunk -> +0.03 c/u (máx 0.10)
    """
    s = 0.0
    qn = _norm_soft(query)

    issuer_norm = _norm_soft(issuer or "")
    if issuer_norm and issuer_norm in qn:
        s += 0.15

    try:
        if year is not None and int(year) >= 2020:
            s += 0.10
    except Exception:
        pass

    toks = [t for t in re.split(r"[^a-záéíóúñ0-9]+", qn) if t and len(t) > 3]
    if toks:
        textn = _norm_soft(chunk_text or "")
        hits = sum(1 for t in set(toks[:4]) if t in textn)
        s += min(0.10, 0.03 * hits)

    return max(0.0, min(1.0, s))

def _load_doc_meta(conn: sqlite3.Connection, doc_ids: List[int]) -> Dict[int, Dict]:
    if not doc_ids:
        return {}
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Descubrir columnas disponibles en documents_v3
    cols = [r[1] for r in cur.execute("PRAGMA table_info(documents_v3)").fetchall()]

    sel = ["id", "title", "path"]

    # issuer (o sinónimo v3)
    if "issuer" in cols:
        sel.append("issuer")
    elif "organismo_v3" in cols:
        sel.append("organismo_v3 AS issuer")
    else:
        sel.append("NULL AS issuer")

    # year (o year_v3)
    if "year" in cols:
        sel.append("year")
    elif "year_v3" in cols:
        sel.append("year_v3 AS year")
    else:
        sel.append("NULL AS year")

    # tipo (varios posibles)
    if "tipo" in cols:
        sel.append("tipo")
    elif "tipo_documento" in cols:
        sel.append("tipo_documento AS tipo")
    elif "tipo_doc" in cols:
        sel.append("tipo_doc AS tipo")
    elif "tipo_documento_v3" in cols:
        sel.append("tipo_documento_v3 AS tipo")
    else:
        sel.append("NULL AS tipo")

    # metadata JSON (por si ahí vino el tipo)
    if "metadata" in cols:
        sel.append("metadata")
    else:
        sel.append("NULL AS metadata")

    qmarks = ",".join("?" for _ in doc_ids)
    rows = cur.execute(
        f"SELECT {', '.join(sel)} FROM documents_v3 WHERE id IN ({qmarks})",
        doc_ids
    ).fetchall()

    meta: Dict[int, Dict] = {}
    for r in rows:
        t = r["tipo"]
        # Fallback: intentar extraer tipo desde metadata JSON si no vino en columna
        if (t is None or t == "") and r["metadata"]:
            try:
                import json
                md = json.loads(r["metadata"])
                t = md.get("tipo") or md.get("tipo_documento") or md.get("tipo_doc") or md.get("tipo_documento_v3")
            except Exception:
                pass

        meta[int(r["id"])] = {
            "title":  r["title"],
            "path":   r["path"],
            "issuer": r["issuer"],
            "year":   r["year"],
            "tipo":   t,
        }
    return meta

def _mmr_diversify(items: List[Dict], lambda_mmr: float, topk: int) -> List[Dict]:
    """
    Maximal Marginal Relevance sobre vectores (si un item no tiene vec, sim=0).
    items requieren: 'vec' (np.ndarray) y 'score_cos' (float).
    """
    if lambda_mmr is None or topk <= 1 or not items:
        return items[:topk]

    selected: List[Dict] = []
    candidates = items[:]
    while candidates and len(selected) < topk:
        if not selected:
            selected.append(candidates.pop(0))
            continue

        best_idx = -1
        best_val = -1e9
        for i, it in enumerate(candidates):
            rel = it.get("score_cos", 0.0)
            div = 0.0
            v = it.get("vec")
            if v is not None:
                for s in selected:
                    sv = s.get("vec")
                    if sv is not None:
                        div = max(div, float(np.dot(v, sv)))
            mmr = lambda_mmr * rel - (1.0 - lambda_mmr) * div
            if mmr > best_val:
                best_val = mmr
                best_idx = i

        if best_idx >= 0:
            selected.append(candidates.pop(best_idx))
        else:
            break

    return selected[:topk]

def search(
    conn: sqlite3.Connection,
    query: str,
    topk: int = DEFAULT_TOPK,
    candidates: int = DEFAULT_CANDIDATES,
    with_snippet: bool = False,
    filtro_tipo: Optional[str] = None,
    max_per_doc: Optional[int] = None,
    diversificar: bool = True,
    modo_general: bool = False,
    use_faiss: bool = True,
):

    """
    Pipeline: FTS -> (coseno + meta + BM25) -> orden -> MMR -> enriquecer
    Retorna dicts con:
      chunk_id, doc_id, title, path, page,
      score, score_cos, score_bm25, score_meta, snippet
    """
    # 1) Candidatos (FTS)
    cands = _fetch_fts_candidates(conn, query, candidates)

    # 1.b) Fallback por FAISS si FTS no trajo nada
    if not cands and use_faiss:
        extra = _faiss_candidates(query, k=candidates * 2)
        if extra:
            extra_ids = [int(e["chunk_id"]) for e in extra]
            qmarks = ",".join("?" for _ in extra_ids)
            cur = conn.cursor()
            rows = cur.execute(f"""
                SELECT
                    c.chunk_id,
                    c.doc_id,
                    c.chunk_index,
                    c.page,
                    c.text,
                    (SELECT bm25(fts_chunks_v3) FROM fts_chunks_v3 WHERE rowid = c.chunk_id) AS bm25
                FROM chunks_v3 AS c
                WHERE c.chunk_id IN ({qmarks})
            """, extra_ids).fetchall()
            cands = [{
                "chunk_id": int(r[0]),
                "doc_id": int(r[1]),
                "chunk_index": int(r[2]),
                "page": r[3],
                "text": r[4],
                "bm25": float(r[5] or 0.0),
            } for r in rows]

    # Si sigue vacío, no hay nada que rerankear
    if not cands:
        return []

    
    # --- Opcional: ampliar recall con FAISS, pero dando prioridad a FAISS antes de cortar ---
    if use_faiss:
        # 1) Top-N semánticos por FAISS
        extra = _faiss_candidates(query, k=candidates * 2)
        extra_ids = [int(e["chunk_id"]) for e in extra] if extra else []

        faiss_rows = []
        if extra_ids:
            qmarks = ",".join("?" for _ in extra_ids)
            cur = conn.cursor()
            faiss_rows_raw = cur.execute(f"""
                SELECT
                    c.chunk_id,
                    c.doc_id,
                    c.chunk_index,
                    c.page,
                    c.text,
                    (SELECT bm25(fts_chunks_v3) FROM fts_chunks_v3 WHERE rowid = c.chunk_id) AS bm25
                FROM chunks_v3 AS c
                WHERE c.chunk_id IN ({qmarks})
            """, extra_ids).fetchall()

            for r in faiss_rows_raw:
                faiss_rows.append({
                    "chunk_id": int(r[0]),
                    "doc_id": int(r[1]),
                    "chunk_index": int(r[2]),
                    "page": r[3],
                    "text": r[4],
                    "bm25": float(r[5] or 0.0),
                })

        # 2) MERGE con prioridad: primero FAISS, luego los FTS de cands (sin duplicar)
        merged = []
        seen = set()
        for r in faiss_rows:
            cid = int(r["chunk_id"])
            if cid not in seen:
                seen.add(cid)
                merged.append(r)
        for r in cands:
            cid = int(r["chunk_id"])
            if cid not in seen:
                seen.add(cid)
                merged.append(r)

        # 3) Cortar recién ahora
        cands = merged[:candidates]


    # 2) Doc meta cache
    doc_ids = list({int(r["doc_id"]) for r in cands})
    docs_meta = _load_doc_meta(conn, doc_ids)

    # 3) Query embedding
    qv = _embed_query(query)

    # 4) Scoring
    scored: List[Dict] = []
    for r in cands:
        cid   = int(r["chunk_id"])
        did   = int(r["doc_id"])
        bm25v = float(r["bm25"])
        text  = r["text"] or ""

        vec = _get_chunk_vec_v3(conn, cid)
        cosv = float(np.dot(vec, qv)) if vec is not None else 0.0
        bm25_proxy = _bm25_to_proxy(bm25v)
        dm = docs_meta.get(did, {})
        meta_s = _meta_score(text, dm.get("issuer"), dm.get("year"), query)

        # --- Boost por "frase exacta" (normalizada, sin tildes, exige consulta con ≥2 palabras) ---
        qn = _norm_soft(query)
        tn = _norm_soft(text)
        # condición mínima para considerar "frase": al menos 2 palabras y largo decente
        cond_phrase = (len(qn.split()) >= 2 and len(qn) >= 12)
        has_phrase = 1.0 if (cond_phrase and qn in tn) else 0.0


        w_meta = 0.0 if modo_general else W_META
        score = (W_COS * cosv) + (W_BM25 * bm25_proxy) + (w_meta * meta_s) + (PHRASE_BOOST * has_phrase)


        scored.append({
            "chunk_id": cid,
            "doc_id": did,
            "chunk_index": int(r["chunk_index"]),
            "page": r["page"],
            "text": text,
            "score": float(score),
            "score_cos": float(cosv),
            "score_bm25": float(bm25_proxy),
            "score_meta": float(meta_s),
            "vec": vec,  # para MMR
        })

    # --- RERANK FINAL con cross-encoder (opcional) ---
    if RERANK_XENCODER and scored:
        xe = _get_xe()
        # tomamos los mejores M según score actual
        tmp = sorted(scored, key=lambda x: x["score"], reverse=True)[:min(RERANK_TOPM, len(scored))]
        pairs = [(query, it["text"]) for it in tmp]
        xe_scores = xe.predict(pairs)
        for it, s in zip(tmp, xe_scores):
            s_norm = _sigmoid(s)  # [0..1]
            it["score_xe"] = float(s_norm)
            it["score"] = (1.0 - RERANK_WEIGHT) * it["score"] + RERANK_WEIGHT * s_norm
    # --- fin RERANK ---

    scored.sort(key=lambda x: x["score"], reverse=True)

    # 5) Diversidad
    lambda_mmr = 0.65 if diversificar else None
    final_items = _mmr_diversify(scored, lambda_mmr, topk) if lambda_mmr else scored[:topk]

    # Límite por documento (si se pide)
    if max_per_doc and max_per_doc > 0:
        from collections import defaultdict
        per_doc = defaultdict(int)
        filtered = []
        for it in final_items:
            if per_doc[it["doc_id"]] >= max_per_doc:
                continue
            filtered.append(it)
            per_doc[it["doc_id"]] += 1
            if len(filtered) >= topk:
                break
        final_items = filtered


    # 6) Enriquecer salida + anti-duplicados por (doc_id, page)
    out: List[Dict] = []
    seen = set()
    for it in final_items:
        key = (it["doc_id"], it.get("page"))
        if key in seen:
            continue
        seen.add(key)

        dm = docs_meta.get(it["doc_id"], {})
        title = dm.get("title") or (dm.get("path") or "").split("\\")[-1]
        path  = dm.get("path")
        page  = it.get("page")
        snip = None
        if with_snippet:
            txt = (it.get("text") or "").replace("\n", " ")
            snip = textwrap.shorten(txt, width=SNIPPET_W, placeholder="…")
        issuer = dm.get("issuer")      # ← agregar
        year   = dm.get("year")        # ← agregar
        tipo   = dm.get("tipo")   # ← NUEVO

        out.append({
            "chunk_id": it["chunk_id"],
            "doc_id": it["doc_id"],
            "title": title,
            "path": path,
            "page": page,
            "issuer": issuer,   # ← NUEVO
            "year": year,       # ← NUEVO
            "tipo": tipo,          # ← NUEVO
            "score": it["score"],
            "score_cos": it["score_cos"],
            "score_bm25": it["score_bm25"],
            "score_meta": it["score_meta"],
            "snippet": snip,
        })
        if len(out) >= topk:
            break

    return out

def search_print(conn: sqlite3.Connection, query: str, topk: int = DEFAULT_TOPK, candidates: int = DEFAULT_CANDIDATES):
    rows = search(conn, query, topk=topk, candidates=candidates, with_snippet=True)
    print(f"\nQ: {query}")
    if not rows:
        print("  (sin resultados)")
        return
    for i, r in enumerate(rows, 1):
        pg = r["page"] if r["page"] is not None else "-"
        print(f"  {i}. {r['title']}  [p.{pg}]  score={r['score']:.3f} (cos={r['score_cos']:.3f} | bm25={r['score_bm25']:.3f} | meta={r['score_meta']:.3f})")
        if r["snippet"]:
            print(f"     {r['snippet']}")
