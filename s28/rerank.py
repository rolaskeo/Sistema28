# s28/rerank.py
# Búsqueda híbrida (FTS5 + meta + embeddings opcional), score jurídico y MMR.
# Diseñado para CPU, sin dependencias externas (solo stdlib) y robusto ante consultas FTS5.

from __future__ import annotations
import sqlite3, math, hashlib, re, os
from typing import List, Dict, Any, Tuple, Optional

# === Configuración ===
# Usa la DB en la carpeta del proyecto por defecto
DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sistema28.db")


def set_db_path(db_path: str):
    """Permite cambiar el path de la DB en caliente si hiciera falta."""
    global DEFAULT_DB
    DEFAULT_DB = db_path


# === Utilidades numéricas / MMR ===

def l2norm(v: List[float]) -> List[float]:
    s = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / s for x in v]


def cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def mmr_select(items: List[Dict[str, Any]], k: int = 10, lam: float = 0.7) -> List[Dict[str, Any]]:
    """
    Selecciona top-k balanceando relevancia (score) y diversidad (1 - similitud coseno).
    items: [{'score': float, 'vec': Optional[List[float]], ...}, ...]
    """
    selected: List[Dict[str, Any]] = []
    remaining = items[:]
    while remaining and len(selected) < k:
        if not selected:
            i = max(remaining, key=lambda x: x.get("score", 0.0))
            selected.append(i)
            remaining.remove(i)
            continue

        def novelty(x: Dict[str, Any]) -> float:
            # Mayor = más similar a los ya seleccionados (penaliza)
            if x.get("vec") is None:
                return 0.0
            sims = []
            for s in selected:
                v = s.get("vec")
                if v is None:
                    continue
                sims.append(cosine(x["vec"], v))
            return max(sims) if sims else 0.0

        best = max(remaining, key=lambda x: lam * x.get("score", 0.0) - (1 - lam) * novelty(x))
        selected.append(best)
        remaining.remove(best)
    return selected


def hash_query(q: str) -> str:
    return hashlib.sha256(q.encode("utf-8")).hexdigest()[:16]


# === Sanitización de consultas FTS5 ===
_FTS_SAFE_CHARS = r"0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ"


def normalize_match_query(q: str) -> str:
    """
    Limpia la query para FTS5 MATCH.
    - Reemplaza cualquier char no permitido por espacio (puntos, comillas, etc.).
    - Colapsa espacios.
    - Mantiene números y letras (incluye acentos y ñ).
    """
    q = q.strip()
    q = re.sub(fr"[^{_FTS_SAFE_CHARS}\s]", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q


def normalize_plain(q: str) -> str:
    """Versión normalizada para cómputos simples (coverage)."""
    q = q.replace("_", " ").lower()
    q = re.sub(r"\s+", " ", q).strip()
    return q


# === Lectura opcional de embeddings ===

def get_chunk_vec(con: sqlite3.Connection, chunk_id: int) -> Tuple[Optional[List[float]], int]:
    """
    Devuelve (vector, dim) para un chunk. Intenta primero 'embeddings', luego 'chunk_embeddings' (legado).
    Formatos soportados:
      - Texto "f1,f2,..." (más común en tus tablas).
    """
    cur = con.cursor()

    # 1) Tabla nueva 'embeddings'
    try:
        cur.execute("SELECT vector, dim FROM embeddings WHERE chunk_id=?", (chunk_id,))
        row = cur.fetchone()
        if row:
            vec, dim = row
            if isinstance(vec, str):
                arr = [float(x) for x in vec.split(",") if x]
                if dim is None or dim == 0:
                    dim = len(arr)
                if dim > 0 and len(arr) == dim:
                    return (l2norm(arr), dim)
    except Exception:
        pass

    # 2) Tabla legado 'chunk_embeddings'
    try:
        cur.execute("PRAGMA table_info(chunk_embeddings)")
        cols = [c[1] for c in cur.fetchall()]
        if "chunk_id" in cols:
            cur.execute("SELECT embedding FROM chunk_embeddings WHERE chunk_id=?", (chunk_id,))
            row = cur.fetchone()
            if row and isinstance(row[0], str):
                arr = [float(x) for x in row[0].split(",") if x]
                if len(arr) > 0:
                    return (l2norm(arr), len(arr))
    except Exception:
        pass

    return (None, 0)


# === Scoring jurídico ===

def compute_meta_boost(row: Dict[str, Any], query_norm: str) -> float:
    """
    Señales baratas:
      - chunk_type: RESUELVE/FALLO/DICTAMINA > CONSIDERANDOS > VISTOS > ANEXO/OTROS
      - cobertura de términos de la query en text_norm
      - leve boost si hay organismo_emisor/issuer
    """
    ct = (row.get("chunk_type") or "").upper()
    sec_w = {"RESUELVE": 1.0, "FALLO": 1.0, "DICTAMINA": 0.9, "CONSIDERANDOS": 0.7, "VISTOS": 0.5, "ANEXO": 0.3}
    w_sec = sec_w.get(ct, 0.4)

    txt_terms = (row.get("text_norm") or "").split()
    q_terms = [t for t in query_norm.split() if len(t) > 2]
    cover = 0.0
    if txt_terms and q_terms:
        present = sum(1 for t in q_terms if t in txt_terms)
        cover = present / max(1, len(q_terms))

    org = 0.05 if (row.get("organismo_emisor") or row.get("issuer")) else 0.0
    return 0.10 * w_sec + 0.08 * cover + org


def blend_score(bm25_proxy: float, cos: float, meta: float) -> float:
    """Peso final (ajustable)."""
    return 0.5 * (bm25_proxy or 0.0) + 0.4 * (cos or 0.0) + 0.2 * (meta or 0.0)


# === Búsqueda principal ===

def search(
    query: str,
    k: int = 12,
    db_path: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Devuelve hasta k items con campos:
      id, document_id, titulo, preview, chunk_type, page_num, score, vec
    filters opcionales:
      - chunk_type: lista o str (e.g., "RESUELVE")
      - document_id: int o lista de int
      - issuer: str (coincidencia simple en documents.issuer u organismo_emisor)
    """
    db = db_path or DEFAULT_DB
    con = sqlite3.connect(db)
    cur = con.cursor()

    # --- 1) candidatos por FTS (recall alto con fallback) ---
    q_norm = normalize_plain(query)
    terms = [t for t in normalize_match_query(q_norm).split() if t]
    if not terms:
        con.close()
        return []

    def run_fts(match_expr: str, limit: int) -> list:
        sql = (
            """
            SELECT dc.id, dc.document_id, dc.titulo, dc.chunk_text, dc.chunk_type, dc.page_num, dc.text_norm,
                   dc.organismo_emisor
            FROM fts_chunks
            JOIN document_chunks AS dc ON dc.id = fts_chunks.rowid
            WHERE fts_chunks MATCH ?
            LIMIT ?
            """
        )
        cur.execute(sql, (match_expr, limit))
        return cur.fetchall()

    # Estrategia A: AND (todos los términos)
    expr_and = " ".join(terms)  # AND implícito
    rows = run_fts(expr_and, 400)

    # Estrategia B: OR (si AND no devuelve nada)
    if not rows and len(terms) > 1:
        expr_or = " OR ".join(terms)
        rows = run_fts(expr_or, 400)

    # Estrategia C: Unión por término (si aún vacío)
    if not rows:
        bag = []
        seen = set()
        for t in terms:
            r = run_fts(t, 120)  # top por término
            for x in r:
                if x[0] not in seen:
                    seen.add(x[0])
                    bag.append(x)
            if len(bag) >= 400:
                break
        rows = bag

    cands = rows
    if not cands:
        con.close()
        return []

    # --- Filtros opcionales ---
    filters = filters or {}
    if filters.get("chunk_type"):
        types = filters["chunk_type"]
        if isinstance(types, (list, tuple, set)):
            keep = {str(x).upper() for x in types}
            cands = [r for r in cands if (r[4] or "").upper() in keep]
        else:
            keep = str(types).upper()
            cands = [r for r in cands if (r[4] or "").upper() == keep]

    if filters.get("document_id"):
        ids = filters["document_id"]
        if isinstance(ids, (list, tuple, set)):
            ids = set(ids)
            cands = [r for r in cands if r[1] in ids]
        else:
            cands = [r for r in cands if r[1] == int(ids)]

    # issuer (si se requiere, hacemos un lookup liviano por documento)
    issuer_filter = filters.get("issuer")
    issuers_by_doc: Dict[int, str] = {}
    if issuer_filter:
        cur.execute("SELECT id, COALESCE(issuer, organismo) FROM documents")
        issuers_by_doc = {docid: (iss or "") for docid, iss in cur.fetchall()}
        low = issuer_filter.lower()
        cands = [r for r in cands if low in (issuers_by_doc.get(r[1], "").lower() or (r[7] or "").lower())]

    # --- 2) Scoring ---
    items: List[Dict[str, Any]] = []
    q_terms = q_norm.split()
    for cid, docid, titulo, chunk_text, ctype, pnum, tnorm, org_emisor in cands:
        # BM25-proxy: proporción de términos presentes (barato)
        hits = 0
        if tnorm:
            tset = set(tnorm.split())
            for t in q_terms:
                if t and t in tset:
                    hits += 1
        bm25_proxy = hits / max(1, len(q_terms))

        meta = compute_meta_boost(
            {"chunk_type": ctype, "text_norm": tnorm, "organismo_emisor": org_emisor, "issuer": None},
            q_norm,
        )

        # Embedding opcional del chunk (si existe en DB). No calculamos embedding de la query (CPU barato).
        vec, dim = get_chunk_vec(con, cid)
        cos = 0.0  # si luego agregás embedding de la query, aquí usarías coseno real

        score = blend_score(bm25_proxy, cos, meta)

        items.append(
            {
                "id": cid,
                "document_id": docid,
                "titulo": titulo or "",
                "preview": (chunk_text or "")[:600].replace("\n", " "),
                "chunk_type": ctype or "",
                "page_num": pnum,
                "score": float(score),
                "vec": vec,
            }
        )

    # --- 3) Orden inicial y MMR ---
    items.sort(key=lambda x: x["score"], reverse=True)
    final = mmr_select(items, k=k, lam=0.7)

    con.close()
    return final


# === Helper de impresión ===

def print_results(items: List[Dict[str, Any]]):
    if not items:
        print("(sin resultados)")
        return
    for i, it in enumerate(items, 1):
        sec = (it.get("chunk_type") or "").upper()
        p = it.get("page_num")
        print(f"{i:02d}. [chunk {it['id']}] ({sec}, pág {p}) score={it['score']:.3f}")
        prev = it.get("preview", "")
        if prev:
            print("     " + prev)


# === Ejecución directa (debug rápido) ===
if __name__ == "__main__":
    q = "concesion vial ley 17.520"
    rs = search(q, k=10)
    print_results(rs)
