# scripts/migrate_legacy_embeddings.py
#
# One-shot para dejar tu base "lista":
#  1) Migra vectores legados de `chunk_embeddings` -> tabla canónica `embeddings`.
#  2) (Opcional) Genera embeddings FALTANTES vía API (si EMBED_API_URL/KEY/MODEL están en .env o entorno).
#  3) (Opcional) Backfill de `documents.keywords` a partir de los chunks (barato, sin IA).
#  4) Muestra conteos finales. (No toca FTS: no es necesario para embeddings/keywords.)
#
# Requisitos: solo stdlib. Lee variables desde .env si existe.
#  - EMBED_API_URL, EMBED_API_KEY, EMBED_MODEL, EMBED_DIM (opcional) para la etapa 2.
#
# Uso:
#   python scripts/migrate_legacy_embeddings.py --embed --keywords --batch 32
#     --embed     -> llama al endpoint de embeddings para faltantes
#     --keywords  -> llena documents.keywords con términos más frecuentes
#     --batch N   -> tamaño de lote para llamadas de embedding (default 16)

from __future__ import annotations
import os, sqlite3, json, re, time, urllib.request, argparse
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parent.parent
DB = str(ROOT / "sistema28.db")

# =============== util: carga .env sencilla ==================

def load_env(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            env[k.strip()] = v.strip()
    # Mezcla con entorno (entorno domina)
    for k, v in os.environ.items():
        env[k] = v
    return env

CFG = load_env(ROOT / ".env")
EMBED_API_URL = CFG.get("EMBED_API_URL", "").strip()
EMBED_API_KEY = CFG.get("EMBED_API_KEY", "").strip()
EMBED_MODEL   = CFG.get("EMBED_MODEL", "").strip()
try:
    EMBED_DIM = int(CFG.get("EMBED_DIM", "0") or 0)
except Exception:
    EMBED_DIM = 0

# =============== Paso 1: migración de embeddings legados ==================

SCHEMA_EMB = """
CREATE TABLE IF NOT EXISTS embeddings (
  chunk_id INTEGER PRIMARY KEY,
  vector   TEXT NOT NULL,
  dim      INTEGER NOT NULL
);
"""

def migrate_legacy(con: sqlite3.Connection) -> int:
    cur = con.cursor()
    cur.executescript(SCHEMA_EMB)
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_embeddings'")
    if not cur.fetchone():
        print("[migrate] No existe tabla legado 'chunk_embeddings'. Saltando migración…")
        return 0

    cur.execute("SELECT chunk_id, embedding FROM chunk_embeddings")
    moved = 0
    for cid, emb in cur.fetchall():
        if not isinstance(emb, str) or not emb.strip():
            continue
        # dim por conteo de números en el string
        dim = len([x for x in emb.split(',') if x])
        try:
            cur.execute("INSERT OR IGNORE INTO embeddings (chunk_id, vector, dim) VALUES (?,?,?)", (cid, emb, dim))
            if cur.rowcount:
                moved += 1
        except Exception:
            # fila corrupta -> ignorar
            pass
    con.commit()
    return moved

# =============== Paso 2: generar embeddings faltantes (API) ==================

HEADERS = {"Content-Type": "application/json"}
if EMBED_API_KEY:
    HEADERS["Authorization"] = f"Bearer {EMBED_API_KEY}"

TOKEN = re.compile(r"[a-záéíóúüñ]{3,}")
STOP = set("""
la el los las un una unos unas y o u de del a ante bajo con contra desde durante en entre hacia hasta mediante para por según sin so sobre tras al lo le les que se su sus es son fue fueron ser será serán han hay había
art artículo articulo ley decreto resolucion resolución disposición disposicion
""".split())


def fetch_missing_ids(con: sqlite3.Connection) -> List[int]:
    cur = con.cursor()
    cur.execute("""
        SELECT dc.id
        FROM document_chunks dc
        LEFT JOIN embeddings e ON e.chunk_id = dc.id
        WHERE e.chunk_id IS NULL
    """)
    return [r[0] for r in cur.fetchall()]


def load_chunk_texts(con: sqlite3.Connection, ids: List[int]) -> List[str]:
    if not ids:
        return []
    cur = con.cursor()
    q = ",".join("?" for _ in ids)
    cur.execute(f"SELECT COALESCE(text_norm, chunk_text) FROM document_chunks WHERE id IN ({q})", ids)
    return [r[0] or "" for r in cur.fetchall()]


def call_embed(texts: List[str]) -> List[List[float]]:
    if not EMBED_API_URL or not EMBED_MODEL:
        raise RuntimeError("Config de embeddings incompleta: EMBED_API_URL/EMBED_MODEL")
    payload = {"model": EMBED_MODEL, "input": texts}
    req = urllib.request.Request(EMBED_API_URL, data=json.dumps(payload).encode("utf-8"), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as resp:
        j = json.loads(resp.read().decode("utf-8"))
        if isinstance(j, dict) and "data" in j:
            return [d.get("embedding", []) for d in j["data"]]
        if isinstance(j, list) and j and isinstance(j[0], list):
            return j
        raise RuntimeError("Respuesta de embeddings no reconocida")


def save_embeddings(con: sqlite3.Connection, ids: List[int], vecs: List[List[float]]):
    cur = con.cursor()
    cur.execute(SCHEMA_EMB)
    for cid, v in zip(ids, vecs):
        if not v:
            continue
        dim = len(v)
        if EMBED_DIM and dim != EMBED_DIM:
            print(f"[warn] dim={dim} difiere de EMBED_DIM={EMBED_DIM} (chunk {cid})")
        s = ",".join(f"{x:.6f}" for x in v)
        cur.execute("INSERT OR REPLACE INTO embeddings (chunk_id, vector, dim) VALUES (?,?,?)", (cid, s, dim))
    con.commit()

# =============== Paso 3: keywords por documento ==================

def doc_ids(con: sqlite3.Connection) -> List[int]:
    cur = con.cursor(); cur.execute("SELECT id FROM documents")
    return [r[0] for r in cur.fetchall()]


def top_terms(texts: List[str], k: int = 12) -> List[str]:
    from collections import Counter
    cnt = Counter()
    for t in texts:
        for w in TOKEN.findall(t or ""):
            if w in STOP: continue
            cnt[w] += 1
    return [w for w, _ in cnt.most_common(k)]


def backfill_keywords(con: sqlite3.Connection, topk: int = 12) -> int:
    cur = con.cursor()
    updated = 0
    for did in doc_ids(con):
        cur.execute("SELECT COALESCE(text_norm, chunk_text) FROM document_chunks WHERE document_id=?", (did,))
        texts = [r[0] or "" for r in cur.fetchall()]
        if not texts: continue
        kws = ", ".join(top_terms(texts, k=topk))
        cur.execute("UPDATE documents SET keywords=? WHERE id=?", (kws, did))
        updated += 1
    con.commit(); return updated

# =============== Conteos y main ==================

def counts(con: sqlite3.Connection) -> Dict[str, int]:
    cur = con.cursor()
    out = {}
    for t in ["documents", "document_chunks", "embeddings"]:
        cur.execute(f"SELECT COUNT(*) FROM {t}"); out[t] = cur.fetchone()[0]
    return out


def main():
    ap = argparse.ArgumentParser(description="Migrar embeddings legados, generar faltantes por API y backfill de keywords")
    ap.add_argument("--embed", action="store_true", help="Generar embeddings faltantes vía API")
    ap.add_argument("--keywords", action="store_true", help="Backfill de documents.keywords")
    ap.add_argument("--batch", type=int, default=16, help="Tamaño de lote para embeddings (default 16)")
    ap.add_argument("--sleep", type=float, default=0.2, help="Retardo entre lotes (s)")
    ap.add_argument("--topk", type=int, default=12, help="Keywords por documento (default 12)")
    args = ap.parse_args()

    con = sqlite3.connect(DB)

    # Paso 1: migración
    moved = migrate_legacy(con)
    print(f"[migrate] Filas migradas desde 'chunk_embeddings' -> 'embeddings': {moved}")

    # Paso 2: embeddings faltantes (opcional)
    if args.embed:
        ids = fetch_missing_ids(con)
        print(f"[embed] Chunks sin vector: {len(ids)}")
        for i in range(0, len(ids), args.batch):
            part = ids[i:i+args.batch]
            texts = load_chunk_texts(con, part)
            try:
                vecs = call_embed(texts)
            except Exception as e:
                print("[embed] Error API:", e)
                time.sleep(2)
                continue
            save_embeddings(con, part, vecs)
            print(f"[embed] guardados {len(part)} vectores")
            if args.sleep:
                time.sleep(args.sleep)

    # Paso 3: backfill keywords (opcional)
    if args.keywords:
        upd = backfill_keywords(con, topk=args.topk)
        print(f"[keywords] documentos actualizados: {upd}")

    # Resumen
    c = counts(con)
    print("[counts]", c)
    con.close()

if __name__ == "__main__":
    main()
