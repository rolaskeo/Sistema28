# scripts/migrate_chunk_embeddings_fix.py
#
# Migra embeddings desde la tabla LEGADA `chunk_embeddings(chunk_id, embedding BLOB)`
# hacia la tabla canónica `embeddings(chunk_id INTEGER PRIMARY KEY, model TEXT, dim INTEGER, vector BLOB, created_at TEXT)`.
# - Detecta y convierte formatos habituales del campo `embedding` (texto con comas, JSON, binario float32).
# - Inserta sin sobreescribir si ya existe una fila en `embeddings` (INSERT OR IGNORE).
# - Modelo por parámetro --model (ej: all-MiniLM-L6-v2). Dimensión se infiere del vector.
#
# Uso:
#   python scripts/migrate_chunk_embeddings_fix.py --model all-MiniLM-L6-v2 --batch 1000
#
from __future__ import annotations
import os, sqlite3, argparse, json, sys
from array import array
from typing import Any, List, Tuple, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = str(ROOT / "sistema28.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
  chunk_id   INTEGER PRIMARY KEY,
  model      TEXT,
  dim        INTEGER NOT NULL,
  vector     BLOB NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

def ensure_schema(con: sqlite3.Connection):
    con.executescript(SCHEMA)


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    cur = con.cursor(); cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def iter_legacy_rows(con: sqlite3.Connection, batch: int = 1000) -> Iterator[Tuple[int, List[Tuple[int, Any]]]]:
    cur = con.cursor()
    # solo filas que NO estén ya en embeddings
    cur.execute(
        """
        SELECT COUNT(*) FROM chunk_embeddings ce
        LEFT JOIN embeddings e ON e.chunk_id = ce.chunk_id
        WHERE e.chunk_id IS NULL
        """
    )
    total = cur.fetchone()[0]

    cur2 = con.cursor()
    cur2.execute(
        """
        SELECT ce.chunk_id, ce.embedding
        FROM chunk_embeddings ce
        LEFT JOIN embeddings e ON e.chunk_id = ce.chunk_id
        WHERE e.chunk_id IS NULL
        """
    )
    while True:
        rows = cur2.fetchmany(batch)
        if not rows:
            break
        yield total, rows


def to_float_list(blob_or_text: Any) -> List[float]:
    """Convierte el valor almacenado en `embedding` a una lista[float].
    Soporta: str con comas/espacios, JSON de lista, y binario float32.
    """
    # 1) Si viene como memoryview/bytes -> intentar decodificar como texto
    if isinstance(blob_or_text, (bytes, memoryview, bytearray)):
        b = bytes(blob_or_text)
        # 1a) intentar utf-8 texto
        try:
            s = b.decode('utf-8', errors='strict').strip()
            if s:
                # si parece JSON de lista
                if s.startswith('[') and s.endswith(']'):
                    arr = json.loads(s)
                    return [float(x) for x in arr]
                # si es "1.0, 2.0, 3.0" o similar
                if ("," in s) or (" " in s):
                    parts = [p for p in s.replace("\n", " ").replace("\t", " ").split() if p not in {',',';'}]
                    # si tenía comas, split por coma primero
                    if "," in s:
                        parts = [p for chunk in s.split(',') for p in chunk.strip().split() if p]
                    return [float(p) for p in parts]
        except Exception:
            pass
        # 1b) si no es texto válido: probar float32 binario (múltiplo de 4 bytes)
        if len(b) % 4 == 0 and len(b) > 0:
            try:
                arrf = array('f')
                arrf.frombytes(b)
                return [float(x) for x in arrf]
            except Exception:
                pass
        # 1c) último intento como JSON binario
        try:
            arr = json.loads(b.decode('utf-8', errors='ignore'))
            if isinstance(arr, list):
                return [float(x) for x in arr]
        except Exception:
            pass
        raise ValueError("No pude decodificar embedding binario")

    # 2) Si viene como str directamente
    if isinstance(blob_or_text, str):
        s = blob_or_text.strip()
        if not s:
            return []
        if s.startswith('[') and s.endswith(']'):
            arr = json.loads(s)
            return [float(x) for x in arr]
        parts = [p for p in s.replace("\n", " ").replace("\t", " ").split() if p not in {',',';'}]
        if "," in s:
            parts = [p for chunk in s.split(',') for p in chunk.strip().split() if p]
        return [float(p) for p in parts]

    # 3) Otro tipo -> vacío
    return []


def floats_to_f32_blob(vals: List[float]) -> bytes:
    arrf = array('f', [float(x) for x in vals])
    return arrf.tobytes()


def migrate(model: str, batch: int = 1000) -> None:
    con = sqlite3.connect(DB)
    ensure_schema(con)

    if not table_exists(con, 'chunk_embeddings'):
        print("[migrate] No existe tabla 'chunk_embeddings' en esta DB.")
        con.close(); return

    moved = 0
    for total, rows in iter_legacy_rows(con, batch=batch):
        for cid, emb in rows:
            try:
                vec = to_float_list(emb)
                if not vec:
                    continue
                dim = len(vec)
                blob = floats_to_f32_blob(vec)
                con.execute(
                    "INSERT OR IGNORE INTO embeddings (chunk_id, model, dim, vector) VALUES (?,?,?,?)",
                    (int(cid), model, dim, sqlite3.Binary(blob)),
                )
                if con.total_changes:
                    moved += 1
            except Exception as e:
                # saltar fila problemática
                print(f"[warn] cid={cid} no migrado: {e}")
        con.commit()
        print(f"[migrate] Progreso: {moved}/{total} migrados…")

    # Resumen
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM embeddings"); emb_n = cur.fetchone()[0]
    print(f"[migrate] DONE. Total en embeddings: {emb_n} (nuevos: {moved})")
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Migrar embeddings legados chunk_embeddings -> embeddings")
    ap.add_argument("--model", required=True, help="Nombre de modelo a registrar (ej: all-MiniLM-L6-v2)")
    ap.add_argument("--batch", type=int, default=1000, help="Tamaño de lote (default 1000)")
    args = ap.parse_args()
    migrate(model=args.model, batch=args.batch)
