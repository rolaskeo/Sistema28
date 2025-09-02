# s28/indexer.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, sqlite3
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from s28.chunker import chunk_text, Chunk

# ---------- Dependencias opcionales ----------
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_SENTENCE_TF = True
except Exception:
    _HAS_SENTENCE_TF = False

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# ---------- Conexión ----------
def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

# ---------- Backends de Embeddings ----------
class EmbeddingBackend:
    name: str
    dim: int
    def embed(self, texts: List[str]) -> np.ndarray: ...
    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

class SentenceTFBackendMiniLM(EmbeddingBackend):
    """
    Usa exactamente el modelo que venías usando:
    all-MiniLM-L6-v2  (dim = 384)
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not _HAS_SENTENCE_TF:
            raise RuntimeError("sentence-transformers no está instalado. pip install sentence-transformers")
        self._model = SentenceTransformer(model_name, device="cpu")
        self.name = f"st:{model_name}"
        try:
            self.dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            # MiniLM-L6-v2 es 384
            self.dim = 384

    def embed(self, texts: List[str]) -> np.ndarray:
        v = self._model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
        return v.astype(np.float32, copy=False)

# ---------- Utils ----------
def _np_to_blob(x: np.ndarray) -> bytes:
    assert x.dtype == np.float32
    return x.tobytes(order="C")

def _blob_to_np(b: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32, count=dim)

# ---------- Gestor FAISS (append-safe) ----------
class FaissManager:
    """
    Maneja un índice FAISS basado en producto interno (con vectores L2-normalizados -> cos sim).
    Persiste:
      - index_path: binario FAISS (por ej., 'faiss_index.bin')
      - meta_path:  JSON con: {"dim": 384, "model": "st:all-MiniLM-L6-v2", "ids": [chunk_id, ...]}
    Si ya existe metadata con otro formato, intenta conservar claves y extender "ids".
    """
    def __init__(self, index_path: str, meta_path: str):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.meta: Dict[str, Any] = {}

    def load(self) -> None:
        if _HAS_FAISS and os.path.isfile(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.isfile(self.meta_path):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
            except Exception:
                self.meta = {}
        if "ids" not in self.meta:
            self.meta["ids"] = []

    def save(self) -> None:
        if _HAS_FAISS and self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def ensure_index(self, dim: int) -> None:
        if not _HAS_FAISS:
            raise RuntimeError("FAISS no disponible. Instalar faiss-cpu para usar índice vectorial.")
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
        # Chequeo de dimensión
        if self.index.d != dim:
            raise ValueError(f"Dimensión FAISS existente ({self.index.d}) != {dim}")

    def known_ids(self) -> set:
        return set(self.meta.get("ids", []))

    def add(self, vectors: np.ndarray, ids: List[int], model_name: str, dim: int) -> None:
        """
        vectors: shape (N, dim), float32
        ids:     lista de chunk_id (N)
        """
        if vectors.size == 0:
            return
        self.ensure_index(dim)
        # Normalizar L2 para que el producto interno sea coseno
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        # Actualizar metadata
        self.meta["dim"] = dim
        self.meta["model"] = model_name
        current = self.meta.get("ids", [])
        current.extend([int(x) for x in ids])
        self.meta["ids"] = current

# ---------- Indexer (safe con DB/FAISS existentes) ----------
class Indexer:
    """
    Indexador 'no intrusivo':
    - manage_schema=False  => NO crea/alterar tablas (usa las existentes)
    - Usa all-MiniLM-L6-v2 vía sentence-transformers para ser compatible con tu pipeline previo
    - Anexa embeddings que falten (solo chunks sin vector)
    - Anexa al FAISS existente (solo ids que falten)
    """
    def __init__(self, db_path: str, manage_schema: bool = False):
        self.db_path = db_path
        self.conn = _connect(db_path)
        self.manage_schema = manage_schema
        # Si en algún momento querés que cree tablas nuevas auxiliares, podrías agregar un método aparte.

    # ---- Documentos + Chunker: agrega sin tocar lo existente
    def index_document_text(self, title: str, text: str, source_path: Optional[str] = None,
                            meta: Optional[Dict[str, Any]] = None,
                            min_words: int = 60, target_words: int = 100, max_words: int = 140) -> int:
        """
        Inserta un documento y sus chunks en tablas existentes:
          - documents(id, title, source_path, created_at?, meta_json?)
          - chunks(chunk_id, doc_id, chunk_index, text, start_char, end_char, word_count, sentence_count)
          - fts_chunks(rowid, text)  (si tenés FTS5 con external content; si no, podés omitir este insert)
        """
        # Insert documento (adaptá columnas si difieren en tu esquema)
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        cur = self.conn.execute(
            "INSERT INTO documents(title, source_path, meta_json) VALUES (?, ?, ?)",
            (title, source_path, meta_json)
        )
        doc_id = int(cur.lastrowid)

        chunks: List[Chunk] = chunk_text(text, min_words=min_words, target_words=target_words, max_words=max_words)
        rows = []
        for i, ch in enumerate(chunks, 1):
            rows.append((doc_id, i, ch.text, ch.start_char, ch.end_char, ch.word_count, ch.sentence_count))

        self.conn.executemany(
            "INSERT INTO chunks (doc_id, chunk_index, text, start_char, end_char, word_count, sentence_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows
        )
        self.conn.commit()

        # Espejo FTS si existiese (si tu FTS se llama distinto, adaptá)
        try:
            cr = self.conn.execute("SELECT chunk_id, text FROM chunks WHERE doc_id=?", (doc_id,)).fetchall()
            self.conn.executemany("INSERT INTO fts_chunks(rowid, text) VALUES (?, ?)",
                                  [(r["chunk_id"], r["text"]) for r in cr])
            self.conn.commit()
        except Exception:
            # Si no existe fts_chunks o es distinto, lo ignoramos para no romper nada
            pass

        return doc_id

    # ---- Embeddings: generar sólo los que falten (MiniLM-L6-v2)
    def embed_missing_minilm(self, batch_size: int = 128) -> Tuple[str, int, int]:
        """
        Genera embeddings para chunks que NO tienen vector en 'embeddings'.
        Retorna: (backend_name, dim, cantidad_generada)
        Asume tabla embeddings(chunk_id PRIMARY KEY, dim, vector BLOB).
        """
        backend = SentenceTFBackendMiniLM("all-MiniLM-L6-v2")
        to_do = self.conn.execute("""
            SELECT c.chunk_id, c.text
            FROM chunks c
            WHERE NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.chunk_id = c.chunk_id)
            ORDER BY c.chunk_id
        """).fetchall()
        if not to_do:
            return (backend.name, backend.dim, 0)

        i, total = 0, 0
        while i < len(to_do):
            batch = to_do[i:i+batch_size]
            vecs = backend.embed([r["text"] for r in batch])  # (B, dim)
            data = []
            for r, v in zip(batch, vecs):
                data.append((r["chunk_id"], backend.dim, _np_to_blob(v)))
            self.conn.executemany("INSERT OR REPLACE INTO embeddings(chunk_id, dim, vector) VALUES (?, ?, ?)", data)
            self.conn.commit()
            total += len(batch)
            i += batch_size
        return (backend.name, backend.dim, total)

    # ---- FAISS: anexar sólo lo que falte
    def faiss_append_missing(self, index_path: str = "faiss_index.bin", meta_path: str = "faiss_metadata.json",
                             model_name: str = "st:all-MiniLM-L6-v2") -> Tuple[int, int]:
        """
        Anexa al índice FAISS existente los embeddings que todavía no estén en 'ids' del metadata.
        Retorna: (agregados, total_ids_final)
        """
        if not _HAS_FAISS:
            raise RuntimeError("FAISS no está disponible (pip install faiss-cpu).")

        fm = FaissManager(index_path, meta_path)
        fm.load()

        # Ver chunk_ids ya indexados en FAISS
        already = fm.known_ids()

        # Levantar todos los embeddings (chunk_id, dim, vector)
        rows = self.conn.execute("SELECT e.chunk_id, e.dim, e.vector FROM embeddings e ORDER BY e.chunk_id").fetchall()
        if not rows:
            return (0, len(already))

        # Detectar faltantes
        missing_rows = [r for r in rows if int(r["chunk_id"]) not in already]
        if not missing_rows:
            return (0, len(already))

        dim = int(missing_rows[0]["dim"])
        vecs = np.vstack([_blob_to_np(r["vector"], dim) for r in missing_rows]).astype(np.float32)
        ids = [int(r["chunk_id"]) for r in missing_rows]

        fm.add(vecs, ids, model_name=model_name, dim=dim)
        fm.save()
        return (len(ids), len(fm.meta.get("ids", [])))

    # ---- Búsquedas (compatibles con tu esquema)
    def search_fts(self, query: str, k: int = 10) -> List[sqlite3.Row]:
        return self.conn.execute("""
            SELECT c.chunk_id, c.doc_id, c.chunk_index, c.text,
                   highlight(fts_chunks, 0, '[', ']') AS snippet
            FROM fts_chunks
            JOIN chunks c ON c.chunk_id = fts_chunks.rowid
            WHERE fts_chunks MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, k)).fetchall()

    def search_vector_bruteforce(self, query: str, k: int = 10) -> List[sqlite3.Row]:
        """
        Búsqueda semántica sin FAISS (para validar). Usa el mismo backend MiniLM.
        """
        backend = SentenceTFBackendMiniLM("all-MiniLM-L6-v2")
        qv = backend.embed_one(query).astype(np.float32, copy=False)

        rows = self.conn.execute("""
            SELECT e.chunk_id, e.dim, e.vector, c.doc_id, c.chunk_index, c.text
            FROM embeddings e
            JOIN chunks c ON c.chunk_id=e.chunk_id
        """).fetchall()
        if not rows:
            return []

        dim = int(rows[0]["dim"])
        mat = np.vstack([_blob_to_np(r["vector"], dim) for r in rows])
        # Similaridad coseno via normalización + dot
        q = qv.copy()
        q /= (np.linalg.norm(q) + 1e-12)
        X = mat.copy()
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        sims = X @ q
        top = np.argsort(-sims)[:k]
        return [rows[int(i)] for i in top]
