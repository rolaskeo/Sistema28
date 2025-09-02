# s28/ingestor_v3.py
# -*- coding: utf-8 -*-
"""
Ingesta V3 con spaCy: PDF / TXT / DOCX -> documents_v3, chunks_v3, embeddings_v3.
- Sentence chunking (~100–120 palabras), preservando citas y abreviaturas (spaCy).
- Idempotente por documento (borra chunks/embeddings previos del doc y reinsertar).
- Metadatos mínimos: path (UNIQUE), title (si podemos inferir), issuer/author opcional.
- PDF: guarda número de página; TXT/DOCX: page = NULL.
- Embeddings: all-MiniLM-L6-v2 (384d) con sentence-transformers.
"""

import os, sys, argparse, glob, sqlite3, hashlib, subprocess

from typing import List, Tuple, Optional
import re

# ---- Dependencias de extracción/embeddings ----
# pip install spacy sentence-transformers pymupdf python-docx Unidecode
import spacy
from unidecode import unidecode

# PDF
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# DOCX
try:
    import docx
except ImportError:
    docx = None

# Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np

# ------------------ utilidades ------------------

def norm_text(s: str) -> str:
    """Normalización suave para FTS: minúsculas + sin tildes."""
    s = s or ""
    s = s.lower()
    s = unidecode(s)
    # compactar espacios
    s = re.sub(r"\s+", " ", s).strip()
    return s

def chunk_sentences(sentences: List[str], min_words: int, max_words: int) -> List[str]:
    """Agrupa oraciones en chunks de ~min..max palabras."""
    chunks = []
    buf = []
    wc = 0
    for sent in sentences:
        w = len(sent.split())
        if wc + w > max_words and wc >= min_words:
            chunks.append(" ".join(buf).strip())
            buf, wc = [sent], w
        else:
            buf.append(sent)
            wc += w
    if buf:
        chunks.append(" ".join(buf).strip())
    return [c for c in chunks if len(c.split()) >= max_words * 0.3]  # evita fragmentos ínfimos

def ensure_schema(conn: sqlite3.Connection):
    """Asegura que exista el esquema V3 (por si alguien corre directo este módulo)."""
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents_v3 (
      id INTEGER PRIMARY KEY,
      path TEXT UNIQUE,
      title TEXT,
      author TEXT,
      issuer TEXT,
      year INTEGER,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks_v3 (
      chunk_id INTEGER PRIMARY KEY,
      doc_id INTEGER NOT NULL,
      chunk_index INTEGER NOT NULL,
      text TEXT NOT NULL,
      text_norm TEXT,
      page INTEGER,
      offset_start INTEGER,
      offset_end INTEGER,
      law_refs TEXT,
      FOREIGN KEY(doc_id) REFERENCES documents_v3(id)
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings_v3 (
      chunk_id INTEGER PRIMARY KEY,
      model TEXT,
      dim INTEGER NOT NULL,
      vector BLOB NOT NULL,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY(chunk_id) REFERENCES chunks_v3(chunk_id)
    )""")
    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks_v3 USING fts5(
      text_norm,
      content='chunks_v3',
      content_rowid='chunk_id',
      tokenize='unicode61 remove_diacritics 2'
    )""")
    # triggers FTS (best-effort)
    cur.execute("""
    CREATE TRIGGER IF NOT EXISTS chunks_v3_ai AFTER INSERT ON chunks_v3 BEGIN
      INSERT INTO fts_chunks_v3(rowid, text_norm) VALUES (new.chunk_id, new.text_norm);
    END;""")
    cur.execute("""
    CREATE TRIGGER IF NOT EXISTS chunks_v3_ad AFTER DELETE ON chunks_v3 BEGIN
      INSERT INTO fts_chunks_v3(fts_chunks_v3, rowid, text_norm) VALUES('delete', old.chunk_id, old.text_norm);
    END;""")
    cur.execute("""
    CREATE TRIGGER IF NOT EXISTS chunks_v3_au AFTER UPDATE ON chunks_v3 BEGIN
      INSERT INTO fts_chunks_v3(fts_chunks_v3, rowid, text_norm) VALUES('delete', old.chunk_id, old.text_norm);
      INSERT INTO fts_chunks_v3(rowid, text_norm) VALUES (new.chunk_id, new.text_norm);
    END;""")
    conn.commit()

def upsert_document(conn: sqlite3.Connection, path: str, title: Optional[str]=None,
                    author: Optional[str]=None, issuer: Optional[str]=None,
                    year: Optional[int]=None) -> int:
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO documents_v3(path) VALUES(?)", (path,))
    # si existe, actualizamos metadatos si vienen
    cur.execute("SELECT id, title, author, issuer, year FROM documents_v3 WHERE path=?", (path,))
    row = cur.fetchone()
    doc_id = int(row[0])
    new_title = title if title else row[1]
    new_author = author if author else row[2]
    new_issuer = issuer if issuer else row[3]
    new_year = year if (year is not None) else row[4]
    cur.execute("""
      UPDATE documents_v3 SET title=?, author=?, issuer=?, year=? WHERE id=?
    """, (new_title, new_author, new_issuer, new_year, doc_id))
    conn.commit()
    return doc_id

def clear_doc_chunks(conn: sqlite3.Connection, doc_id: int):
    """Deja el documento en estado limpio (idempotente)."""
    cur = conn.cursor()
    cur.execute("SELECT chunk_id FROM chunks_v3 WHERE doc_id=?", (doc_id,))
    ids = [r[0] for r in cur.fetchall()]
    if ids:
        qmarks = ",".join(["?"]*len(ids))
        cur.execute(f"DELETE FROM embeddings_v3 WHERE chunk_id IN ({qmarks})", ids)
        cur.execute(f"DELETE FROM chunks_v3 WHERE chunk_id IN ({qmarks})", ids)
        conn.commit()
    # rebuild fts (best-effort)
    try:
        cur.execute("INSERT INTO fts_chunks_v3(fts_chunks_v3) VALUES('rebuild')")
        conn.commit()
    except Exception:
        pass

def embedder(model_name: str="all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    return vecs.astype(np.float32)

# ------------------ extracción por tipo ------------------

def extract_pdf(path: str) -> List[Tuple[str, int]]:
    """Devuelve lista de (texto, page_num 1-based) por página."""
    if not fitz:
        raise RuntimeError("PyMuPDF (fitz) no está instalado: pip install pymupdf")
    doc = fitz.open(path)
    out = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        out.append((text, i))
    doc.close()
    return out

def extract_txt(path: str) -> List[Tuple[str, Optional[int]]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    return [(txt, None)]

def extract_docx(path: str) -> List[Tuple[str, Optional[int]]]:
    if not docx:
        raise RuntimeError("python-docx no está instalado: pip install python-docx")
    d = docx.Document(path)
    txt = "\n".join(p.text for p in d.paragraphs)
    return [(txt, None)]

# ------------------ sentence chunking ------------------

def sentences_spacy(nlp, text: str) -> List[str]:
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text and s.text.strip()]
    # opcional: unir líneas quebradas raras
    return sents

# ------------------ ingest de un archivo ------------------

def ingest_file(conn: sqlite3.Connection, nlp, model: SentenceTransformer,
                path: str, min_words: int, max_words: int, normalize: bool):
    # 1) extraer por tipo
    p = path.lower()
    per_pages = []
    if p.endswith(".pdf"):
        per_pages = extract_pdf(path)  # [(text, page)]
    elif p.endswith(".txt"):
        per_pages = extract_txt(path)  # [(text, None)]
    elif p.endswith(".docx"):
        per_pages = extract_docx(path) # [(text, None)]
    else:
        print(f"  [skip] extensión no soportada: {path}")
        return

    # 2) upsert documento con título básico (de nombre de archivo)
    title_guess = os.path.splitext(os.path.basename(path))[0]
    doc_id = upsert_document(conn, path=path, title=title_guess)

    # 3) idempotencia: borrar chunks/embeddings previos del doc
    clear_doc_chunks(conn, doc_id)

    # 4) sentence chunking por página/bloque y acumulación de chunk_index global
    chunk_texts, chunk_pages = [], []
    for block_text, pg in per_pages:
        if not block_text or not block_text.strip():
            continue
        sents = sentences_spacy(nlp, block_text)
        for ch in chunk_sentences(sents, min_words=min_words, max_words=max_words):
            chunk_texts.append(ch)
            chunk_pages.append(pg)  # puede ser None

    if not chunk_texts:
        print(f"  [warn] sin texto/chunks: {path}")
        return

    # 5) embeddings
    vecs = embed_texts(model, chunk_texts)  # (N, dim)
    dim = int(vecs.shape[1])

    # 6) insertar chunks + embeddings
    cur = conn.cursor()
    for idx, (txt, pg) in enumerate(zip(chunk_texts, chunk_pages)):
        text_norm = norm_text(txt) if normalize else None
        cur.execute("""
          INSERT INTO chunks_v3(doc_id, chunk_index, text, text_norm, page, offset_start, offset_end, law_refs)
          VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)
        """, (doc_id, idx, txt, text_norm, pg))
        chunk_id = cur.lastrowid
        vec = vecs[idx].tobytes()
        cur.execute("""
          INSERT INTO embeddings_v3(chunk_id, model, dim, vector)
          VALUES (?, ?, ?, ?)
        """, (chunk_id, "all-MiniLM-L6-v2", dim, vec))
    conn.commit()

    # 7) rebuild fts (best-effort)
    try:
        cur.execute("INSERT INTO fts_chunks_v3(fts_chunks_v3) VALUES('rebuild')")
        conn.commit()
    except Exception:
        pass

    print(f"  [ok] {path} -> {len(chunk_texts)} chunks")

# ------------------ CLI ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    # Solo uno de --file o --repo
    ap.add_argument("--file", help="Archivo único a ingestar")
    ap.add_argument("--repo", help="Carpeta base para ingesta recursiva")
    ap.add_argument("--glob", default="**/*.*", help="Patrón glob recursivo (ej: **/*.pdf)")
    ap.add_argument("--spacy", default="es_core_news_md")
    ap.add_argument("--max-words", type=int, default=120)
    ap.add_argument("--min-words", type=int, default=60)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument(
        "--rebuild-faiss",
        action="store_true",
        help="Al finalizar, reconstruye FAISS desde V3 (faiss_index.bin + faiss_metadata.json)"
    )
    args = ap.parse_args()

    # abrir DB y asegurar esquema
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)

    # cargar spaCy y embeddings
    print(f"[ingestor_v3] cargando spaCy: {args.spacy}")
    nlp = spacy.load(args.spacy)
    nlp.add_pipe("sentencizer", config={"punct_chars": [".", "?", "!", ";", "…"]}, first=True) if "sentencizer" not in nlp.pipe_names else None

    print("[ingestor_v3] cargando embeddings: all-MiniLM-L6-v2 (384d)")
    model = embedder("all-MiniLM-L6-v2")

    # rutas
    files = []
    if args.file:
        files = [os.path.abspath(args.file)]
    elif args.repo:
        base = os.path.abspath(args.repo)
        pattern = os.path.join(base, args.glob)
        files = glob.glob(pattern, recursive=True)
        files = [f for f in files if os.path.isfile(f)]
    else:
        raise SystemExit("Debe indicar --file o --repo")

    # filtrar extensiones soportadas
    SUP = {".pdf", ".txt", ".docx"}
    files = [f for f in files if os.path.splitext(f)[1].lower() in SUP]
    if not files:
        print("[ingestor_v3] no se encontraron archivos para ingestar.")
        return

    print(f"[ingestor_v3] archivos a procesar: {len(files)}")
    for path in files:
        try:
            ingest_file(conn, nlp, model, path, args.min_words, args.max_words, args.normalize)
        except Exception as e:
            print(f"  [error] {path}: {e}")

    # --- rebuild FAISS opcional ---
    if args.rebuild_faiss:
        print("[ingestor_v3] Reconstruyendo FAISS desde V3…")
        py = sys.executable
        # Ruta al script: ../scripts/build_faiss_from_v3.py
        script = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),  # sube de s28/ a la raíz
            "scripts",
            "build_faiss_from_v3.py"
        )
        env = os.environ.copy()
        env["DATABASE_PATH"] = os.path.abspath(args.db)

        # (Opcional) si querés forzar salida en la carpeta de la DB:
        # base_dir = os.path.dirname(os.path.abspath(args.db))
        # env["FAISS_INDEX_PATH"]    = os.path.join(base_dir, "faiss_index.bin")
        # env["FAISS_METADATA_PATH"] = os.path.join(base_dir, "faiss_metadata.json")

        subprocess.run([py, script], check=True, env=env)

    print("[ingestor_v3] FIN.")

if __name__ == "__main__":
    main()
