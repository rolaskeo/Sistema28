"""
S28 – Paso 2 (con notas al pie y número de página): Carga + Exploración/Descarga (Streamlit)

Ejecutar:
  streamlit run streamlit_carga.py

Requisitos (instalar una vez):
  pip install streamlit python-dotenv sentence-transformers faiss-cpu pypdf pymupdf

Lee variables desde .env (en la raíz del proyecto):
  DATABASE_PATH, FAISS_INDEX_PATH, FAISS_METADATA_PATH, EMBEDDING_MODEL
  (opcional) BACKUPS_DIR, BACKUP_RETENTION, UPLOADS_DIR

Novedad de esta versión:
- **Separación de texto principal vs. pie de página** (PyMuPDF) para PDFs con notas.
- **Inyección inline** del texto de cada nota al pie cuando se detecta un marcador en el chunk.
- **Tabla `footnotes`** en la DB (document_id, page, marker, texto) + guardado del **número de página** en cada chunk.

Funcionalidad existente:
- Página **Inicio** con métricas y accesos.
- Página **Cargar**: Título/Tipo obligatorios, PDF(s), Palabras clave opcionales, Valoración por tipo.
  • Dedup por doc_hash (PDF) y por chunk_hash (texto de chunk).  
  • Remueve del texto las líneas iguales al Título (evita que sea chunk).  
  • Guarda PDF en `UPLOADS_DIR` con nombre basado en hash.  
  • Actualiza FAISS + metadatos con swap atómico.  
  • Limpia el formulario tras cargar.
- Página **Explorar**:  
  a) Conteo total de documentos.  
  b) Conteo por tipo.  
  c) Búsqueda por **título** (substring, case-insensitive).  
  d) **Descarga** del PDF almacenado.
- (Opcional) Backup rotativo diario de DB + FAISS (3 archivos) con retención.
"""
from __future__ import annotations

import io
import os
import json
import subprocess
import hashlib
import sqlite3
import re
import unicodedata

from pathlib import Path
from datetime import datetime, date
from typing import List

import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # carga .env desde el cwd
import sys, sentence_transformers, transformers
print("[S28] PY:", sys.executable)
print("[S28] ST:", sentence_transformers.__version__)
print("[S28] TR:", transformers.__version__)

from pathlib import Path
from dotenv import load_dotenv
import os, sys

# Directorio de la app (NO el cwd)
APP_DIR = Path(__file__).resolve().parent

# Cargar .env desde la carpeta de la app
load_dotenv(APP_DIR / ".env")

# ---- Ruta de DB: única fuente de verdad ----
_db_env = os.environ.get("DATABASE_PATH") or os.environ.get("S28_DB")
if not _db_env:
    _db_env = APP_DIR / "sistema28.db"  # default junto a la app
DB_PATH = str(Path(_db_env).expanduser().resolve())
DATABASE_PATH = DB_PATH  # alias para compatibilidad con llamadas existentes
print("[S28] DB:", DB_PATH)

# ---- Rutas de carpetas normalizadas (absolutas) ----
def _abs_under_app(p):
    p = Path(p)
    return str(p if p.is_absolute() else (APP_DIR / p).resolve())

UPLOADS_DIR     = _abs_under_app(os.getenv("UPLOADS_DIR", "uploads"))
BACKUPS_DIR     = _abs_under_app(os.getenv("BACKUPS_DIR", "backups"))
DICTAMENES_DIR  = _abs_under_app(os.getenv("DICTAMENES_DIR", "dictamenes"))
DICTAMEN_ESTRUCTURA_PATH = _abs_under_app(os.getenv("DICTAMEN_ESTRUCTURA", "plantilla/estructura.json"))

# Asegurar import del paquete s28 aunque el cwd sea otro
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# --- v3: búsqueda híbrida + advisor ---
import sys, os, sqlite3, importlib
# APP_DIR ya fue agregado a sys.path arriba; no agregamos cwd para evitar paquetes sombra
import s28.rerank as _rerank
_rerank = importlib.reload(_rerank)   # fuerza tomar la versión actualizada del módulo
v3_search = _rerank.search

# Log de control (lo verás en la consola de Streamlit)
print("[S28] v3_search from:", getattr(_rerank, "__file__", "?"))


# Opcional para generar .docx en la página "Generar"
try:
    from docx import Document  # pip install python-docx
except Exception:  # pragma: no cover
    Document = None

# IA (DeepSeek) – importar pipeline de generación desde generar_dictamen.py
try:
    from generar_dictamen import (
        get_topk_chunks,
        get_topk_chunks_v3,
        call_remote_model,
        associate_paragraphs_with_sources,
        create_docx_ptn,
        save_dictamen_record,
        MAX_CONTEXT_CHUNKS,
        MODEL_ID,
        # NUEVO:
        backfill_refs_from_v3,
    )
    HAVE_GEN = True
except Exception as e:  # pragma: no cover
    HAVE_GEN = False
    GEN_IMPORT_ERR = str(e)

# PDF readers
try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit("Instalá pypdf o PyPDF2: pip install pypdf") from e

# PyMuPDF (layout-aware) para detectar pie de página
if os.getenv("DISABLE_FITZ", "0").lower() in ("1", "true", "yes", "on"):
    HAVE_FITZ = False
else:
    try:
        import fitz  # PyMuPDF
        HAVE_FITZ = True
    except Exception:
        HAVE_FITZ = False


# Embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:  # pragma: no cover
    raise SystemExit("Falta sentence-transformers: pip install sentence-transformers") from e

load_dotenv()

FAISS_BIN = Path(os.getenv("FAISS_INDEX_PATH", "faiss_index.bin"))
FAISS_META = Path(os.getenv("FAISS_METADATA_PATH", "faiss_metadata.json"))
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
BACKUPS_DIR = os.getenv("BACKUPS_DIR", "backups")
BACKUP_RETENTION = int(os.getenv("BACKUP_RETENTION", "7"))  # snapshots por fecha
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "uploads"))
DICTAMENES_DIR = Path(os.getenv("DICTAMENES_DIR", "dictamenes"))

# Tipos de documento disponibles
TIPOS = [
    "sentencia",
    "norma",
    "Doctrina",
    "opinion_propia",
    "dictamen_generado",
    "informe_tecnico",
    "otro",
]

# Valoración por tipo (peso por defecto)
TIPO_PESO = {
    "sentencia": 1.3,
    "norma": 1.2,
    "Doctrina": 0.9,
    "opinion_propia": 1.0,
    "dictamen_generado": 0.9,
    "informe_tecnico": 1.1,
    "otro": 1.0,
}

# --- Navegación global ---
if "nav" not in st.session_state:
    st.session_state["nav"] = "Inicio"

# --- Claves dinámicas para limpiar el formulario de Cargar ---
if "form_run_id" not in st.session_state:
    st.session_state["form_run_id"] = 0
fid = st.session_state["form_run_id"]

# ------------------------ Utilidades ------------------------

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256(); h.update(data); return h.hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def safe_filename(name: str) -> str:
    base = re.sub(r"[^\w\-\.]+", "_", name, flags=re.UNICODE)
    return re.sub(r"_+", "_", base).strip("._")


def ensure_uploads_dir() -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOADS_DIR


def store_pdf_bytes(data: bytes, original_name: str, d_hash: str) -> Path:
    """Guarda el PDF en UPLOADS_DIR como <hash12>_<safe_name>.pdf y devuelve la ruta."""
    ensure_uploads_dir()
    safe = safe_filename(original_name)
    out = UPLOADS_DIR / f"{d_hash[:12]}_{safe}"
    out.write_bytes(data)
    return out.resolve()


@st.cache_resource
def cargar_modelo():
    return SentenceTransformer(EMB_MODEL)


def abrir_db():
    # Abrimos una conexión NUEVA por llamada (sin cachear) y apta para hilos de Streamlit
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=60000;")  # esperar hasta 60s si hay lock
    # Crear tablas base si no existen
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            titulo TEXT NOT NULL,
            tipo TEXT NOT NULL,
            organismo TEXT,
            fecha TEXT,
            path TEXT,
            doc_hash TEXT UNIQUE,
            created_at TEXT,
            keywords TEXT,
            valor_tipo REAL
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_text TEXT
        );
        """
    )

    # Migraciones: añadir columnas que falten
    def has_col(table, col):
        cur = con.execute(f"PRAGMA table_info({table});")
        return any(r[1] == col for r in cur.fetchall())

    # document_chunks: asegurar columnas requeridas (incluye page)
    for col, ddl in [
        ("document_id", "INTEGER"),
        ("chunk_text", "TEXT"),
        ("chunk_hash", "TEXT"),
        ("tipo", "TEXT"),
        ("posicion", "INTEGER"),
        ("page", "INTEGER"),  # <<--- NUEVO
    ]:
        if not has_col("document_chunks", col):
            con.execute(f"ALTER TABLE document_chunks ADD COLUMN {col} {ddl};")

    # Tabla de notas al pie
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS footnotes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            page INTEGER NOT NULL,
            marker INTEGER NOT NULL,
            texto TEXT,
            UNIQUE(document_id, page, marker)
        );
        """
    )

    # Índices
    con.execute("CREATE INDEX IF NOT EXISTS ix_documents_tipo ON documents(tipo);")
    con.execute("CREATE INDEX IF NOT EXISTS ix_chunks_docid ON document_chunks(document_id);")
    con.execute("CREATE INDEX IF NOT EXISTS ix_chunks_tipo ON document_chunks(tipo);")
    con.execute("CREATE INDEX IF NOT EXISTS ix_footnotes_doc ON footnotes(document_id);")

    # Índice único en chunk_hash (limpiando duplicados si hiciera falta)
    try:
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_chunks_hash ON document_chunks(chunk_hash);")
    except Exception:
        con.execute(
            """
            DELETE FROM document_chunks
            WHERE rowid NOT IN (
                SELECT MIN(rowid) FROM document_chunks GROUP BY chunk_hash
            );
            """
        )
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_chunks_hash ON document_chunks(chunk_hash);")

    con.commit()
    return con


@st.cache_resource
def cargar_faiss():
    """Carga índice FAISS y metadatos. Si no existe, crea uno nuevo."""
    import faiss
    if FAISS_BIN.exists():
        index = faiss.read_index(str(FAISS_BIN))
        meta = json.loads(FAISS_META.read_text(encoding="utf-8")) if FAISS_META.exists() else []
    else:
        # Dim por defecto para all-MiniLM-L6-v2
        index = faiss.IndexFlatIP(384)
        meta = []
    return index, meta


def guardar_faiss(index, meta: List[dict]):
    """Guarda FAISS+metadatos con swap atómico para evitar corrupción."""
    import faiss
    from tempfile import NamedTemporaryFile
    # Binario
    with NamedTemporaryFile(delete=False) as fbin:
        faiss.write_index(index, fbin.name)
        tmp_bin = Path(fbin.name)
    # JSON
    with NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as fmeta:
        json.dump(meta, fmeta, ensure_ascii=False, indent=2)
        tmp_meta = Path(fmeta.name)
    tmp_bin.replace(FAISS_BIN)
    tmp_meta.replace(FAISS_META)


def rotar_backups(db: str = DB_PATH, faiss_bin: Path = FAISS_BIN, faiss_meta: Path = FAISS_META,
                  carpeta: str = BACKUPS_DIR, retencion: int = BACKUP_RETENTION):
    """Crea un snapshot por día (si no existe) y elimina los más viejos."""
    bdir = Path(carpeta); bdir.mkdir(exist_ok=True)
    hoy = date.today().strftime("%Y%m%d")
    # si ya hay backup de hoy, no hacer nada
    if (bdir / f"s28_{hoy}.db").exists():
        return
    # copiar existentes
    if Path(db).exists():
        (bdir / f"s28_{hoy}.db").write_bytes(Path(db).read_bytes())
    if faiss_bin.exists():
        (bdir / f"s28_{hoy}.faiss").write_bytes(faiss_bin.read_bytes())
    if faiss_meta.exists():
        (bdir / f"s28_{hoy}.meta.json").write_bytes(faiss_meta.read_bytes())
    # limpieza de retención
    fechas = sorted({p.stem.split(".")[0] for p in bdir.glob("s28_*.*")}, reverse=True)
    for fecha in fechas[retencion:]:
        for f in bdir.glob(f"{fecha}.*"):
            try:
                f.unlink()
            except Exception:
                pass


# -------- Normalización / chunking --------

def pdf_a_texto(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    partes = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        partes.append(txt)
    return "\n".join(partes)


def normalizar_linea(s: str) -> str:
    s = s.strip().casefold()
    s = re.sub(r"[\s_]+", " ", s)
    s = re.sub(r"[\.,;:¡!¿\?\-–—\(\)\[\]\{\}\"']", "", s)
    return s.strip()


def limpiar_titulo_del_texto(texto: str, titulo: str) -> str:
    """Elimina líneas que coincidan exactamente con el título provisto."""
    if not titulo:
        return texto
    ntitulo = normalizar_linea(titulo)
    nuevas = []
    for line in texto.splitlines():
        if normalizar_linea(line) == ntitulo:
            continue
        nuevas.append(line)
    return "\n".join(nuevas)


def chunkear(texto: str, max_palabras=50) -> List[str]:
    # Corte por punto, recomponiendo hasta ~50 palabras
    oraciones = [o.strip() for o in texto.replace("\n", " ").split(".") if o.strip()]
    salida, bloque, cuenta = [], [], 0
    for o in oraciones:
        palabras = o.split()
        if cuenta + len(palabras) <= max_palabras:
            bloque.append(o)
            cuenta += len(palabras)
        else:
            if bloque:
                salida.append(". ".join(bloque) + ".")
            bloque, cuenta = [o], len(palabras)
    if bloque:
        salida.append(". ".join(bloque) + ".")
    return salida


# -------- Notas al pie (layout-aware) --------

FOOTNOTE_MARKER_RE = re.compile(r"(?:\[(\d{1,2})\]|(?<!\d)\^?(\d{1,2})(?!\d))")


def pdf_iter_pages_fitz(file_bytes: bytes):
    """
    Itera páginas devolviendo (page_no, main_text, footnotes_dict).
    Usa PyMuPDF para separar zona principal vs. pie de página por geometría y patrón.
    Si PyMuPDF no está disponible, cae a extracción plana con pypdf (sin notas).
    """
    if not HAVE_FITZ:
        reader = PdfReader(io.BytesIO(file_bytes))
        for i, p in enumerate(reader.pages, start=1):
            txt = p.extract_text() or ""
            yield i, txt, {}
        return

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pdict = page.get_text("dict")  # bloques/lineas/spans con coords y tamaños
        H = float(page.rect.height)
        foot_top = H * 0.82  # último ~18% como candidato a pie

        main_lines, foot_lines = [], []
        for block in pdict.get("blocks", []):
            for line in block.get("lines", []):
                y0 = min((s["bbox"][1] for s in line.get("spans", [])), default=0.0)
                text = "".join(s.get("text", "") for s in line.get("spans", []))
                text = text.strip()
                if not text:
                    continue
                looks_foot = (y0 >= foot_top) or re.match(r"^\s*\[?\d{1,2}\]?[)\.\-–—]\s+\S", text)
                if looks_foot:
                    foot_lines.append(text)
                else:
                    main_lines.append(text)

        main_text = "\n".join(main_lines)

        notas: dict[int, str] = {}
        for ln in foot_lines:
            m = re.match(r"^\s*\[?(\d{1,2})\]?[)\.\-–—]\s*(.+)$", ln)
            if m:
                n = int(m.group(1)); txt = m.group(2).strip()
                if n in notas:
                    notas[n] = (notas[n] + " " + txt).strip()
                else:
                    notas[n] = txt

        yield i + 1, main_text, notas  # páginas 1-indexed


def inyectar_notas_inline(chunk: str, notas: dict[int, str]) -> str:
    """Inserta ' [n.X: ...]' junto a cada marcador si existe la nota."""
    def repl(m):
        n = m.group(1) or m.group(2)
        try:
            n = int(n)
        except:
            return m.group(0)
        if n in notas and notas[n]:
            return f"{m.group(0)} [n.{n}: {notas[n]}]"
        return m.group(0)
    return FOOTNOTE_MARKER_RE.sub(repl, chunk)

# --------- Utilidades de citas / referencias ---------

def extraer_anio(fecha_str: str | None) -> str:
    """Devuelve solo el año (YYYY) a partir de 'YYYY-MM' o 'YYYY-MM-DD'.
    Si no puede parsear, devuelve 's/f' (sin fecha)."""
    if not fecha_str:
        return "s/f"
    try:
        # Intento ISO estricto
        y = datetime.fromisoformat(fecha_str[:10]).year
        return str(y)
    except Exception:
        m = re.search(r"(19|20)\d{2}", fecha_str)
        return m.group(0) if m else "s/f"

def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))

def _normalize_title_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = _strip_accents(s)
    # unificar "n°", "nº", "nro.", "n." → "n"
    s = re.sub(r"\b(n[º°o]\.?|nro\.?)\b", "n", s)
    # quitar puntos en números (13.064 → 13064)
    s = re.sub(r"(?<=\d)\.(?=\d)", "", s)
    # separadores comunes → espacio
    s = s.replace("_", " ").replace("-", " ")
    # quitar no alfanumético (deja espacio)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    # colapsar espacios
    s = re.sub(r"\s{2,}", " ", s).strip()
    # singularizaciones mínimas útiles
    singular = {
        "obras": "obra",
        "publicas": "publica",
        "leyes": "ley",
        "resoluciones": "resolucion",
        "decretos": "decreto",
        "contrataciones": "contratacion",
    }
    toks = [singular.get(t, t) for t in s.split()]
    return " ".join(toks)

def _match_title(query: str, title: str) -> bool:
    q = _normalize_title_text(query).split()
    t = set(_normalize_title_text(title).split())
    q_sig = [x for x in q if len(x) >= 2]
    return all(tok in t for tok in q_sig)

def _anchor_token_for_like(query: str) -> str:
    toks = _normalize_title_text(query).split()
    toks = [t for t in toks if len(t) >= 3]
    if not toks:
        return ""
    toks.sort(key=len, reverse=True)  # ancla = token más largo
    return toks[0]

# -------- Búsqueda semántica (sumarios) --------

def buscar_sumarios_semantico(consulta: str, top_k: int = 5, filtro_tipo: str | None = None):
    index, meta = cargar_faiss()
    if index.ntotal == 0 or not consulta.strip():
        return []
    model = cargar_modelo()
    q_emb = model.encode([consulta.strip()], normalize_embeddings=True).astype("float32")
    k = max(top_k * 4, 10)  # sobre-muestreo para re-rank
    D, I = index.search(q_emb, min(k, index.ntotal))
    resultados = []
    for idx, score in zip(I[0], D[0]):
        m = meta[idx]
        if filtro_tipo and m.get("tipo") != filtro_tipo:
            continue
        w = float(m.get("valor_tipo", 1.0))
        bonus = 0.0
        kws = (m.get("keywords") or "")
        if kws:
            kl = [k.strip().lower() for k in kws.split(",") if k.strip()]
            ch_low = (m.get("chunk_text", "").lower())
            if any(kk and kk in ch_low for kk in kl):
                bonus += 0.02
        final = float(score) * w + bonus
        resultados.append((final, m))
    resultados.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in resultados[:top_k]]

# --- NUEVO: buscar sumarios con v3 ---
def buscar_sumarios_v3(
    db_path: str,
    query: str,
    topk: int = 5,
    filtro_tipo: str | None = None,
    candidates: int = 200,
    max_per_doc: int | None = None,
    diversificar: bool = True,
    modo_general: bool = False,
    use_faiss: bool = True,
):
    import sqlite3
    from pathlib import Path

    conn = sqlite3.connect(db_path)
    try:        
        rows = v3_search(
            conn, query,
            topk=topk,
            candidates=candidates,
            with_snippet=True,
            filtro_tipo=filtro_tipo,
            max_per_doc=max_per_doc,
            diversificar=diversificar,
            modo_general=modo_general,
            use_faiss=use_faiss,
        )
    finally:
        conn.close()

    # --- Adaptación al esquema legacy de la UI ---
    hits = []
    for r in (rows or []):
        # Mapeos básicos
        titulo = r.get("title") or ""
        texto  = r.get("snippet") or r.get("text") or ""
        ruta   = r.get("path") or ""
        # Evitar el error de '.' : si no es un archivo, no pasar ruta
        try:
            p = Path(ruta) if ruta else None
            if not (p and p.is_file()):
                ruta = ""  # así Path("") -> '.', pero vamos a arreglar el exists() abajo
        except Exception:
            ruta = ""

        hits.append({
            "titulo": titulo,
            "chunk_text": texto,
            "page": r.get("page"),
            "tipo": r.get("tipo") or None,   # ← AHORA SÍ
            "organismo_emisor": r.get("issuer") or r.get("organismo") or "",
            "organismo": r.get("issuer") or "",
            "fecha_documento": r.get("year"),  # la UI llama extraer_anio(...)
            "document_path": ruta,
            # si tu UI usa otros campos, agregalos aquí
        })

    return hits

# --- NUEVO: buscar por título con v3 ---
def buscar_por_titulo_v3(db_path: str, texto: str, limit: int = 50):
    import sqlite3
    anchor = _anchor_token_for_like(texto)

    conn = sqlite3.connect(db_path)
    conn.row_factory = lambda c,r:{d[0]: r[i] for i,d in enumerate(c.description)}
    try:
        if anchor:
            # Prefiltrado ancho por LIKE (rápido) sobre versión suavizada del título
            cands = conn.execute("""
                SELECT
                    d.id   AS doc_id,
                    d.title AS titulo,
                    COALESCE(d.issuer,'') AS organismo,
                    COALESCE(d.author,'') AS autor,
                    COALESCE(d.year,'')   AS anio,
                    d.path,
                    (SELECT COUNT(*) FROM chunks_v3 WHERE doc_id=d.id) AS n_sumarios
                FROM documents_v3 d
                WHERE lower(replace(replace(d.title,'_',' '),'.','')) LIKE '%' || ? || '%'
                ORDER BY d.id DESC
                LIMIT 800
            """, (anchor.lower(),)).fetchall()
        else:
            cands = conn.execute("""
                SELECT
                    d.id   AS doc_id,
                    d.title AS titulo,
                    COALESCE(d.issuer,'') AS organismo,
                    COALESCE(d.author,'') AS autor,
                    COALESCE(d.year,'')   AS anio,
                    d.path,
                    (SELECT COUNT(*) FROM chunks_v3 WHERE doc_id=d.id) AS n_sumarios
                FROM documents_v3 d
                ORDER BY d.id DESC
                LIMIT 800
            """).fetchall()
    finally:
        conn.close()

    # Filtrado fino en Python con normalización robusta (obra/obras, acentos, 13.064/13064, etc.)
    resultados = [r for r in cands if _match_title(texto, r["titulo"])]

    # Si no hay hits, ampliamos el universo y reintentamos
    if not resultados:
        conn = sqlite3.connect(db_path)
        conn.row_factory = lambda c,r:{d[0]: r[i] for i,d in enumerate(c.description)}
        try:
            all_rows = conn.execute("""
                SELECT
                    d.id   AS doc_id,
                    d.title AS titulo,
                    COALESCE(d.issuer,'') AS organismo,
                    COALESCE(d.author,'') AS autor,
                    COALESCE(d.year,'')   AS anio,
                    d.path,
                    (SELECT COUNT(*) FROM chunks_v3 WHERE doc_id=d.id) AS n_sumarios
                FROM documents_v3 d
                ORDER BY d.id DESC
                LIMIT 3000
            """).fetchall()
            resultados = [r for r in all_rows if _match_title(texto, r["titulo"])]
        finally:
            conn.close()

    return resultados[:limit]


# ======================== UI / NAVEGACIÓN ========================

st.set_page_config(page_title="Sistema 28", layout="wide")

# ---- Estilos (inspirado en tu mock) ----
st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff; }
    .s28-title { text-align:center; font-size:64px; font-weight:800; margin: 0.5rem 0 2rem 0; }
    .s28-sub { text-align:center; color:#607d8b; margin-top:-1.2rem; }
    /* Botones grandes */
    .stButton>button {
        width: 100%;
        padding: 18px 28px;
        font-size: 20px;
        border-radius: 18px;
        border: 1px solid #cfd8dc;
        background: linear-gradient(180deg,#f8f9fa,#e9ecef);
        box-shadow: 0 2px 8px rgba(0,0,0,.06);
        transition: all .15s ease;
    }
    .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 8px 14px rgba(0,0,0,.10); }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.title("Sistema 28")
    nav = st.radio(
        "Navegación",
        ["Inicio", "Generar", "Buscar/Descargar", "Cargar"],
        index=["Inicio", "Generar", "Buscar/Descargar", "Cargar"].index(st.session_state["nav"]) if st.session_state.get("nav") in ["Inicio","Generar","Buscar/Descargar","Cargar"] else 0,
        label_visibility="collapsed",
    )
    st.session_state["nav"] = nav

# ------------------------ PÁGINA: INICIO ------------------------
if st.session_state["nav"] == "Inicio":
    st.markdown(
        '<div class="s28-title"><span style="color:#e53935">S</span><span style="color:#1e88e5">28</span></div>',
        unsafe_allow_html=True,
    )
    mid = st.columns([1,2,1])[1]
    with mid:
        if st.button("⚖️  Generar Dictamen jurídico", use_container_width=True):
            st.session_state["nav"] = "Generar"; st.rerun()
        st.write("")
        if st.button("🔎  Buscar (documentos o sumarios)", use_container_width=True):
            st.session_state["nav"] = "Buscar/Descargar"; st.rerun()
        st.write("")
        if st.button("📤  Cargar documentos nuevos", use_container_width=True):
            st.session_state["nav"] = "Cargar"; st.rerun()

    st.markdown("""
    <p class='s28-sub'>Accesos rápidos • Podés volver a esta pantalla desde la barra lateral</p>
    """, unsafe_allow_html=True)

# ------------------------ PÁGINA: GENERAR ------------------------
elif st.session_state["nav"] == "Generar":
    st.header("Generar Dictamen jurídico (IA)")
    if not HAVE_GEN:
        st.error("No pude importar el generador (generar_dictamen.py). Detalle: " + GEN_IMPORT_ERR)
        st.stop()

    with st.form("form_gen_ia"):
        consulta = st.text_area("Consulta jurídica", height=160, placeholder="Describí el caso, dudas y normativa aplicable…")
        try:
            k_default = int(os.getenv("MAX_CONTEXT_CHUNKS", str(MAX_CONTEXT_CHUNKS if 'MAX_CONTEXT_CHUNKS' in globals() else 5)))
        except Exception:
            k_default = 5
        k = st.slider("Antecedentes a recuperar (Top K)", 1, 10, k_default, 1)
        temperature = st.slider("Temperatura (creatividad)", 0.0, 1.0, 0.2, 0.1)
        gen = st.form_submit_button("Generar dictamen (.docx) con IA", use_container_width=True)

    if gen:
        if not consulta.strip():
            st.error("Ingresá la consulta."); st.stop()

        with st.spinner("📚 Buscando antecedentes relevantes (v3)…"):
            hits = get_topk_chunks_v3(DATABASE_PATH, consulta, k=k)

        if not hits:
            st.warning("No se encontraron antecedentes relevantes en el índice.")
            st.stop()

        # Construir contexto legible (igual que CLI)
        context_parts = []
        for i, hit in enumerate(hits, start=1):
            snippet = (hit.get("chunk_text") or "")[:1200]
            autor = (hit.get('organismo_emisor') or hit.get('organismo') or '—')
            titulo = hit.get('titulo','')
            anio = extraer_anio(hit.get('fecha_documento'))
            pag = hit.get('page','—')
            header = f"[{i}] {autor} — {titulo} ({anio})" + (f" | pág. {pag}" if (pag not in (None, '—')) else "")
            context_parts.append(header + "\n" + snippet)
        contexto_text = "\n\n".join(context_parts)

        # Cargar epígrafes desde JSON (estructura.json)
        try:
            with open(DICTAMEN_ESTRUCTURA_PATH, "r", encoding="utf-8") as f:
                esquema = (json.load(f).get("secciones") or [])
                esquema = [s for s in esquema if isinstance(s, str) and s.strip()]
        except Exception:
            esquema = []
        if not esquema:
            esquema = ["I. Antecedentes", "II. Análisis", "III. Conclusión"]
        epigrafes = "\n- " + "\n- ".join(esquema)


        system_prompt = (
            "Sos un abogado experto en derecho administrativo argentino. "
            "Utilizá únicamente el contexto que te brindo para redactar un proyecto de dictamen jurídico. "
            "No inventes normas ni citas: si una norma es mencionada, debe estar textual en el contexto provisto. "
            "Estructurá el dictamen exactamente con estos epígrafes:" + epigrafes + " "
            "Usá citas en formato [n] donde n refiere a las fuentes recuperadas. "
            "Aplicá el Manual de Estilo PTN 2023 en lo que corresponda."
        )

        user_prompt = (
            f"Consulta: {consulta}\n\n"
            f"Contexto relevante:\n{contexto_text}\n\n"
            "Redactá el dictamen técnico-jurídico."
        )

        with st.spinner("🧠 Redactando con IA (DeepSeek)…"):
            try:
                response_text = call_remote_model(system_prompt, user_prompt, max_tokens=1200, temperature=float(temperature))
            except Exception as e:
                st.error(f"Error al llamar al modelo remoto: {e}")
                st.stop()

        st.subheader("Borrador")
        st.write(response_text)

        with st.spinner("🔎 Asociando párrafos con fuentes y generando Word (PTN)…"):
            associated = associate_paragraphs_with_sources(response_text)
            # NUEVO: completar referencias con v3 (hits)
            associated = backfill_refs_from_v3(associated, hits)

            try:
                filename = create_docx_ptn(consulta, associated)
            except Exception as e:
                st.error(f"Error creando el .docx: {e}"); st.stop()
            try:
                dictamen_id = save_dictamen_record(consulta, filename, MODEL_ID if 'MODEL_ID' in globals() else os.getenv('MODEL_ID','deepseek/deepseek-chat'), hits)
            except Exception as e:
                st.warning(f"Dictamen generado pero no se pudo guardar el registro en DB: {e}")
                dictamen_id = None

        st.success((f"Dictamen generado y guardado (ID {dictamen_id})." if dictamen_id else "Dictamen generado."))
        try:
            with open(filename, "rb") as fh:
                st.download_button("⬇️ Descargar dictamen (.docx)", data=fh.read(), file_name=os.path.basename(filename), mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
        except Exception as e:
            st.warning(f"No pude ofrecer la descarga directa: {e}")

# ------------------------ PÁGINA: CARGAR ------------------------
elif st.session_state["nav"] == "Cargar":

    st.header("Carga e indexación de documentos")

    with st.form("form_carga"):
        archivos = st.file_uploader("Elegí uno o varios archivos (pdf, word o txt)", type=["pdf","docx","txt"], accept_multiple_files=True, key=f"updf_{fid}")
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            titulo = st.text_input("Título (obligatorio)", value="", max_chars=250, key=f"titulo_{fid}")
        with col2:
            tipo = st.selectbox("Tipo (obligatorio)", options=TIPOS, index=None, placeholder="Seleccioná…", key=f"tipo_{fid}")
        with col3:
            fecha_doc = st.date_input(
            "Fecha del documento",
            value=date.today(),
            min_value=date(1900, 1, 1),
            max_value=date.today(),
            key=f"fecha_{fid}"
        )
        organismo = st.text_input("Organismo (opcional)", value="", key=f"org_{fid}")
        autor = st.text_input("Autor (opcional)", value="", key=f"autor_{fid}")   # ← NUEVO
        keywords_input = st.text_input("Palabras clave (coma separadas, opcional)", value="", key=f"kw_{fid}")
        # slider de valoración (peso) con default según tipo
        valor_def = TIPO_PESO.get(tipo, 1.0) if tipo else 1.0
        valor_tipo = st.slider("Valoración del tipo (peso)", 0.5, 2.0, float(valor_def), 0.1, key=f"valor_{fid}")
        submitted = st.form_submit_button("Procesar e indexar", use_container_width=True)

    if submitted:
        if not archivos:
            st.error("Subí al menos un archivo.")
            st.stop()
        if not titulo or tipo is None:
            st.error("Completá Título y Tipo.")
            st.stop()

        # Normalizar palabras clave a una cadena 'a, b, c'
        kw_str = ",".join([k.strip() for k in (keywords_input or "").replace(";", ",").split(",") if k.strip()])

        con = abrir_db(); cur = con.cursor()
        index, meta = cargar_faiss()
        model = cargar_modelo()

        nuevos_docs = 0
        nuevos_chunks = 0

        # ver dimensión esperada del modelo (para detectar FAISS incompatible)
        try:
            d_model = model.get_sentence_embedding_dimension()
        except Exception:
            d_model = 384

        if index.ntotal == 0 and hasattr(index, 'd') and index.d != d_model:
            import faiss
            st.warning("Recreando índice FAISS para ajustar la dimensión de embeddings.")
            index = faiss.IndexFlatIP(d_model)
            meta = []

        for f in archivos:
            saved_path = None
            existing_path = None     
            contenido = f.read()
            d_hash = sha256_bytes(contenido)     

            # ¿Existe ya ese documento? dedup por hash del PDF
            cur.execute("SELECT id, path FROM documents WHERE doc_hash=?", (d_hash,))
            row = cur.fetchone()
            if row:
                doc_id, existing_path = row
                p = Path(existing_path) if existing_path else None
                if not p or not p.exists():
                    try:
                        saved_path = store_pdf_bytes(contenido, f.name, d_hash)
                        cur.execute(
                            "UPDATE documents SET path=? WHERE id=?",
                            (str(Path(saved_path).resolve()), doc_id),
                        )

                        st.info(f"Documento existente: se restauró el archivo en disco ({saved_path.name}).")
                    except Exception as e:
                        st.warning(f"Documento existente pero no se pudo restaurar el archivo: {e}")
                else:
                    st.info(f"Saltado (duplicado exacto por hash): {f.name}")
            else:
                # Guardar archivo en UPLOADS_DIR y registrar path
                try:
                    saved_path = store_pdf_bytes(contenido, f.name, d_hash)
                except Exception as e:
                    st.error(f"No se pudo guardar el PDF: {e}")
                    continue

                # UPSERT por doc_hash: si ya existe, actualiza metadatos básicos y path
                cur.execute(
                    """
                    INSERT INTO documents (titulo, tipo, organismo, fecha, path, doc_hash, created_at, keywords, valor_tipo)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(doc_hash) DO UPDATE SET
                        titulo=excluded.titulo,
                        tipo=excluded.tipo,
                        organismo=excluded.organismo,
                        fecha=excluded.fecha,
                        path=excluded.path,
                        keywords=excluded.keywords,
                        valor_tipo=excluded.valor_tipo
                    """,
                    (
                        titulo.strip(),
                        tipo,
                        organismo.strip(),
                        fecha_doc.isoformat(),
                        str(saved_path),   # ← ya es absoluto por A). Si no hiciste A), usa str(Path(saved_path).resolve())
                        d_hash,
                        datetime.now().isoformat(),
                        kw_str,
                        float(valor_tipo),
                    ),
                )


                # Obtener id por hash (sirve tanto inserción como actualización)
                row_id = cur.execute("SELECT id FROM documents WHERE doc_hash = ?", (d_hash,)).fetchone()
                doc_id = row_id[0] if row_id else None            

                nuevos_docs += 1

            con.commit()  # asegura que la escritura en 'documents' esté visible antes del subproceso

            # --- indexación v3 para este archivo (siempre corre; es idempotente) ---
            try:
                ingest_target = str(Path(saved_path or existing_path).resolve())
                if ingest_target:
                    cmd = [
                        "python", "-m", "s28.ingestor_v3",
                        "--db", str(DB_PATH),
                        "--file", ingest_target,
                        "--spacy", "es_core_news_md",
                        "--min-words", "60",
                        "--max-words", "120",
                        "--normalize",
                    ]
                    # Ocultar ventana de consola en Windows
                    creationflags = 0
                    if os.name == "nt":
                        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

                    res = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        creationflags=creationflags,  # ← esto evita que se abra el cmd
                    )

                    if res.returncode != 0:
                        st.warning(f"Ingestor v3 falló: {res.stderr.strip() or 'error desconocido'}")
                    else:
                        st.info("Ingestor v3: OK (documento indexado en tablas v3).")
                        try:
                            nice_title = titulo.strip()
                            issuer_ui  = organismo.strip()
                            author_ui  = autor.strip()   # viene del field nuevo (puede ser "")
                            year_ui    = str(fecha_doc.year) if fecha_doc else ""

                            with sqlite3.connect(DB_PATH) as c2:
                                c2.execute("""
                                    UPDATE documents_v3
                                    SET
                                        title  = ?,
                                        issuer = CASE WHEN ? <> '' THEN ? ELSE issuer END,
                                        author = CASE WHEN ? <> '' THEN ? ELSE author END,
                                        year   = CASE WHEN ? <> '' THEN ? ELSE year END
                                    WHERE path = ?
                                """, (
                                    nice_title,
                                    issuer_ui, issuer_ui,
                                    author_ui, author_ui,
                                    year_ui,   year_ui,
                                    ingest_target,
                                ))
                        except Exception as e:
                            st.warning(f"No se pudieron actualizar metadatos v3: {e}")                        
                else:
                    st.warning("No se pudo determinar la ruta a ingestar en v3.")
            except Exception as e:
                st.warning(f"No se pudo ejecutar el ingestor v3: {e}")
            # --- fin v3 ---

            # ----------- NUEVO: procesar por página con notas al pie -----------
            # Procesamiento legacy SOLO si es PDF (DOCX/TXT los maneja v3)
            ext = (Path(f.name).suffix or "").lower()
            if ext == ".pdf":
                # ----------- Procesar por página con notas al pie (solo PDF) -----------
                for page_no, page_text, notas in pdf_iter_pages_fitz(contenido):
                    page_text = limpiar_titulo_del_texto(page_text, titulo)

                    # Guardar notas en tabla (no interrumpe si falla)
                    if notas:
                        try:
                            for marker, ntext in notas.items():
                                cur.execute(
                                    "INSERT OR IGNORE INTO footnotes (document_id, page, marker, texto) VALUES (?, ?, ?, ?)",
                                    (doc_id, page_no, int(marker), ntext),
                                )
                        except Exception:
                            pass

                    for i, ch in enumerate(chunkear(page_text, max_palabras=50)):
                        ch = ch.strip()
                        if not ch:
                            continue
                        ch = inyectar_notas_inline(ch, notas)
                        c_hash = sha256_text(ch)

                        inserted = False
                        try:
                            cur.execute(
                                """
                                INSERT INTO document_chunks (document_id, chunk_text, chunk_hash, tipo, posicion, page)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (doc_id, ch, c_hash, tipo, i, page_no),
                            )
                            inserted = True
                        except sqlite3.IntegrityError:
                            inserted = False  # chunk duplicado, seguimos con el próximo

                        if inserted:
                            # Embedding legacy SOLO para chunks realmente insertados
                            text_for_emb = f"{ch} [KW: {kw_str}]" if kw_str else ch
                            emb = model.encode([text_for_emb], normalize_embeddings=True)

                            if hasattr(index, 'd') and index.d != emb.shape[1]:
                                import faiss
                                st.warning("Dimensión de FAISS distinta a la del modelo. Se recrea el índice y metadatos.")
                                index = faiss.IndexFlatIP(emb.shape[1])
                                meta = []

                            index.add(np.ascontiguousarray(emb.astype("float32")))
                            meta.append({
                                "document_id": doc_id,
                                "titulo": titulo.strip(),
                                "tipo": tipo,
                                "posicion": i,
                                "page": page_no,
                                "hash": c_hash,
                                "valor_tipo": float(valor_tipo),
                                "keywords": kw_str,
                                "chunk_text": ch,
                                "document_path": str(saved_path or existing_path or ""),
                                "tipo_documento": tipo,
                                "organismo_emisor": organismo.strip(),
                                "fecha_documento": fecha_doc.isoformat(),
                            })
                            nuevos_chunks += 1


            # ----------- FIN NUEVO -----------

        # Persistir cambios
        con.commit(); con.close()
        if nuevos_chunks:
            guardar_faiss(index, meta)
            # backup diario (opcional)
            try:
                rotar_backups()
            except Exception:
                pass

        st.success(f"Carga completa • Documentos nuevos: {nuevos_docs} • Chunks nuevos: {nuevos_chunks}")

        # Limpiar el formulario para evitar recarga accidental
        st.session_state["form_run_id"] += 1
        st.rerun()

# ------------------------ PÁGINA: EXPLORAR ------------------------
elif st.session_state["nav"] == "Buscar/Descargar":
    st.header("Buscar / Descargar (documentos o sumarios)")
    con = abrir_db(); cur = con.cursor()

    # --- MÉTRICAS: documentos y sumarios (legacy + v3) ---

    def _count_safe(cur, sql, params=()):
        try:
            return cur.execute(sql, params).fetchone()[0]
        except Exception:
            return 0

    # Documentos
    legacy_total = _count_safe(cur, "SELECT COUNT(*) FROM documents")
    v3_total     = _count_safe(cur, "SELECT COUNT(*) FROM documents_v3")

    # Total únicos por path (evita doble conteo)
    total_unicos = _count_safe(cur, """
        SELECT COUNT(*) FROM (
            SELECT path FROM documents WHERE path IS NOT NULL
            UNION
            SELECT path FROM documents_v3 WHERE path IS NOT NULL
        ) AS t
    """)

    # Sumarios (chunks)
    chunks_v3     = _count_safe(cur, "SELECT COUNT(*) FROM chunks_v3")
    chunks_legacy = _count_safe(cur, "SELECT COUNT(*) FROM document_chunks")  # por si existe la tabla legacy

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Documentos cargados por Interfaz", legacy_total)
    with c2:
        st.metric("Documentos ingeridos por Sistema (s28 v3)", v3_total)
    with c3:
        st.metric("Documentos Totales", total_unicos)
    with c4:
        st.metric("Sumarios disponibles (chunks v3)", chunks_v3)

    # (Opcional) mostrar sumarios legacy si existen
    if chunks_legacy:
        st.caption(f"Sumarios legacy (FAISS): {chunks_legacy}")

    with st.expander("Desglose (opcional)"):
        por_tipo_legacy = []
        try:
            por_tipo_legacy = cur.execute("""
                SELECT COALESCE(tipo,'(Sin tipo)'), COUNT(*)
                FROM documents
                GROUP BY COALESCE(tipo,'(Sin tipo)')
                ORDER BY COUNT(*) DESC
            """).fetchall()
        except Exception:
            pass

        por_org_v3 = []
        try:
            por_org_v3 = cur.execute("""
                SELECT COALESCE(issuer,'(Sin organismo)'), COUNT(*)
                FROM documents_v3
                GROUP BY COALESCE(issuer,'(Sin organismo)')
                ORDER BY COUNT(*) DESC
                LIMIT 12
            """).fetchall()
        except Exception:
            pass

        if por_tipo_legacy:
            st.write("**Legacy por tipo**:")
            for t, c in por_tipo_legacy:
                st.write(f"- **{t}**: {c}")

        if por_org_v3:
            st.write("**v3 por organismo**:")
            for org, c in por_org_v3:
                st.write(f"- **{org}**: {c}")
        # --- fin MÉTRICAS ---


    # -------- BÚSQUEDA POR TÍTULO --------
    st.subheader("Buscar por título")
    with st.form("form_buscar_titulo"):
        bt_col1, bt_col2 = st.columns([3,1])
        with bt_col1:
            q = st.text_input(
                "Texto a buscar en títulos",
                value="",
                placeholder="Ej.: naturaleza del contrato… o expte 123",
                key="q_tit",
            )
        with bt_col2:
            filtro_tipo = st.selectbox(
                "Filtrar por tipo",
                options=["(Todos)"] + TIPOS,
                index=0,
                key="filtro_tit",
            )
        go_title = st.form_submit_button("Buscar títulos", use_container_width=True)

    if go_title:
        resultados_titulo = buscar_por_titulo_v3(DATABASE_PATH, q, limit=50)
        if not resultados_titulo:
            st.info("Sin coincidencias en títulos (v3).")
        else:
            for r in resultados_titulo:
                doc_id  = r["doc_id"]
                titulo  = r["titulo"]
                org     = r["organismo"] or "—"
                autor   = r["autor"] or "—"
                anio    = r["anio"] or "—"
                n_sum   = r["n_sumarios"]
                path    = r["path"]

                header = f"#{doc_id} — {titulo} · {anio}"
                with st.expander(header):
                    st.caption(f"Organismo: {org} · Autor: {autor} · Sumarios: {n_sum}")
                    if path:
                        try:
                            p = Path(path)
                            if p.exists():
                                st.download_button(
                                    label="⬇️ Descargar archivo",
                                    data=p.read_bytes(),
                                    file_name=p.name,
                                    use_container_width=True,
                                    key=f"dl_tit_{doc_id}",
                                )
                        except Exception as e:
                            st.warning(f"No se pudo ofrecer descarga: {e}")
                    else:
                        st.warning("Archivo no disponible en disco (registro antiguo o ruta inválida).")

    # -------- BÚSQUEDA POR SUMARIO (CONTENIDO) --------
    st.subheader("Buscar sumarios por contenido")
    with st.form("form_buscar_sumarios"):
        sq1, sq2, sq3 = st.columns([3,1,1])
        with sq1:
            qsum = st.text_input(
                "Texto a buscar en el contenido",
                value="",
                placeholder="Ej.: caducidad del contrato, interés público…"
            )
        with sq2:
            tipo_sem = st.selectbox("Filtrar tipo (contenido)", options=["(Todos)"] + TIPOS, index=0)
        with sq3:
            topk = st.slider("Top-K", 3, 10, 5, 1)

        # Controles nuevos
        colA, colB, colC, colD = st.columns([1,1,1,1])
        with colA:
            candidates = st.slider(
                "Candidatos (recall)",
                50, 400, 200, 10,
                help="Cuántos candidatos iniciales tomar (FTS + FAISS opcional)"
            )
        with colB:
            max_per_doc = st.slider(
                "Máx. por documento",
                1, 5, int(os.getenv("MAX_PER_DOC", "2")), 1
            )
        with colC:
            diversificar = st.checkbox("Diversificar (MMR)", value=True)
        with colD:
            modo_general = st.checkbox(
                "Modo general",
                value=False,
                help="Desactiva la heurística de meta (issuer/año/tokens) para consultas amplias"
            )

        use_ai = st.checkbox("Búsqueda inteligente (IA): 3–5 bullets con citas [i]", value=False)
        go_sem = st.form_submit_button("Buscar sumarios (contenido)", use_container_width=True)

    if go_sem:
        filtro = None if tipo_sem == "(Todos)" else tipo_sem
        hits = buscar_sumarios_v3(
            DATABASE_PATH,
            qsum,
            topk=topk,
            filtro_tipo=filtro,
            candidates=candidates,
            max_per_doc=max_per_doc,
            diversificar=diversificar,
            modo_general=modo_general,
            use_faiss=True,   # activa el recall extra por FAISS si está disponible
        )

        # >>> PEGAR AQUÍ: Advisor v3 (opcional, sin costo LLM)
        if hits:  # solo tiene sentido si hay resultados
            if st.checkbox("Mostrar advisor (panorama + posturas)", value=False, key="advisor_sum_v3"):
                from s28.summarize import summarize_chunks, format_sumario, mapa_posturas, format_posturas
                
                # Usamos el motor v3 directo para pedir un poco más de contexto (p.ej., top 20)
                conn = sqlite3.connect(DATABASE_PATH)
                conn.row_factory = lambda c,r:{d[0]: r[i] for i,d in enumerate(c.description)}
                rows_v3 = v3_search(conn, qsum, topk=20, candidates=80, with_snippet=True)
                conn.close()

                # Construimos el panorama y el mapa de posturas
                sumario = summarize_chunks(rows_v3, max_sentences=8)
                posturas = mapa_posturas(rows_v3)

                # Pintamos arriba de la lista de resultados
                st.markdown("### 🧭 Panorama")
                st.write(format_sumario(sumario))
                st.markdown("### ⚖️ Posturas")
                st.write(format_posturas(posturas))
        # <<< FIN Advisor v3

        if not hits:
            st.info("Sin resultados por contenido.")
        else:
            for i, h in enumerate(hits, start=1):
                titulo_h = h.get("titulo", "")
                page_h = h.get("page")
                tipo_h = h.get("tipo")
                ch = (h.get("chunk_text") or "")[:700]
                autor_h = (h.get("organismo_emisor") or h.get("organismo") or "—")
                anio_h = extraer_anio(h.get("fecha_documento"))
                pag_h = page_h if page_h else "—"
                st.markdown(f"**[{i}]** {autor_h} — *{titulo_h}* ({anio_h})" + (f", pág. {pag_h}" if pag_h != "—" else ""))
                st.caption(f"Tipo: {tipo_h}")
                st.write(ch)
                p = Path(h.get("document_path") or "")
                if p.is_file():
                    try:
                        st.download_button(
                            "⬇️ Descargar PDF",
                            data=p.read_bytes(),
                            file_name=p.name,
                            mime="application/pdf",
                            key=f"dl_sem_{i}",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.warning(f"No se pudo ofrecer descarga: {e}")
                st.divider()

            if use_ai and HAVE_GEN:
                parts = []
                for i, h in enumerate(hits, start=1):
                    autor = (h.get('organismo_emisor') or h.get('organismo') or '—')
                    titulo = h.get('titulo','')
                    anio = extraer_anio(h.get('fecha_documento'))
                    pag = h.get('page','—')
                    head = f"[{i}] {autor} — {titulo} ({anio})" + (f" | pág. {pag}" if pag != '—' else "")
                    body = (h.get('chunk_text') or '')[:700]
                    parts.append(head + "\n" + body)
                ctx = "\n\n".join(parts)

                system = (
                    "Sos un asistente jurídico. Resumí y cruzá la información SOLO del contexto numerado; "
                    "no inventes. Entregá 3–5 bullets concisos con citas [i]."
                )
                user = (
                    f"Consulta: {qsum}\n\nContexto numerado:\n{ctx}\n\n"
                    "Dame bullets del tipo: 'en [i] sostuviste X (opinión propia), apoyada en [j] (sentencia), "
                    "pero [k] (doctrina) presenta una postura distinta'."
                )

                with st.spinner("IA preparando síntesis con citas…"):
                    try:
                        ai_text = call_remote_model(system, user, max_tokens=500, temperature=0.2)
                        st.markdown("**Síntesis (IA):**")
                        st.write(ai_text)
                    except Exception as e:
                        st.warning(f"No se pudo invocar la IA: {e}")

    con.close()

    st.divider()
    if st.button("⬅ Volver al Inicio", use_container_width=True):
        st.session_state["nav"] = "Inicio"; st.rerun()
