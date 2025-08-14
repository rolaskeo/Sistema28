"""
S28 – Paso 2 (completo): Carga + Exploración/Descarga (Streamlit)

Ejecutar:
  streamlit run streamlit_carga.py

Requisitos (instalar una vez):
  pip install streamlit python-dotenv sentence-transformers faiss-cpu pypdf

Lee variables desde .env (en la raíz del proyecto):
  DATABASE_PATH, FAISS_INDEX_PATH, FAISS_METADATA_PATH, EMBEDDING_MODEL
  (opcional) BACKUPS_DIR, BACKUP_RETENTION, UPLOADS_DIR

Funcionalidad:
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
import hashlib
import sqlite3
import re
from pathlib import Path
from datetime import datetime, date
from typing import List

import numpy as np
import streamlit as st
from dotenv import load_dotenv

# Opcional para generar .docx en la página "Generar"
try:
    from docx import Document  # pip install python-docx
except Exception:  # pragma: no cover
    Document = None

# IA (DeepSeek) – importar pipeline de generación desde generar_dictamen.py
try:
    from generar_dictamen import (
        get_topk_chunks,
        call_remote_model,
        associate_paragraphs_with_sources,
        create_docx_ptn,
        save_dictamen_record,
        MAX_CONTEXT_CHUNKS,
        MODEL_ID,
    )
    HAVE_GEN = True
except Exception as e:  # pragma: no cover
    HAVE_GEN = False
    GEN_IMPORT_ERR = str(e)

# PDF reader
try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit("Instalá pypdf o PyPDF2: pip install pypdf") from e

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:  # pragma: no cover
    raise SystemExit("Falta sentence-transformers: pip install sentence-transformers") from e

# FAISS
try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("Falta faiss-cpu: pip install faiss-cpu") from e

load_dotenv()

DB_PATH = os.getenv("DATABASE_PATH", "sistema28.db")
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
    "opinion_propia",
    "dictamen_generado",
    "informe_tecnico",
    "otro",
]

# Valoración por tipo (peso por defecto)
TIPO_PESO = {
    "sentencia": 1.3,
    "norma": 1.2,
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
    return out


@st.cache_resource
def cargar_modelo():
    return SentenceTransformer(EMB_MODEL)


def abrir_db():
    # Abrimos una conexión NUEVA por llamada (sin cachear) y apta para hilos de Streamlit
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA foreign_keys=ON;")
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
            created_at TEXT
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

    # documents: keywords, valor_tipo
    if not has_col("documents", "keywords"):
        con.execute("ALTER TABLE documents ADD COLUMN keywords TEXT;")
    if not has_col("documents", "valor_tipo"):
        con.execute("ALTER TABLE documents ADD COLUMN valor_tipo REAL;")

    # document_chunks: asegurar columnas requeridas
    for col, ddl in [
        ("document_id", "INTEGER"),
        ("chunk_text", "TEXT"),
        ("chunk_hash", "TEXT"),
        ("tipo", "TEXT"),
        ("posicion", "INTEGER"),
    ]:
        if not has_col("document_chunks", col):
            con.execute(f"ALTER TABLE document_chunks ADD COLUMN {col} {ddl};")

    # Índices
    con.execute("CREATE INDEX IF NOT EXISTS ix_documents_tipo ON documents(tipo);")
    con.execute("CREATE INDEX IF NOT EXISTS ix_chunks_docid ON document_chunks(document_id);")
    con.execute("CREATE INDEX IF NOT EXISTS ix_chunks_tipo ON document_chunks(tipo);")

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

        with st.spinner("📚 Buscando antecedentes relevantes en FAISS…"):
            hits = get_topk_chunks(consulta, k=k)
        if not hits:
            st.warning("No se encontraron antecedentes relevantes en el índice.")
            st.stop()

        # Construir contexto legible (igual que CLI)
        context_parts = []
        for i, hit in enumerate(hits, start=1):
            snippet = (hit.get("chunk_text") or "")[:1200]
            header = (
    f"[{i}] {hit.get('tipo_documento','')} | "
    f"{hit.get('organismo_emisor','')} | "
    f"{hit.get('fecha_documento','')}\n"
)
            context_parts.append(header + snippet)
        contexto_text = "\n\n".join(context_parts)

        system_prompt = (
            "Sos un abogado experto en derecho administrativo argentino. "
            "Utilizá únicamente el contexto que te brindo para redactar un proyecto de dictamen jurídico. "
            "No inventes normas ni citas: si una norma es mencionada, debe estar textual en el contexto provisto. "
            "Estructurá el dictamen en: I. Antecedentes; II. Análisis; III. Conclusión. "
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
        archivos = st.file_uploader("Elegí uno o varios PDF", type=["pdf"], accept_multiple_files=True, key=f"updf_{fid}")
        col1, col2, col3 = st.columns([2,1,1])
        with col1:
            titulo = st.text_input("Título (obligatorio)", value="", max_chars=250, key=f"titulo_{fid}")
        with col2:
            tipo = st.selectbox("Tipo (obligatorio)", options=TIPOS, index=None, placeholder="Seleccioná…", key=f"tipo_{fid}")
        with col3:
            fecha_doc = st.date_input("Fecha del documento", value=date.today(), key=f"fecha_{fid}")
        organismo = st.text_input("Organismo (opcional)", value="", key=f"org_{fid}")
        keywords_input = st.text_input("Palabras clave (coma separadas, opcional)", value="", key=f"kw_{fid}")
        # slider de valoración (peso) con default según tipo
        valor_def = TIPO_PESO.get(tipo, 1.0) if tipo else 1.0
        valor_tipo = st.slider("Valoración del tipo (peso)", 0.5, 2.0, float(valor_def), 0.1, key=f"valor_{fid}")
        submitted = st.form_submit_button("Procesar e indexar", use_container_width=True)

    if submitted:
        if not archivos:
            st.error("Subí al menos un PDF.")
            st.stop()
        if not titulo or tipo is None:
            st.error("Completá Título y Tipo.")
            st.stop()

        # Normalizar palabras clave a una cadena 'a, b, c'
        kw_items = [k.strip().lower() for k in (keywords_input or "").replace(";", ",").split(",") if k.strip()]
        kw_str = ",".join(sorted(set(kw_items)))

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
            st.warning("Recreando índice FAISS para ajustar la dimensión de embeddings.")
            index = faiss.IndexFlatIP(d_model)
            meta = []

        for f in archivos:
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
                        cur.execute("UPDATE documents SET path=? WHERE id=?", (str(saved_path), doc_id))
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

                cur.execute(
                    """
                    INSERT INTO documents (titulo, tipo, organismo, fecha, path, doc_hash, created_at, keywords, valor_tipo)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        titulo.strip(),
                        tipo,
                        organismo.strip(),
                        fecha_doc.isoformat(),
                        str(saved_path),
                        d_hash,
                        datetime.now().isoformat(),
                        kw_str,
                        float(valor_tipo),
                    ),
                )
                doc_id = cur.lastrowid
                nuevos_docs += 1

            # Extraer texto y limpiar líneas que igualan el título
            texto_raw = pdf_a_texto(contenido)
            texto = limpiar_titulo_del_texto(texto_raw, titulo)

            # Chunkear y vectorizar
            for i, ch in enumerate(chunkear(texto, max_palabras=50)):
                ch = ch.strip()
                if not ch:
                    continue
                c_hash = sha256_text(ch)
                try:
                    cur.execute(
                        """
                        INSERT INTO document_chunks (document_id, chunk_text, chunk_hash, tipo, posicion)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (doc_id, ch, c_hash, tipo, i),
                    )
                except sqlite3.IntegrityError:
                    continue  # chunk ya existente (por hash)

                # enriquecer embedding con keywords (si existen) de manera ligera
                text_for_emb = f"{ch} [KW: {kw_str}]" if kw_str else ch
                emb = model.encode([text_for_emb], normalize_embeddings=True)

                # Si la dim del índice no coincide, recrear y limpiar metadatos para evitar corrupción
                if hasattr(index, 'd') and index.d != emb.shape[1]:
                    st.warning("Dimensión de FAISS distinta a la del modelo. Se recrea el índice y metadatos.")
                    index = faiss.IndexFlatIP(emb.shape[1])
                    meta = []

                index.add(np.ascontiguousarray(emb.astype("float32")))
                meta.append({
                    "document_id": doc_id,
                    "titulo": titulo.strip(),
                    "tipo": tipo,
                    "posicion": i,
                    "hash": c_hash,
                    "valor_tipo": float(valor_tipo),
                    "keywords": kw_str,
                    # Campos necesarios para generar_dictamen.py
                    "chunk_text": ch,
                    "document_path": str(saved_path) if 'saved_path' in locals() else (str(p) if 'p' in locals() and p else ""),
                    "tipo_documento": tipo,
                    "organismo_emisor": organismo.strip(),
                    "fecha_documento": fecha_doc.isoformat(),
                })
                nuevos_chunks += 1

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

    # Métricas
    total = cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    por_tipo = cur.execute("SELECT tipo, COUNT(*) FROM documents GROUP BY tipo ORDER BY COUNT(*) DESC").fetchall()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Documentos totales", total)
    with c2:
        st.write("**Por tipo**:")
    
        if por_tipo:
            for t, c in por_tipo:
                st.write(f"- **{t}**: {c}")
        else:
            st.caption("Sin documentos aún.")

    st.subheader("Buscar por título")
    with st.expander("Filtros avanzados (opcional)"):
        cfd, cfh = st.columns(2)
        with cfd:
            f_desde = st.date_input("Desde", value=None)
        with cfh:
            f_hasta = st.date_input("Hasta", value=None)

# ...
    if f_desde:
        where.append("date(fecha) >= date(?)"); args.append(f_desde.isoformat())
    if f_hasta:
        where.append("date(fecha) <= date(?)"); args.append(f_hasta.isoformat())

    bt_col1, bt_col2 = st.columns([3,1])
    with bt_col1:
        q = st.text_input("Texto a buscar en títulos", value="", placeholder="Ej.: naturalez… o expte 123")
    with bt_col2:
        filtro_tipo = st.selectbox("Filtrar por tipo", options=["(Todos)"] + TIPOS, index=0)

    where = []
    args: List[object] = []
    if q.strip():
        where.append("LOWER(titulo) LIKE ?")
        args.append(f"%{q.strip().lower()}%")
    if filtro_tipo != "(Todos)":
        where.append("tipo = ?")
        args.append(filtro_tipo)
    sql = "SELECT id, titulo, tipo, fecha, organismo, path FROM documents"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT 200"  # safety

    rows = cur.execute(sql, tuple(args)).fetchall()
    con.close()

    if not rows:
        st.info("Sin resultados (ajustá el texto o el tipo).")
    else:
        st.write(f"Resultados: {len(rows)}")
        for (doc_id, titulo, tipo, fecha, organismo, path) in rows:
            with st.expander(f"#{doc_id} — {titulo}  ·  {tipo}  ·  {fecha}"):
                st.caption(f"Organismo: {organismo or '—'}")
                p = Path(path) if path else None
                if p and p.exists():
                    try:
                        data = p.read_bytes()
                        st.download_button(
                            label="⬇️ Descargar PDF",
                            data=data,
                            file_name=p.name,
                            mime="application/pdf",
                            key=f"dl_{doc_id}",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"No se pudo leer el archivo: {e}")
                else:
                    st.warning("Archivo no disponible en disco (registro antiguo o ruta inválida).")

    st.divider()
    if st.button("⬅ Volver al Inicio", use_container_width=True):
        st.session_state["nav"] = "Inicio"; st.rerun()
