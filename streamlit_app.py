# streamlit_app.py
"""
Interfaz Streamlit para Sistema 28
- Usa la lógica y funciones de generar_dictamen.py (debe estar en la misma carpeta)
- Opciones: 1) Generar dictamen (Word + guardar en DB) 2) Ver/valorar antecedentes 3) Subir documento manual
- Muestra resultados, permite valorar antecedentes y descargar el .docx generado.
"""

import streamlit as st
import os
import json
from datetime import datetime
import sqlite3
from io import BytesIO

# Importar funciones/objetos ya probados en tu script principal
# (el archivo generar_dictamen.py debe estar en la misma carpeta)
from generar_dictamen import (
    get_topk_chunks,
    call_remote_model,
    associate_paragraphs_with_sources,
    create_docx_ptn,
    save_dictamen_record,
    init_db,
    DB_PATH,
    OUTPUT_DIR,
    MAX_CONTEXT_CHUNKS,
)

# Inicializar DB (crea tablas si faltan)
init_db()

st.set_page_config(page_title="Sistema 28 — Interfaz", layout="wide")

st.title("Sistema 28 — Interfaz gráfica")
st.markdown("Generador de dictámenes | Recuperación de antecedentes | Registro de documentos")

# Sidebar: API key (se puede pegar aquí si no está en .env)
st.sidebar.header("Configuración")
api_key_input = st.sidebar.text_input("API key (OpenRouter / DeepSeek)", type="password")
if api_key_input:
    # si el usuario pone la clave acá, exportamos la variable de entorno usada por tu script
    os.environ["API_KEY"] = api_key_input

# Control de cantidad de chunks a recuperar
max_chunks = st.sidebar.slider("Máx. de antecedentes (chunks) a usar en prompt", 1, 20, MAX_CONTEXT_CHUNKS)

# Main menu
mode = st.radio("Seleccioná la operación", ("Generar dictamen", "Ver / Valorar antecedentes", "Subir documento"))

# --- Utilities para la app ---
def save_rating_ui(antecedent_id: int, rating: int, comment: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE antecedentes_relevantes SET valoracion = ?, comentario = ? WHERE id = ?", (rating, comment, antecedent_id))
    conn.commit()
    conn.close()

def insert_antecedente_standalone(hit, rating: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO antecedentes_relevantes (dictamen_id, chunk_idx, document_path, tipo_documento, organismo_emisor, fecha_documento, snippet, similitud, valoracion)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, hit.get("idx"), hit.get("document_path"), hit.get("tipo_documento"), hit.get("organismo_emisor"), hit.get("fecha_documento"), hit.get("chunk_text")[:800], hit.get("similitud"), rating))
    conn.commit()
    conn.close()

def db_query(sql, params=()):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return rows

def make_downloadable_docx(path: str):
    with open(path, "rb") as f:
        data = f.read()
    return data

# --- MODE: Generar dictamen ---
if mode == "Generar dictamen":
    st.header("Generar dictamen jurídico")
    consulta = st.text_area("Ingrese la consulta jurídica (breve):", height=80)
    keywords_text = st.text_input("Palabras clave (opcional, separadas por coma) — sirve para filtrar antecedentes")
    use_remote_model = st.checkbox("Usar motor remoto (OpenRouter / DeepSeek). Si no, se intentará el modelo local si está configurado.", value=True)

    cols = st.columns([1,1,1])
    with cols[0]:
        btn_generate = st.button("Generar dictamen")
    with cols[1]:
        st.write("")
        btn_preview = st.button("Previsualizar antecedentes")
    with cols[2]:
        st.write("")
        st.info("Asegurate que la API_KEY esté configurada (sidebar) si usás motor remoto.")

    if btn_preview:
        if not consulta.strip():
            st.warning("Ingresá una consulta.")
        else:
            st.info(f"Recuperando hasta {max_chunks} bloques relevantes...")
            hits = get_topk_chunks(consulta, k=max_chunks)
            if not hits:
                st.warning("No se encontraron antecedentes.")
            else:
                for i,h in enumerate(hits, start=1):
                    st.markdown(f"**[{i}]** {h.get('tipo_documento','')} | {h.get('organismo_emisor','')} | {h.get('fecha_documento','')}")
                    st.write(h.get('chunk_text')[:800] + ("..." if len(h.get('chunk_text',''))>800 else ""))

    if btn_generate:
        if not consulta.strip():
            st.warning("Ingresá una consulta para generar dictamen.")
        else:
            st.info("Buscando antecedentes relevantes...")
            hits = get_topk_chunks(consulta, k=max_chunks)
            if not hits:
                st.warning("No se encontraron antecedentes relevantes.")
            else:
                # Build context parts (trimmed)
                context_parts = []
                for i, hit in enumerate(hits, start=1):
                    snippet = hit.get("chunk_text", "")[:1200]
                    header = f"[{i}] {hit.get('tipo_documento','')} | {hit.get('organismo_emisor','')} | {hit.get('fecha_documento','')}\n"
                    context_parts.append(header + snippet)
                contexto_text = "\n\n".join(context_parts)

                system_prompt = (
                    "Sos un abogado experto en derecho administrativo argentino. "
                    "Utilizá únicamente el contexto que te brindo para redactar un proyecto de dictamen jurídico. "
                    "No inventes normas ni citas: si una norma es mencionada, debe estar textual en el contexto provisto. "
                    "Estructurá el dictamen en: I. Antecedentes; II. Análisis; III. Conclusión. "
                    "Sé conciso, preciso y aplicá el Manual de Estilo PTN 2023 en el lenguaje."
                )
                user_prompt = f"Consulta: {consulta}\n\nContexto relevante:\n{contexto_text}\n\nRedactá el dictamen técnico-jurídico."

                st.info("Solicitando redacción al modelo...")
                with st.spinner("Llamando al modelo (puede demorar unos segundos)..."):
                    try:
                        response_text = call_remote_model(system_prompt, user_prompt, max_tokens=1024, temperature=0.2, attempts=3, timeout=60)
                    except Exception as e:
                        st.error(f"Error llamando al modelo: {e}")
                        response_text = None

                if not response_text:
                    st.warning("No se obtuvo respuesta del modelo.")
                else:
                    st.subheader("Previsualización del dictamen (parte)")
                    st.write(response_text[:4000])

                    # Asociar párrafos con fuentes
                    st.info("Asociando párrafos con fuentes (semántico)...")
                    with st.spinner("Asociando..."):
                        associated = associate_paragraphs_with_sources(response_text, top_k_per_par=3)

                    st.info("Generando archivo Word con formato PTN...")
                    with st.spinner("Creando .docx..."):
                        filename = create_docx_ptn(consulta, associated)
                    st.success(f"Documento creado: {filename}")

                    # Guardar en BD
                    dictamen_id = save_dictamen_record(consulta, filename, os.getenv("MODEL_ID","deepseek/deepseek-chat"), hits)
                    st.success(f"Dictamen guardado con id {dictamen_id}")

                    # Ofrecer descarga
                    data = make_downloadable_docx(filename)
                    basename = os.path.basename(filename)
                    st.download_button(label="Descargar dictamen (.docx)", data=data, file_name=basename, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

                    # Mostrar antecedentes usados y permitir valoración rápida
                    st.markdown("### Antecedentes usados (valoralos si querés)")
                    for i, hit in enumerate(hits, start=1):
                        st.write(f"**[{i}]** {hit.get('tipo_documento')} | {hit.get('organismo_emisor')} | {hit.get('fecha_documento')}")
                        st.write(hit.get('chunk_text')[:600] + ("..." if len(hit.get('chunk_text',''))>600 else ""))
                        col1, col2 = st.columns([1,3])
                        with col1:
                            rating = st.selectbox(f"Valoración (antecedente {i})", options=["-","1","2","3","4","5"], key=f"rating_{dictamen_id}_{i}")
                        with col2:
                            comment = st.text_input(f"Comentario (opcional) antecedente {i}", key=f"comment_{dictamen_id}_{i}")
                        if st.button(f"Guardar valoración antecedente {i}", key=f"save_{dictamen_id}_{i}"):
                            if rating != "-" :
                                # Insert como antecedente asociado (actualizamos antecedente ya insertado en save_dictamen_record)
                                # Buscamos el último antecedente_relevante insertado para este dictamen y chunk_idx
                                conn = sqlite3.connect(DB_PATH)
                                cur = conn.cursor()
                                # buscamos por dictamen_id y chunk_idx
                                cur.execute("SELECT id FROM antecedentes_relevantes WHERE dictamen_id = ? AND chunk_idx = ? ORDER BY id DESC LIMIT 1", (dictamen_id, hit.get("idx")))
                                res = cur.fetchone()
                                if res:
                                    aid = res[0]
                                    cur.execute("UPDATE antecedentes_relevantes SET valoracion = ?, comentario = ? WHERE id = ?", (int(rating), comment, aid))
                                    conn.commit()
                                    st.success("Valoración guardada.")
                                else:
                                    # si no existe, insertamos un registro stand-alone con dictamen_id
                                    cur.execute("""INSERT INTO antecedentes_relevantes (dictamen_id, chunk_idx, document_path, tipo_documento, organismo_emisor, fecha_documento, snippet, similitud, valoracion, comentario) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                                                (dictamen_id, hit.get("idx"), hit.get("document_path"), hit.get("tipo_documento"), hit.get("organismo_emisor"), hit.get("fecha_documento"), hit.get("chunk_text")[:800], hit.get("similitud"), int(rating), comment))
                                    conn.commit()
                                    st.success("Valoración guardada (nuevo registro).")
                                conn.close()
                            else:
                                st.warning("Elegí una valoración (1-5) antes de guardar.")

# --- MODE: Ver / Valorar antecedentes ---
elif mode == "Ver / Valorar antecedentes":
    st.header("Buscar y valorar antecedentes")
    consulta = st.text_input("Búsqueda libre (consulta):")
    if st.button("Buscar"):
        if not consulta.strip():
            st.warning("Ingresá una consulta.")
        else:
            st.info(f"Buscando hasta {max_chunks} antecedentes...")
            hits = get_topk_chunks(consulta, k=max_chunks)
            if not hits:
                st.warning("No se encontraron antecedentes.")
            else:
                st.write(f"Se encontraron {len(hits)} antecedentes (primeros {len(hits)} mostrados).")
                for i, h in enumerate(hits, start=1):
                    st.markdown(f"**[{i}] {h.get('tipo_documento','')} | {h.get('organismo_emisor','')} | {h.get('fecha_documento','')}**")
                    st.write(h.get('chunk_text')[:1000] + ("..." if len(h.get('chunk_text',''))>1000 else ""))
                    cols = st.columns([1,2,2])
                    with cols[0]:
                        rating = st.selectbox(f"Valoración {i}", ["-","1","2","3","4","5"], key=f"val_{i}")
                    with cols[1]:
                        comment = st.text_input(f"Comentario {i}", key=f"com_{i}")
                    with cols[2]:
                        if st.button(f"Guardar {i}", key=f"save_hit_{i}"):
                            if rating != "-":
                                insert_antecedente_standalone(h, int(rating))
                                st.success("Valoración guardada.")
                            else:
                                st.warning("Elegí una valoración antes de guardar.")

    # También permitimos ver valoraciones ya guardadas
    st.markdown("---")
    st.subheader("Valoraciones previas (últimas 20)")
    rows = db_query("SELECT id, dictamen_id, chunk_idx, snippet, similitud, valoracion, comentario FROM antecedentes_relevantes ORDER BY id DESC LIMIT 20")
    for r in rows:
        aid, did, chunk_idx, snippet, sim, val, com = r
        st.write(f"ID: {aid} | dictamen_id: {did} | chunk_idx: {chunk_idx} | similitud: {sim:.4f} | valor: {val}")
        st.write(snippet[:300] + ("..." if len(snippet)>300 else ""))
        if com:
            st.write(f"Comentario: {com}")

# --- MODE: Subir documento ---
elif mode == "Subir documento":
    st.header("Subir documento (registro manual)")
    st.info("Campos obligatorios: título, tipo de documento, 3-5 palabras clave, relevancia (1-5), contenido (texto).")
    with st.form("upload_form"):
        titulo = st.text_input("Título")
        tipo = st.selectbox("Tipo de documento", ["sentencia", "dictamen", "doctrina", "opinión propia", "informe técnico", "normativo"])
        kws = st.text_input("Palabras clave (separadas por coma) — mínimo 3, máximo 5")
        relevancia = st.slider("Relevancia (1-5)", 1, 5, 3)
        contenido = st.text_area("Contenido / texto (pegar aquí)", height=200)
        submitted = st.form_submit_button("Registrar documento")
    if submitted:
        kws_list = [k.strip() for k in kws.split(",") if k.strip()]
        if not (titulo and tipo and contenido and 3 <= len(kws_list) <= 5):
            st.warning("Faltan datos o cantidad de palabras clave incorrecta (deben ser 3-5).")
        else:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO documentos_subidos (titulo, tipo_documento, palabras_clave, relevancia, contenido, fecha_subida)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (titulo, tipo, ", ".join(kws_list), relevancia, contenido, datetime.now().isoformat()))
            conn.commit()
            conn.close()
            st.success("Documento registrado. Para indexarlo en FAISS ejecutá el proceso de chunking/reindexado (script aparte).")

st.markdown("---")
st.caption("Sistema 28 de Rolando Keumurdji Rizzuti — Interfaz Streamlit. Basado en la lógica del CLI (generar_dictamen.py).")
