import re, unicodedata, sqlite3, sys, json

DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"

# --- utils ---
def normalize_text(s: str) -> str:
    if not s: return ""
    s = s.replace("_", " ")
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# detección simple de secciones/clases
SEC_PATTERNS = [
    (r"\b(vistos|resulta|considerando[s]?|considerandos)\b", "CONSIDERANDOS"),
    (r"\b(resuelve|se resuelve|se dispone|dispon[eé]|por ello|por lo expuesto)\b", "RESUELVE"),
    (r"\b(fallo|se falla|se decide|se declara|se hace lugar|se rechaza)\b", "FALLO"),
    (r"\b(anexo[s]?|anexos?)\b", "ANEXO"),
]

LAW_REGEX = re.compile(
    r"\b(ley(?:\s*n[°º.]?\s*\d+(?:\.\d+)?|\s+\d{4,}))\b"
    r"|(?:decreto(?:\s*n[°º.]?\s*\d+/\d{2,4}|\s+\d+/\d{2,4}))"
    r"|(?:resoluci[oó]n(?:\s*n[°º.]?\s*\d+/\d{2,4}|\s+\d+/\d{2,4}))"
    r"|(?:disposici[oó]n(?:\s*n[°º.]?\s*\d+/\d{2,4}|\s+\d+/\d{2,4}))"
    r"|(?:art[ií]culo(?:s)?\s+\d+[a-z]?)",
    flags=re.IGNORECASE
)

def detect_chunk_type(text: str):
    t = normalize_text(text)
    for pat, label in SEC_PATTERNS:
        if re.search(pat, t):
            return label
    return "OTROS"

def extract_law_refs(text: str):
    found = LAW_REGEX.findall(text)
    # LAW_REGEX tiene grupos; flatten y limpiar
    flat = []
    for item in found:
        if isinstance(item, tuple):
            item = [x for x in item if x]
            flat.extend(item)
        elif item:
            flat.append(item)
    # normalizar capitalización suave
    flat = [re.sub(r"\s+", " ", s).strip() for s in flat]
    # dedup conservando orden
    seen, out = set(), []
    for s in flat:
        k = normalize_text(s)
        if k not in seen:
            seen.add(k); out.append(s)
    return out

con = sqlite3.connect(DB)
cur = con.cursor()

# Traer columnas actuales por si alguna falta
cur.execute("PRAGMA table_info(document_chunks);")
cols = [c[1] for c in cur.fetchall()]
need_page_num = "page_num" in cols
has_page_legacy = "page" in cols

# Selección de filas a completar
cur.execute("""
SELECT id, chunk_text, page_num, chunk_type, text_norm, law_refs, page
FROM document_chunks
""")
rows = cur.fetchall()

updated = 0
for rid, chunk_text, page_num, chunk_type, text_norm, law_refs, legacy_page in rows:
    txt = chunk_text or ""
    norm = text_norm or normalize_text(txt)

    ctype = chunk_type or detect_chunk_type(txt)
    refs = json.dumps(extract_law_refs(txt), ensure_ascii=False) if not law_refs else law_refs

    # page_num: si está null, usar 'page' legacy si existe
    pnum = page_num
    if need_page_num and (pnum is None) and has_page_legacy:
        pnum = legacy_page

    cur.execute("""
        UPDATE document_chunks
        SET text_norm = ?, chunk_type = ?, law_refs = ?, page_num = COALESCE(?, page_num)
        WHERE id = ?
    """, (norm, ctype, refs, pnum, rid))
    updated += 1

con.commit()

# Re-indexar FTS para reflejar text_norm (si la tabla existe)
cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='fts_chunks';")
if cur.fetchone():
    # Limpieza y reconstrucción vinculada al content='document_chunks'
    # opción 1: forzar 'optimize'
    try:
        cur.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('optimize');")
    except Exception:
        pass
    # opción 2: reconstrucción total (más segura si FTS quedó desfasado)
    cur.execute("DELETE FROM fts_chunks;")
    # volver a poblar con el contenido actual
    cur.execute("SELECT id, text_norm FROM document_chunks")
    for rid, norm in cur.fetchall():
        cur.execute("INSERT INTO fts_chunks(rowid, text_norm, chunk_id) VALUES (?, ?, ?)",
                    (rid, norm or "", rid))
    con.commit()

con.close()
print(f"Backfill/enriquecimiento completado. Filas actualizadas: {updated} ✅")
