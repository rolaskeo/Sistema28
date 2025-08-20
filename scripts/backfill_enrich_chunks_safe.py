import re, unicodedata, sqlite3, json

DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"

def normalize_text(s: str) -> str:
    if not s: return ""
    s = s.replace("_", " ")
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"\s+", " ", s).strip()
    return s

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
    flat = []
    for item in found:
        if isinstance(item, tuple):
            item = [x for x in item if x]
            flat.extend(item)
        elif item:
            flat.append(item)
    flat = [re.sub(r"\s+", " ", s).strip() for s in flat]
    seen, out = set(), []
    for s in flat:
        k = normalize_text(s)
        if k not in seen:
            seen.add(k); out.append(s)
    return out

con = sqlite3.connect(DB)
cur = con.cursor()

# Traer filas a actualizar
cur.execute("""
SELECT id, chunk_text, page_num, chunk_type, text_norm, law_refs, page
FROM document_chunks
""")
rows = cur.fetchall()

updated = 0
for rid, chunk_text, page_num, chunk_type, text_norm, law_refs, legacy_page in rows:
    txt = chunk_text or ""
    norm = normalize_text(txt) if not text_norm else text_norm
    ctype = chunk_type or detect_chunk_type(txt)
    refs  = law_refs or json.dumps(extract_law_refs(txt), ensure_ascii=False)
    pnum  = page_num if page_num is not None else legacy_page

    # UPDATE: esto dispara el trigger AU y refresca FTS automáticamente
    cur.execute("""
        UPDATE document_chunks
        SET text_norm = ?, chunk_type = ?, law_refs = ?, page_num = COALESCE(?, page_num)
        WHERE id = ?
    """, (norm, ctype, refs, pnum, rid))
    updated += 1

con.commit()
con.close()
print(f"Backfill/enriquecimiento completado. Filas actualizadas: {updated} ✅")
