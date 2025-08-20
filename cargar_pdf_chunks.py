# cargar_pdf_chunks.py (parche v3.0)
# Ingesta con metadatos: text_norm, chunk_type, law_refs, page_num, offsets.
# - Segmenta a ~100 palabras (respetando puntos cuando es posible)
# - Detecta secciones y referencias normativas
# - Inserta en `documents` y `document_chunks`
# Dependencias: PyPDF2 (o pdfminer.six si preferís), pero aquí uso PyPDF2 por simplicidad.

from __future__ import annotations
import os, re, unicodedata, sqlite3, hashlib
from typing import List, Tuple
from PyPDF2 import PdfReader

DB = os.path.join(os.path.dirname(__file__), "sistema28.db")

# ==== Utilidades de normalización y metadatos ====
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
    flags=re.IGNORECASE,
)


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("_", " ")
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def detect_chunk_type(text: str) -> str:
    t = normalize_text(text)
    for pat, label in SEC_PATTERNS:
        if re.search(pat, t):
            return label
    return "OTROS"


def extract_law_refs(text: str) -> str:
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
            seen.add(k)
            out.append(s)
    # Guardamos como string JSON simple (lista join-eada) para mantener stdlib
    import json
    return json.dumps(out, ensure_ascii=False)


# ==== Segmentación a ~100 palabras (respetando puntos) ====

def split_into_chunks(text: str, target_words: int = 100) -> List[Tuple[str, int, int]]:
    """Devuelve lista de (chunk_text, off_start, off_end)."""
    words = text.split()
    chunks: List[Tuple[str, int, int]] = []
    if not words:
        return chunks

    i = 0
    acc_chars = 0
    while i < len(words):
        j = min(len(words), i + target_words)
        # expandir hasta el próximo punto si está cerca
        while j < len(words) and (j - i) < target_words + 30 and not words[j-1].endswith(('.', '.”', '.”', '”.')):
            j += 1
        piece = " ".join(words[i:j])
        start = acc_chars
        acc_chars += len(piece) + 1
        end = acc_chars
        chunks.append((piece, start, end))
        i = j
    return chunks


# ==== DB helpers ====

def ensure_doc(con, titulo: str, tipo: str, organismo: str, fecha: str, path: str) -> int:
    cur = con.cursor()
    # hash simple por path + size para idempotencia
    key = f"{path}|{os.path.getsize(path) if os.path.exists(path) else 0}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()

    # Si ya existe en documents, lo reusamos
    cur.execute("SELECT id FROM documents WHERE doc_hash=?", (h,))
    row = cur.fetchone()
    if row:
        return row[0]

    cur.execute(
        """
        INSERT INTO documents (titulo, tipo, organismo, fecha, path, doc_hash)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (titulo, tipo, organismo, fecha, path, h),
    )
    con.commit()
    return cur.lastrowid


def insert_chunk(con, doc_id: int, titulo: str, page_num: int, text: str, off_s: int, off_e: int):
    cur = con.cursor()
    text_norm = normalize_text(text)
    chunk_type = detect_chunk_type(text)
    law_refs = extract_law_refs(text)

    cur.execute(
        """
        INSERT INTO document_chunks (
            document_id, titulo, chunk_text, page_num, offset_start, offset_end,
            text_norm, chunk_type, law_refs
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (doc_id, titulo, text, page_num, off_s, off_e, text_norm, chunk_type, law_refs),
    )


# ==== Ingesta principal desde PDF ====

def ingest_pdf(pdf_path: str, titulo: str = None, tipo: str = "PDF", organismo: str = None, fecha: str = None, db_path: str = DB):
    con = sqlite3.connect(db_path)
    try:
        reader = PdfReader(pdf_path)
        titulo = titulo or os.path.basename(pdf_path)
        doc_id = ensure_doc(con, titulo=titulo, tipo=tipo or "PDF", organismo=organismo or "", fecha=fecha or "", path=pdf_path)

        for p_idx, page in enumerate(reader.pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception:
                raw = ""
            raw = re.sub(r"\s+", " ", (raw or "")).strip()
            if not raw:
                continue

            for chunk_text, off_s, off_e in split_into_chunks(raw, target_words=100):
                insert_chunk(con, doc_id, titulo, p_idx, chunk_text, off_s, off_e)
        con.commit()
        print(f"Ingesta OK: {pdf_path}")
    finally:
        con.close()


# ==== CLI simple ====
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python cargar_pdf_chunks.py ruta/al/archivo.pdf [TITULO] [TIPO] [ORGANISMO] [FECHA]")
        raise SystemExit(1)
    pdf = sys.argv[1]
    t = sys.argv[2] if len(sys.argv) > 2 else None
    tp = sys.argv[3] if len(sys.argv) > 3 else "PDF"
    org = sys.argv[4] if len(sys.argv) > 4 else None
    f = sys.argv[5] if len(sys.argv) > 5 else None
    ingest_pdf(pdf, titulo=t, tipo=tp, organismo=org, fecha=f)
