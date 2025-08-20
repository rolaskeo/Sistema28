# s28/summarize.py
# Sumario extractivo (sin LLM) y "mapa de posturas" por reglas simples.
# Trabaja con los resultados devueltos por s28.rerank.search().

from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple

# === Separador de oraciones (sencillo, útil para fallos/dictámenes en español) ===
SENT_SPLIT = re.compile(r"(?<=[\.!?])\s+(?=[A-ZÁÉÍÓÚÑ])")

# Verbos/frases decisorias frecuentes (ajustables)
VERBOS_AFIRMA = {
    "hace lugar", "admite", "concede", "acoge", "declara procedente",
    "tiene por probado", "corresponde", "debe", "ordena"
}
VERBOS_NIEGA = {
    "rechaza", "deniega", "desestima", "improcedente", "inadmisible",
    "no ha lugar", "no corresponde"
}


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = SENT_SPLIT.split(text)
    out: List[str] = []
    for s in parts:
        s = s.strip()
        # Filtrar trozos demasiado cortos o claramente ruidosos
        if len(s) >= 30 and not s.lower().startswith(("fallos", "de justicia", "fiscal")):
            out.append(s)
    return out


def summarize_chunks(chunks: List[Dict[str, Any]], max_sentences: int = 8) -> List[str]:
    """
    Toma una lista de items (id, preview, chunk_type, page_num, ...)
    y devuelve hasta max_sentences oraciones relevantes priorizando RESUELVE/FALLO.
    """
    # 1) Ordenar por prioridad de sección
    pref = {"RESUELVE": 0, "FALLO": 0, "DICTAMINA": 1, "CONSIDERANDOS": 2}
    def sec_key(ct: str) -> int:
        return pref.get((ct or "").upper(), 3)

    ordered = sorted(chunks, key=lambda x: (sec_key(x.get("chunk_type")), -(len(x.get("preview", "")))))

    # 2) Armar pool de oraciones
    pool: List[str] = []
    for ch in ordered:
        pool.extend(split_sentences(ch.get("preview", "")))

    # 3) De-duplicación simple por firma (primeros 45 chars en minúsculas)
    seen = set()
    out: List[str] = []
    for s in pool:
        sig = s[:45].lower()
        if sig not in seen:
            seen.add(sig)
            out.append(s)
        if len(out) >= max_sentences:
            break
    return out


def detectar_postura(texto: str) -> str:
    t = (texto or "").lower()
    if any(v in t for v in VERBOS_AFIRMA):
        return "A FAVOR / ADMITE / HACE LUGAR"
    if any(v in t for v in VERBOS_NIEGA):
        return "EN CONTRA / RECHAZA / DENIEGA"
    return "NEUTRA / DESCRIPTIVA"


def mapa_posturas(chunks: List[Dict[str, Any]], max_por_bucket: int = 6) -> Dict[str, List[Tuple[int, str]]]:
    """
    Devuelve un dict con tres llaves y listas de (chunk_id, oración):
      - "A FAVOR / ADMITE / HACE LUGAR"
      - "EN CONTRA / RECHAZA / DENIEGA"
      - "NEUTRA / DESCRIPTIVA"
    """
    buckets: Dict[str, List[Tuple[int, str]]] = {
        "A FAVOR / ADMITE / HACE LUGAR": [],
        "EN CONTRA / RECHAZA / DENIEGA": [],
        "NEUTRA / DESCRIPTIVA": [],
    }
    for ch in chunks:
        cid = ch.get("id")
        for s in split_sentences(ch.get("preview", "")):
            p = detectar_postura(s)
            if len(buckets[p]) < max_por_bucket:
                buckets[p].append((cid, s))
    return buckets


def format_sumario(sumario: List[str]) -> str:
    return "\n".join(f"- {s}" for s in sumario)


def format_posturas(buckets: Dict[str, List[Tuple[int, str]]]) -> str:
    lines: List[str] = []
    for key, items in buckets.items():
        lines.append(f"[{key}]")
        if not items:
            lines.append("  (sin oraciones detectadas)")
            continue
        for cid, s in items:
            lines.append(f"  - (chunk {cid}) {s}")
        lines.append("")
    return "\n".join(lines)


# Uso de ejemplo si se ejecuta directo (requiere s28.rerank.search)
if __name__ == "__main__":
    try:
        # Intento 1: import relativo (cuando se ejecuta como módulo: python -m s28.summarize)
        from .rerank import search
        items = search("concesion vial ley 17.520", k=12)
    except Exception:
        try:
            # Intento 2: import absoluto (cuando se ejecuta como script: python s28/summarize.py)
            from s28.rerank import search
            items = search("concesion vial ley 17.520", k=12)
        except Exception as e:
            print("Error de prueba:", e)
            items = []

    if items:
        sumario = summarize_chunks(items, max_sentences=8)
        post = mapa_posturas(items)
        print("=== SUMARIO ===")
        print(format_sumario(sumario))
        print("\n=== MAPA DE POSTURAS ===")
        print(format_posturas(post))
