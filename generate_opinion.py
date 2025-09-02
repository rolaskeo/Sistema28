# generate_opinion.py
# Armado de dictamen/parecer preliminar a partir de S28 (v3.0)
# - Recupera chunks con s28.rerank
# - Resume con s28.summarize (extractivo, sin LLM)
# - Opcionalmente pule el borrador con LLM (DeepSeek API o LM Studio local)
# - Exporta a Markdown

from __future__ import annotations
import argparse, os, sqlite3, json, textwrap, datetime, sys, urllib.request, urllib.error
from typing import List, Dict, Any, Optional, Tuple

# --- Imports internos (funcionan si corres desde la raíz del proyecto) ---
try:
    from s28.rerank import search, print_results, DEFAULT_DB
    from s28.summarize import summarize_chunks, mapa_posturas
except Exception:
    # Modo alternativo: si se ejecuta desde otro cwd
    sys.path.append(os.path.dirname(__file__))
    from s28.rerank import search, print_results, DEFAULT_DB  # type: ignore
    from s28.summarize import summarize_chunks, mapa_posturas  # type: ignore

# ---------------------------------------------------------------------------
# Helpers de BD y metadatos
# ---------------------------------------------------------------------------

def open_db(db_path: Optional[str] = None):
    return sqlite3.connect(db_path or DEFAULT_DB)


def fetch_doc_meta(con: sqlite3.Connection, doc_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not doc_ids:
        return {}
    cur = con.cursor()
    q_marks = ",".join("?" for _ in doc_ids)
    cur.execute(f"SELECT id, titulo, COALESCE(issuer, organismo) as issuer, date_issued, tipo FROM documents WHERE id IN ({q_marks})", doc_ids)
    out: Dict[int, Dict[str, Any]] = {}
    for did, titulo, issuer, date_issued, tipo in cur.fetchall():
        out[did] = {"titulo": titulo or "", "issuer": issuer or "", "date_issued": date_issued or "", "tipo": tipo or ""}
    return out


def fetch_chunk_lawrefs_and_pages(con: sqlite3.Connection, chunk_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not chunk_ids:
        return {}
    cur = con.cursor()
    q_marks = ",".join("?" for _ in chunk_ids)
    cur.execute(f"SELECT id, page_num, law_refs FROM document_chunks WHERE id IN ({q_marks})", chunk_ids)
    out: Dict[int, Dict[str, Any]] = {}
    for cid, page_num, law_refs in cur.fetchall():
        refs: List[str] = []
        if isinstance(law_refs, str) and law_refs.strip():
            try:
                data = json.loads(law_refs)
                if isinstance(data, list):
                    refs = [str(x) for x in data]
            except Exception:
                pass
        out[cid] = {"page_num": page_num, "law_refs": refs}
    return out


def aggregate_normas(chunk_meta: Dict[int, Dict[str, Any]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for meta in chunk_meta.values():
        for r in meta.get("law_refs", []) or []:
            key = r.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(r)
    return out

# ---------------------------------------------------------------------------
# Render del borrador (sin LLM)
# ---------------------------------------------------------------------------

def render_markdown(
    query: str,
    items: List[Dict[str, Any]],
    sumario: List[str],
    posturas: Dict[str, List[Tuple[int, str]]],
    doc_meta: Dict[int, Dict[str, Any]],
    chunk_meta: Dict[int, Dict[str, Any]],
    max_fuentes: int = 12,
) -> str:
    fecha = datetime.date.today().isoformat()
    normas = aggregate_normas(chunk_meta)

    def cite(it: Dict[str, Any]) -> str:
        cid = it.get("id")
        did = it.get("document_id")
        p = chunk_meta.get(cid, {}).get("page_num")
        doc_tit = (doc_meta.get(did, {}) or {}).get("titulo", "")
        return f"(chunk {cid}, pág {p}, doc {did}: {doc_tit})"

    # Fuentes resumidas
    fuentes_lines: List[str] = []
    for it in items[:max_fuentes]:
        fuentes_lines.append(f"- {cite(it)}: {it.get('preview','')[:220].replace('\n',' ')}")

    # Posturas formateadas
    post_lines: List[str] = []
    for bucket, pares in posturas.items():
        post_lines.append(f"### {bucket}")
        if not pares:
            post_lines.append("(sin oraciones detectadas)")
        else:
            for cid, s in pares:
                p = chunk_meta.get(cid, {}).get("page_num")
                post_lines.append(f"- (chunk {cid}, pág {p}) {s}")
        post_lines.append("")

    # Marco normativo
    normas_lines = [f"- {n}" for n in normas] if normas else ["- (no detectado en los extractos)"]

    md = f"""
# Dictamen preliminar — {query}

**Fecha:** {fecha}

## 1) Sumario ejecutivo
{os.linesep.join(f'- {s}' for s in sumario) if sumario else '- (no se detectaron oraciones relevantes)'}

## 2) Marco normativo citado en los extractos
{os.linesep.join(normas_lines)}

## 3) Consideraciones y posturas detectadas
{os.linesep.join(post_lines)}

## 4) Fuentes (extractos utilizados)
{os.linesep.join(fuentes_lines)}

---
### Nota metodológica
Borrador armado en base a recuperación de párrafos relevantes mediante FTS5 + ranking propio (S28 v3.0),
resumen extractivo y reglas sobre verbos decisorios. Este documento **no** sustituye la lectura completa de las fuentes.
""".strip()
    return md

# ---------------------------------------------------------------------------
# LLM opcional (DeepSeek / LM Studio)
# ---------------------------------------------------------------------------

def call_deepseek(draft_md: str, api_key: str, model: str = "deepseek-chat", system_prompt: Optional[str] = None) -> str:
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    sysmsg = system_prompt or (
        "Sos un asistente jurídico. Reescribí el borrador en tono de dictamen formal, claro y conciso. "
        "Mantené las citas a (chunk #, pág) y no inventes fuentes."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sysmsg},
            {"role": "user", "content": draft_md},
        ],
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            j = json.loads(resp.read().decode("utf-8"))
            return j.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    except urllib.error.HTTPError as e:
        return f"[DeepSeek HTTP {e.code}] {e.read().decode('utf-8', 'ignore')}"
    except Exception as e:
        return f"[DeepSeek error] {e}"


def call_lmstudio(draft_md: str, base_url: str = "http://localhost:1234/v1/chat/completions", model: str = "local-model") -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Sos un asistente jurídico. Mejora redacción y claridad, sin inventar."},
            {"role": "user", "content": draft_md},
        ],
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base_url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            j = json.loads(resp.read().decode("utf-8"))
            return j.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    except Exception as e:
        return f"[LM Studio error] {e}"

# ---------------------------------------------------------------------------
# CLI principal
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="S28 v3.0 — Generar dictamen preliminar")
    ap.add_argument("query", help="Tema/consulta (ej: 'ley 17.520 concesion vial')")
    ap.add_argument("--k", type=int, default=12, help="Cantidad de chunks a considerar (default: 12)")
    ap.add_argument("--issuer", type=str, default=None, help="Filtrar por organismo/issuer (LIKE)")
    ap.add_argument("--chunk-type", type=str, default=None, help="Filtrar por tipo de sección (RESUELVE, FALLO, etc.)")
    ap.add_argument("--doc-id", type=int, default=None, help="Filtrar por document_id específico")
    ap.add_argument("--db", type=str, default=None, help="Path alternativo a la DB (si no es el default)")
    ap.add_argument("--out", type=str, default=None, help="Path de salida .md (si se omite, se genera uno en ./out/")
    ap.add_argument("--append-sources", action="store_true", help="Anexar los textos completos de los chunks al final")
    ap.add_argument("--llm", type=str, choices=["none", "deepseek", "lmstudio"], default="none", help="Pulir con LLM opcional")
    ap.add_argument("--deepseek-model", type=str, default="deepseek-chat", help="Modelo DeepSeek (si --llm=deepseek)")
    ap.add_argument("--lmstudio-url", type=str, default="http://localhost:1234/v1/chat/completions", help="Endpoint LM Studio")
    ap.add_argument("--lmstudio-model", type=str, default="local-model", help="Nombre del modelo en LM Studio")
    args = ap.parse_args()

    # Filtros para la búsqueda
    filters: Dict[str, Any] = {}
    if args.chunk_type:
        filters["chunk_type"] = args.chunk_type
    if args.doc_id is not None:
        filters["document_id"] = args.doc_id
    if args.issuer:
        filters["issuer"] = args.issuer

    items = search(args.query, k=args.k, db_path=args.db, filters=filters)
    if not items:
        print("(sin resultados relevantes)" )
        return

    sumario = summarize_chunks(items, max_sentences=min(10, max(6, args.k // 2)))
    posts = mapa_posturas(items)

    with open_db(args.db) as con:
        doc_ids = sorted({it.get("document_id") for it in items if it.get("document_id") is not None})
        cid_list = [int(it.get("id")) for it in items if it.get("id") is not None]
        dmeta = fetch_doc_meta(con, doc_ids)
        cmeta = fetch_chunk_lawrefs_and_pages(con, cid_list)

        draft = render_markdown(args.query, items, sumario, posts, dmeta, cmeta)

        # LLM opcional
        polished = None
        if args.llm == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                print("[AVISO] Falta variable de entorno DEEPSEEK_API_KEY. Se deja el borrador sin pulir.")
            else:
                polished = call_deepseek(draft, api_key=api_key, model=args.deepseek_model)
        elif args.llm == "lmstudio":
            polished = call_lmstudio(draft, base_url=args.lmstudio_url, model=args.lmstudio_model)

        final_text = polished if polished and not polished.startswith("[") else draft

        # Si piden anexar textos completos de los chunks
        if args.append_sources:
            lines = [final_text, "\n---\n## Anexo — extractos completos\n"]
            cur = con.cursor()
            for it in items:
                cid = int(it.get("id"))
                cur.execute("SELECT chunk_text, page_num FROM document_chunks WHERE id=?", (cid,))
                row = cur.fetchone()
                if row:
                    full_txt, p = row
                    lines.append(f"\n### Chunk {cid} (pág {p})\n\n{full_txt}\n")
            final_text = "".join(lines)

    # Salida
    out_dir = os.path.join(os.getcwd(), "out")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join(out_dir, f"dictamen_{ts}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"✅ Dictamen generado → {out_path}")


if __name__ == "__main__":
    main()
