# scripts/make_queries_eval_onc.py
import sqlite3, json

DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"
OUT = r".\queries_eval_onc.jsonl"

QUERIES = [
  ("redeterminación de precios en obra pública",              "2019_2020_ONC_Dictamenes"),
  ("impugnación de pliegos en contrataciones públicas",       "2012_2018_ONC_Dictamenes"),
  ("garantía de mantenimiento de oferta y ejecución",         "2012_2018_ONC_Dictamenes"),
  ("ampliación de plazo y penalidades por mora",              "2012_2018_ONC_Dictamenes"),
  ("adicionales de obra y modificaciones del contrato",       "2012_2018_ONC_Dictamenes"),
  ("resolución de contratos por incumplimiento del contratista","2012_2018_ONC_Dictamenes"),
  ("criterios sobre criterios de elegibilidad y solvencia",   "2019_2020_ONC_Dictamenes"),
  ("apertura y evaluación de ofertas",                        "2012_2018_ONC_Dictamenes"),
  ("órdenes de cambio y precios nuevos",                      "2023_ONC_Dictamenes"),
  ("multas y sanciones en el régimen de contrataciones",      "2012_2018_ONC_Dictamenes"),
]

conn = sqlite3.connect(DB); conn.row_factory = sqlite3.Row

def doc_v3_by_title_like(title_like: str):
    r = conn.execute("SELECT id, title, path FROM documents_v3 WHERE title LIKE ? ORDER BY id LIMIT 1", (f"%{title_like}%",)).fetchone()
    return r

def chunk_ids_for_doc(doc_id: int):
    rows = conn.execute("SELECT chunk_id FROM chunks_v3 WHERE doc_id=? ORDER BY chunk_index", (doc_id,)).fetchall()
    return [int(r["chunk_id"]) for r in rows]

out = []
for q, title_like in QUERIES:
    d = doc_v3_by_title_like(title_like)
    if not d:
        print(f"[WARN] no hallé doc_v3 con título ~ {title_like}")
        continue
    chunks = chunk_ids_for_doc(int(d["id"]))
    if not chunks:
        print(f"[WARN] doc {d['id']} sin chunks")
        continue
    out.append({
        "q": q,
        "relevant_chunk_ids_v3": chunks
    })

with open(OUT, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"OK -> {OUT} ({len(out)} queries)")
