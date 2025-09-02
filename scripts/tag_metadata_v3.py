# scripts/tag_metadata_v3.py
import re, sqlite3, os

DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"
conn = sqlite3.connect(DB); conn.row_factory = sqlite3.Row
c = conn.cursor()

# Asegurar columnas
for col, typ in [("tipo_documento_v3","TEXT"), ("organismo_v3","TEXT"), ("year_v3","INTEGER")]:
    try:
        c.execute(f"ALTER TABLE documents_v3 ADD COLUMN {col} {typ}")
    except sqlite3.OperationalError:
        pass

def detect_year(p):
    m = re.search(r"(19|20)\d{2}", p)
    return int(m.group(0)) if m else None

def detect_from_path(path):
    p = (path or "").lower()
    tipo, org = None, None
    if "onc" in p and "dict" in p:  # dictámenes ONC
        tipo = "dictamen"
        org  = "ONC"
    # podés ampliar aquí otras reglas si querés
    return tipo, org, detect_year(p)

docs = c.execute("SELECT id, path FROM documents_v3").fetchall()
n=0
for d in docs:
    tipo, org, yr = detect_from_path(d["path"] or "")
    if any([tipo, org, yr]):
        c.execute("""
          UPDATE documents_v3
          SET tipo_documento_v3 = COALESCE(?, tipo_documento_v3),
              organismo_v3      = COALESCE(?, organismo_v3),
              year_v3           = COALESCE(?, year_v3),
              issuer            = COALESCE(?, issuer)
          WHERE id = ?
        """, (tipo, org, yr, org, d["id"]))
        n+=1

conn.commit()
conn.close()
print(f"OK: metadatos ONC actualizados en {n} documentos.")
