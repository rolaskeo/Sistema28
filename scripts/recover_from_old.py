import sqlite3, os

NEW_DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"
OLD_DB = r"C:\Rolex\Python\Sistema28Script\sistema28_old_corrupt.db"

if not os.path.exists(OLD_DB):
    raise SystemExit(f"No existe el archivo de respaldo: {OLD_DB}")

dst = sqlite3.connect(NEW_DB)
src = sqlite3.connect(f"file:{OLD_DB}?mode=ro", uri=True)

dcur = dst.cursor()
scur = src.cursor()

# Tablas posibles con datos (nombres antiguos y nuevos)
CANDIDATE_TABLES = [
    "documents", "documentos",
    "document_chunks", "document_chunks_backup",
    "embeddings", "chunk_embeddings",
    "dictamenes", "antecedentes_relevantes", "footnotes",
    "relevance_feedback", "rerank_cache", "documentos_subidos",
]

def table_exists(cur, name):
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;", (name,))
    return cur.fetchone() is not None

def rowcount(cur, name):
    cur.execute(f"SELECT COUNT(*) FROM {name}")
    return cur.fetchone()[0]

def colnames(cur, name):
    cur.execute(f"PRAGMA table_info({name});")
    return [r[1] for r in cur.fetchall()]

def copy_table(src_name, dst_name):
    # Si no existe en destino, creamos una tabla compatible mínima copiando el esquema del origen
    if not table_exists(dcur, dst_name):
        scur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (src_name,))
        row = scur.fetchone()
        if row and row[0]:
            dcur.execute(row[0].replace(src_name, dst_name))
            dst.commit()

    # Intersección de columnas (sin id si choca)
    src_cols = colnames(scur, src_name)
    dst_cols = colnames(dcur, dst_name)
    common = [c for c in src_cols if c in dst_cols]

    if not common:
        print(f"- {src_name} → {dst_name}: sin columnas comunes, salto")
        return

    # Evitar duplicar filas si ya hay datos en destino
    if table_exists(dcur, dst_name) and rowcount(dcur, dst_name) > 0:
        print(f"- {dst_name}: ya tiene datos, no copio")
        return

    cols_list = ",".join(f'"{c}"' for c in common)
    placeholders = ",".join(["?"] * len(common))

    # Copia resiliente fila a fila
    scur.execute(f"SELECT {cols_list} FROM {src_name}")
    rows = scur.fetchall()
    ok, fail = 0, 0
    for r in rows:
        try:
            dcur.execute(f'INSERT INTO {dst_name} ({cols_list}) VALUES ({placeholders})', r)
            ok += 1
        except Exception:
            fail += 1
    dst.commit()
    print(f"+ {src_name} → {dst_name}: copiadas {ok}, fallidas {fail}")

# Recorrer candidatos: si existe en origen y tiene filas, lo copiamos
for name in CANDIDATE_TABLES:
    if table_exists(scur, name):
        try:
            n = rowcount(scur, name)
        except Exception:
            n = 0
        if n > 0:
            # si el nombre existe también en destino, usamos el mismo; si no, creamos
            target = name if table_exists(dcur, name) else name
            copy_table(name, target)

# Cerrar
src.close()
dst.close()
print("Recuperación finalizada ✅")
