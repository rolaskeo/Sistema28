# scripts/check_old_mapping.py
import sqlite3

DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"
conn = sqlite3.connect(DB); conn.row_factory = sqlite3.Row

def count(sql):
    return conn.execute(sql).fetchone()[0]

candidates = [
    ("document_chunks", "id"),
    ("document_chunks_backup", "id"),
    ("document_chunks", "chunk_id"),  # por si acaso hubiera columna distinta
]

print("=== Tablas candidatas para chunks (viejo) ===")
for table, pk in candidates:
    try:
        # ¿existen columnas?
        cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]
        if pk not in cols:
            print(f"- {table}.{pk}: NO (columna {pk} no existe) -> columnas={cols}")
            continue
        j = count(f"SELECT COUNT(*) FROM embeddings e JOIN {table} c ON c.{pk}=e.chunk_id")
        print(f"- {table}.{pk}: JOIN hits = {j}")
    except Exception as e:
        print(f"- {table}.{pk}: ERROR -> {e}")

print("\n=== Muestra de 5 embeddings (chunk_id) y presencia en cada candidata ===")
rows = conn.execute("SELECT chunk_id FROM embeddings ORDER BY chunk_id LIMIT 5").fetchall()
ids = [int(r["chunk_id"]) for r in rows]
print("chunk_ids ejemplo:", ids)

for table, pk in candidates:
    try:
        cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]
        if pk not in cols: 
            continue
        found = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {pk} IN ({','.join(['?']*len(ids))})",
            ids
        ).fetchone()[0]
        print(f"- {table}.{pk}: contiene {found}/{len(ids)} ids de ejemplo")
    except Exception as e:
        print(f"- {table}.{pk}: ERROR -> {e}")
