# crear_indice_faiss.py
import sqlite3
import faiss
import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Configuración
DATABASE_PATH = "sistema28.db"
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print("📦 Cargando modelo de embeddings...")
model = SentenceTransformer(EMBEDDING_MODEL)

# Conectar a la base y obtener todos los chunks
print("📂 Leyendo chunks desde la base de datos...")
conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()
cursor.execute("""
    SELECT id, chunk_text, document_path, tipo_documento, organismo_emisor, fecha_documento
    FROM document_chunks
""")
rows = cursor.fetchall()
conn.close()

if not rows:
    print("⚠️ No se encontraron registros en la tabla document_chunks.")
    exit()

# Generar embeddings
print(f"🧠 Generando embeddings para {len(rows)} chunks...")
texts = [r[1] for r in rows]
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# Crear índice FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Guardar índice y metadata
faiss.write_index(index, INDEX_PATH)
metadata = [
    {
        "id": r[0],
        "chunk_text": r[1],
        "document_path": r[2],
        "tipo_documento": r[3],
        "organismo_emisor": r[4],
        "fecha_documento": r[5]
    }
    for r in rows
]
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"✅ Índice FAISS guardado en {INDEX_PATH}")
print(f"✅ Metadata guardada en {METADATA_PATH}")
