import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

# 📦 Cargar modelo multilingüe (mejor para texto jurídico en español)
print("🧠 Cargando modelo multilingüe...")
modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 📂 Conectar a la base de datos
DB_PATH = "sistema28.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 🧹 Borrar embeddings anteriores (opcional si querés regenerar todo)
print("🧽 Limpiando embeddings anteriores...")
cursor.execute("DELETE FROM chunk_embeddings")
conn.commit()

# 🧩 Buscar todos los chunks
cursor.execute("SELECT id, chunk_text FROM document_chunks")
chunks = cursor.fetchall()
print(f"🧩 Procesando {len(chunks)} chunks...")

# 🧠 Generar embeddings y guardar
for chunk_id, chunk_text in chunks:
    embedding = modelo.encode(chunk_text)
    vector_blob = embedding.astype(np.float32).tobytes()

    cursor.execute('''
        INSERT INTO chunk_embeddings (chunk_id, embedding)
        VALUES (?, ?)
    ''', (chunk_id, vector_blob))

conn.commit()
conn.close()

print("✅ Embeddings multilingües generados y guardados.")
