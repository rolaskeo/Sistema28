import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Conectar a la base de datos
DB_PATH = "sistema28.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Crear tabla para embeddings si no existe
cursor.execute('''
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id INTEGER PRIMARY KEY,
    embedding BLOB,
    FOREIGN KEY(chunk_id) REFERENCES document_chunks(id)
)
''')
conn.commit()

# Cargar modelo de embeddings
print("🧠 Cargando modelo de lenguaje...")
modelo = SentenceTransformer('all-MiniLM-L6-v2')

# Obtener los chunks que aún no tienen embeddings
cursor.execute('''
SELECT id, chunk_text FROM document_chunks
WHERE id NOT IN (SELECT chunk_id FROM chunk_embeddings)
''')
chunks = cursor.fetchall()

print(f"🧩 Procesando {len(chunks)} chunks...")

# Procesar y guardar embeddings
for chunk_id, chunk_text in chunks:
    embedding_vector = modelo.encode(chunk_text)
    # Convertir a binario para guardar en SQLite
    embedding_blob = np.array(embedding_vector).tobytes()
    cursor.execute('''
        INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)
    ''', (chunk_id, embedding_blob))

conn.commit()
conn.close()
print("✅ Embeddings generados y guardados.")
