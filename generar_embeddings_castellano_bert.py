import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer, models

# 🧠 Cargar modelo BERT entrenado en castellano
print("🧠 Cargando modelo castellano...")
bert_model = models.Transformer('dccuchile/bert-base-spanish-wwm-uncased')
pooling = models.Pooling(bert_model.get_word_embedding_dimension())
modelo = SentenceTransformer(modules=[bert_model, pooling])

# 📂 Conectar a la base de datos
DB_PATH = "sistema28.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 🧽 Borrar embeddings anteriores (opcional)
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

print("✅ Embeddings en castellano generados y guardados.")
