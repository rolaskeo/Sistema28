import sqlite3
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import os

# === CONFIGURACIÓN GENERAL ===
DB_PATH = "sistema28.db"
TABLE_NAME = "chunk_embeddings"
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"

print("🧠 Cargando modelo castellano...")
model = SentenceTransformer(MODEL_NAME)

# === CONECTAR A LA BASE DE DATOS ===
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# === LIMPIAR EMBEDDINGS ANTERIORES ===
print("🧽 Limpiando embeddings anteriores...")
cursor.execute(f"DELETE FROM {TABLE_NAME}")
conn.commit()

# === OBTENER CHUNKS PARA PROCESAR ===
cursor.execute("SELECT id, texto FROM chunks")
rows = cursor.fetchall()

print(f"🧩 Procesando {len(rows)} chunks...")

# === PROCESAR Y GUARDAR EMBEDDINGS ===
for chunk_id, texto in tqdm(rows):
    embedding = model.encode(texto)
    embedding_blob = embedding.tobytes()
    cursor.execute(f"""
        INSERT INTO {TABLE_NAME} (chunk_id, embedding)
        VALUES (?, ?)
    """, (chunk_id, embedding_blob))

conn.commit()
conn.close()

print("✅ Embeddings en castellano generados y guardados.")
