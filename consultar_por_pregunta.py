import sqlite3
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

print("🧠 Cargando modelo de lenguaje...")
model = SentenceTransformer('dccuchile/bert-base-spanish-wwm-uncased')

def buscar_similares(pregunta, top_k=5):
    embedding_consulta = model.encode(pregunta, convert_to_tensor=True)

    conn = sqlite3.connect('sistema28.db')
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        dc.id,
        dc.chunk_text,
        dc.document_path,
        dc.titulo,
        dc.fecha_documento,
        ce.embedding
    FROM
        document_chunks dc
    JOIN
        chunk_embeddings ce ON dc.id = ce.chunk_id
    """)

    resultados = []

    for row in cursor.fetchall():
        chunk_id = row[0]
        chunk_text = row[1]
        document_path = row[2]
        titulo = row[3]
        fecha_documento = row[4]
        emb_blob = row[5]

        emb_array = np.frombuffer(emb_blob, dtype=np.float32)
        emb_tensor = torch.from_numpy(emb_array)

        similitud = util.cos_sim(embedding_consulta, emb_tensor)[0][0].item()

        resultados.append({
            'id': chunk_id,
            'texto': chunk_text,
            'similitud': similitud,
            'document_path': document_path,
            'titulo': titulo,
            'fecha_documento': fecha_documento
        })

    conn.close()

    resultados.sort(key=lambda x: x['similitud'], reverse=True)

    return resultados[:top_k]

if __name__ == "__main__":
    print("🔎 Ingresá tu consulta jurídica:")
    pregunta = input("> ")

    similares = buscar_similares(pregunta)

    if not similares:
        print("No se encontraron resultados relevantes.")
    else:
        print("\n📚 Fragmentos relevantes encontrados:\n")
        for i, res in enumerate(similares, 1):
            print(f"🔹 Resultado #{i}")
            print(f"🧠 Similitud: {res['similitud']:.4f}")
            print(f"📄 Documento: {res['titulo']} ({res['document_path']})")
            print(f"📅 Fecha: {res['fecha_documento']}")
            print(f"📝 Texto:\n{res['texto']}\n{'-'*100}")
