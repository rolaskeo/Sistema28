import sqlite3
import numpy as np
import struct

# Función para convertir BLOB a array de floats
def blob_to_array(blob):
    return np.array(struct.unpack(f'{len(blob)//4}f', blob), dtype=np.float32)

# Función para convertir array a BLOB
def array_to_blob(array):
    return struct.pack(f'{len(array)}f', *array)

# Conexión a la base de datos
conn = sqlite3.connect("sistema28.db")
cursor = conn.cursor()

# Crear tabla documentos si no existe
cursor.execute("""
CREATE TABLE IF NOT EXISTS documentos (
    id INTEGER PRIMARY KEY,
    contenido TEXT,
    embedding BLOB
);
""")

# Verificar si ya hay datos
cursor.execute("SELECT COUNT(*) FROM documentos")
count = cursor.fetchone()[0]

if count > 0:
    print("⚠️ La tabla 'documentos' ya tiene datos. No se insertará nada.")
else:
    # Traer chunks y embeddings
    cursor.execute("SELECT dc.id, dc.chunk_text, ce.embedding FROM document_chunks dc JOIN chunk_embeddings ce ON dc.id = ce.chunk_id")
    rows = cursor.fetchall()

    for row in rows:
        chunk_id, chunk_text, embedding_blob = row
        embedding_array = blob_to_array(embedding_blob)
        cursor.execute(
            "INSERT INTO documentos (id, contenido, embedding) VALUES (?, ?, ?)",
            (chunk_id, chunk_text, array_to_blob(embedding_array))
        )

    conn.commit()
    print(f"✅ Se insertaron {len(rows)} registros en la tabla 'documentos'.")

conn.close()
