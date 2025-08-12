import sqlite3
import faiss
import os

# Rutas de archivos
DB_PATH = "sistema28.db"
FAISS_INDEX_PATH = "faiss_index.bin"

def verificar_faiss():
    # Verificar que los archivos existen
    if not os.path.exists(DB_PATH):
        print(f"⚠️ No se encontró la base de datos: {DB_PATH}")
        return
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"⚠️ No se encontró el índice FAISS: {FAISS_INDEX_PATH}")
        return

    # Abrir la base de datos
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Contar cuántos registros hay en document_chunks
    cursor.execute("SELECT COUNT(*) FROM document_chunks")
    total_chunks = cursor.fetchone()[0]
    print(f"📚 La base de datos contiene {total_chunks} chunks.")

    # Cargar índice FAISS
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
    except Exception as e:
        print(f"❌ Error al leer el índice FAISS: {e}")
        conn.close()
        return

    # Verificar cantidad de vectores en FAISS
    num_vectors = index.ntotal
    print(f"📦 El índice FAISS contiene {num_vectors} vectores.")

    # Comparar tamaños
    if num_vectors != total_chunks:
        print(f"⚠️ Diferencia detectada: FAISS tiene {num_vectors} vectores, pero la BD tiene {total_chunks} registros.")
    else:
        print("✅ El número de vectores en FAISS coincide con el número de chunks en la base de datos.")

    conn.close()

if __name__ == "__main__":
    verificar_faiss()
