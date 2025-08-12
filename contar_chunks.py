import sqlite3

def contar_chunks():
    conn = sqlite3.connect('sistema28.db')
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM document_chunks")
    total_chunks = cursor.fetchone()[0]

    print(f"🧩 Total de chunks en la base: {total_chunks}")

    conn.close()

if __name__ == "__main__":
    contar_chunks()
