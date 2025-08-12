import sqlite3

conn = sqlite3.connect('sistema28.db')
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(document_chunks)")
columnas = cursor.fetchall()

print("📋 Columnas de 'document_chunks':")
for col in columnas:
    print(f"- {col[1]}")

conn.close()
