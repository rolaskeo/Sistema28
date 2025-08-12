# respaldar_chunks.py

import sqlite3

conn = sqlite3.connect('sistema28.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS document_chunks_backup AS
    SELECT * FROM document_chunks
''')

conn.commit()
conn.close()

print("🛡️ Tabla 'document_chunks_backup' creada como respaldo.")
