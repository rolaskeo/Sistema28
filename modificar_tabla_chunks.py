import sqlite3

# Conectar con la base de datos
conn = sqlite3.connect('sistema28.db')
cursor = conn.cursor()

# Agregar las nuevas columnas, si no existen
try:
    cursor.execute("ALTER TABLE document_chunks ADD COLUMN tipo_documento TEXT")
    print("✅ Columna 'tipo_documento' agregada.")
except sqlite3.OperationalError:
    print("⚠️ La columna 'tipo_documento' ya existe.")

try:
    cursor.execute("ALTER TABLE document_chunks ADD COLUMN organismo_emisor TEXT")
    print("✅ Columna 'organismo_emisor' agregada.")
except sqlite3.OperationalError:
    print("⚠️ La columna 'organismo_emisor' ya existe.")

try:
    cursor.execute("ALTER TABLE document_chunks ADD COLUMN fecha_documento TEXT")
    print("✅ Columna 'fecha_documento' agregada.")
except sqlite3.OperationalError:
    print("⚠️ La columna 'fecha_documento' ya existe.")

# Confirmar y cerrar
conn.commit()
conn.close()
