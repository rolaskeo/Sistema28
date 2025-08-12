import sqlite3

# Conectarse a la base de datos
conn = sqlite3.connect("sistema28.db")
cursor = conn.cursor()

# Listar todas las tablas
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tablas = cursor.fetchall()

if not tablas:
    print("⚠️ No hay tablas en la base de datos.")
else:
    print("✅ Tablas encontradas en la base de datos:")
    for t in tablas:
        print("-", t[0])

conn.close()
