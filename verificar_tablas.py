import sqlite3

conn = sqlite3.connect('sistema28.db')  # Asegurate de que el .db esté en el mismo directorio
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tablas = cursor.fetchall()

print("📦 Tablas encontradas en sistema28.db:")
for tabla in tablas:
    print("-", tabla[0])

conn.close()
