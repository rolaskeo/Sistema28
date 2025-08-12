import sqlite3

# Conexión a la base de datos
conn = sqlite3.connect('sistema28.db')
cursor = conn.cursor()

# Consulta para obtener todos los registros de ambas tablas
cursor.execute("SELECT document_path, chunk_text, keywords FROM document_chunks")
original = cursor.fetchall()

cursor.execute("SELECT document_path, chunk_text, keywords FROM document_chunks_backup")
backup = cursor.fetchall()

# Convertir a sets para comparar
set_original = set(original)
set_backup = set(backup)

# Diferencias
solo_en_original = set_original - set_backup
solo_en_backup = set_backup - set_original

# Resultados
if not solo_en_original and not solo_en_backup:
    print("✅ Ambas tablas son idénticas. No hay diferencias detectadas.")
else:
    print("⚠️ Se detectaron diferencias entre las tablas.")
    if solo_en_original:
        print(f"🔹 {len(solo_en_original)} registros están SOLO en 'document_chunks':")
        for i, row in enumerate(list(solo_en_original)[:5]):
            print(f"  {i+1}. {row}")
        if len(solo_en_original) > 5:
            print("  ...")

    if solo_en_backup:
        print(f"🔸 {len(solo_en_backup)} registros están SOLO en 'document_chunks_backup':")
        for i, row in enumerate(list(solo_en_backup)[:5]):
            print(f"  {i+1}. {row}")
        if len(solo_en_backup) > 5:
            print("  ...")

# Cierre
conn.close()
