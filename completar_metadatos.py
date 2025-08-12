import sqlite3
import re
import os
from datetime import datetime

# Conexión a la base de datos
conn = sqlite3.connect('sistema28.db')
cursor = conn.cursor()

# Buscar todos los registros
cursor.execute("SELECT id, document_path FROM document_chunks")
rows = cursor.fetchall()

def extraer_datos_desde_source(ruta):
    ruta_normalizada = ruta.replace("\\", "/").lower()
    partes = ruta_normalizada.split("/")

    tipo_documento = None
    organismo = None
    fecha = None

    # Buscar organismo por subcarpetas conocidas
    organismos_conocidos = ['csjn', 'onc', 'procuración', 'procuracion', 'procuración caba', 'caba']
    organismo = next((parte.upper() for parte in partes if any(org in parte for org in organismos_conocidos)), None)

    # Buscar tipo de documento por nombre de archivo
    nombre_archivo = os.path.basename(ruta)
    nombre_sin_ext = os.path.splitext(nombre_archivo)[0]

    posibles_tipos = ['fallo', 'dictamen', 'resolución', 'resolucion', 'informe', 'nota', 'disposición', 'disposicion']
    for tipo in posibles_tipos:
        if tipo in nombre_sin_ext.lower():
            tipo_documento = tipo.capitalize()
            break

    # Buscar año o fecha
    match_fecha = re.search(r'(20\d{2})', nombre_sin_ext)
    if match_fecha:
        año = match_fecha.group(1)
        fecha = f"{año}-01-01"

    return tipo_documento, organismo, fecha

# Actualizar cada fila
actualizados = 0
for row_id, source in rows:
    tipo, org, fecha = extraer_datos_desde_source(source)
    cursor.execute("""
        UPDATE document_chunks
        SET tipo_documento = ?, organismo_emisor = ?, fecha_documento = ?
        WHERE id = ?
    """, (tipo, org, fecha, row_id))
    actualizados += 1

conn.commit()
conn.close()

print(f"✅ Metadatos completados en {actualizados} filas.")
