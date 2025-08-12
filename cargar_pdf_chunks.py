import os
import sqlite3
import fitz  # PyMuPDF
import nltk

# Descargar recursos NLTK necesarios (si no están)
nltk.download('punkt')
nltk.download('stopwords')

# ----------------- CONFIGURACIONES -----------------

DB_PATH = 'sistema28.db'  # Ruta a tu base de datos SQLite
PDF_FOLDER = r'C:\Sistema 28\Repositorio'  # Ruta absoluta a carpeta con PDFs
MAX_PALABRAS = 50
MAX_KEYWORDS = 5

# Diccionario jurídico simplificado (puedes ampliarlo)
DICCIONARIO_JURIDICO = set([
    "expropiación", "ad referéndum", "contrato", "licitación", "adjudicación", "prescripción",
    "legitimación", "oferta", "servicio", "jurisdicción", "nulidad", "intervención", "administración",
    "sanción", "recurso", "revisión", "lesividad", "concesión", "plazo", "caducidad"
])

# ----------------- FUNCIONES -----------------

def extraer_texto_pdf(ruta_pdf):
    doc = fitz.open(ruta_pdf)
    texto = ""
    for pagina in doc:
        texto += pagina.get_text()
    return texto

def dividir_en_chunks(texto, max_palabras=50):
    oraciones = nltk.sent_tokenize(texto, language='spanish')
    chunks = []
    actual = ""
    palabras_actuales = 0

    for oracion in oraciones:
        palabras = oracion.split()
        if palabras_actuales + len(palabras) <= max_palabras:
            actual += " " + oracion
            palabras_actuales += len(palabras)
        else:
            if actual:
                chunks.append(actual.strip())
            actual = oracion
            palabras_actuales = len(palabras)

    if actual:
        chunks.append(actual.strip())

    return chunks

def generar_keywords(texto_chunk):
    palabras = nltk.word_tokenize(texto_chunk.lower(), language='spanish')
    palabras_filtradas = [p for p in palabras if p in DICCIONARIO_JURIDICO]
    return list(set(palabras_filtradas))[:MAX_KEYWORDS]

def inicializar_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Borrar datos previos para limpieza total
    cursor.execute("DELETE FROM document_chunks")
    conn.commit()
    print("🧹 Tabla 'document_chunks' vaciada correctamente.")
    conn.close()

def guardar_chunk(fuente, chunk, keywords):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO document_chunks (document_path, chunk_text, keywords)
        VALUES (?, ?, ?)
    ''', (fuente, chunk, ', '.join(keywords)))
    conn.commit()
    conn.close()

def procesar_pdfs():
    inicializar_db()
    for carpeta_raiz, _, archivos in os.walk(PDF_FOLDER):
        print(f"📂 Explorando carpeta: {carpeta_raiz}")
        for archivo in archivos:
            if archivo.lower().endswith('.pdf'):
                ruta_pdf = os.path.join(carpeta_raiz, archivo)
                print(f"📄 Procesando PDF: {ruta_pdf}")
                texto = extraer_texto_pdf(ruta_pdf)
                chunks = dividir_en_chunks(texto, MAX_PALABRAS)
                for chunk in chunks:
                    keywords = generar_keywords(chunk)
                    guardar_chunk(ruta_pdf, chunk, keywords)

# ----------------- EJECUCIÓN -----------------

if __name__ == "__main__":
    procesar_pdfs()
    print("✅ Carga finalizada con chunks reducidos y palabras clave.")
