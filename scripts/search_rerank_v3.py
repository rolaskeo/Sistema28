# scripts/search_rerank_v3.py
import sqlite3
from s28.rerank import search_print

DB = r"C:\Rolex\Python\Sistema28Script\sistema28.db"

if __name__ == "__main__":
    conn = sqlite3.connect(DB)
    # Probá varias:
    search_print(conn, "redeterminación de precios en obra pública", topk=5, candidates=200)
    search_print(conn, "impugnación de pliegos en contrataciones públicas", topk=5, candidates=200)
    search_print(conn, "garantía de mantenimiento de oferta y su ejecución", topk=5, candidates=200)
