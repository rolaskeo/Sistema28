# scripts/ingest_repo_v3.py
# -*- coding: utf-8 -*-
"""
Recorre recursivamente una carpeta "Repositorio" e ingesta todo lo soportado (PDF/TXT/DOCX)
en el esquema V3: documents_v3, chunks_v3, embeddings_v3, con sentence chunking (spaCy).
No toca las tablas viejas.
"""

import os, sys, argparse, glob, subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def run(cmd):
    print(">>", " ".join(cmd))
    res = subprocess.run(cmd, shell=False)
    if res.returncode != 0:
        raise SystemExit(f"Error ejecutando: {' '.join(cmd)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Ruta al sistema28.db")
    ap.add_argument("--repo", required=True, help="Ruta a la carpeta 'Repositorio'")
    ap.add_argument("--spacy_model", default="es_core_news_md")
    args = ap.parse_args()

    # Patrones por extensión (recursivos)
    patterns = ["**/*.pdf", "**/*.txt", "**/*.docx"]

    # Confirmar existencia
    repo = os.path.abspath(args.repo)
    if not os.path.isdir(repo):
        raise SystemExit(f"No existe la carpeta: {repo}")

    # Para cada patrón, llamamos al ingestor v3 (tu módulo)
    # Aquí asumimos que s28/ingestor_v3.py ya soporta --repo y --glob con glob recursivo (**) y usa spaCy.
    for pat in patterns:
        cmd = [
            sys.executable, "-m", "s28.ingestor_v3",
            "--db", args.db,
            "--repo", repo,
            "--glob", pat,
            "--spacy", args.spacy_model,
            "--max-words", "120",     # chunk ~100–120 palabras
            "--min-words", "60",      # evita trozos muy chicos
            "--normalize",            # normalización ligera para FTS
        ]
        run(cmd)

    print("OK: Ingesta V3 completada (PDF/TXT/DOCX).")

if __name__ == "__main__":
    main()
