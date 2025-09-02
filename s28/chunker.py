# s28/chunker.py
# -*- coding: utf-8 -*-
"""
Segmentador de textos jurídicos para Sistema 28.
- Tokeniza por oraciones (spaCy si está disponible; fallback NLTK).
- Agrupa en chunks coherentes (~80–120 palabras por defecto) sin romper citas ni enumeraciones.
- Maneja abreviaturas frecuentes del dominio (Art., arts., págs., etc.).
- Devuelve metadatos útiles (índices, conteos) para indexación posterior.

Uso básico:
    from s28.chunker import chunk_text
    chunks = chunk_text(texto, target_words=100)

CLI:
    python -m s28.chunker /ruta/al/archivo.txt --min 60 --target 100 --max 140
"""

from __future__ import annotations
import re
import sys
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

@dataclass
class Chunk:
    text: str
    start_char: int
    end_char: int
    word_count: int
    sentence_count: int
    notes: Optional[str] = None

    def asdict(self):
        return asdict(self)

# ----------------------------
# Carga de tokenizadores
# ----------------------------
_SPACY_NLP = None
_NLTK_SENT = None
_NLTK_SPAN = None

def _try_load_spacy() -> Optional[object]:
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    try:
        import spacy  # type: ignore
        # Intentar modelos más robustos primero
        for model in ("es_core_news_md", "es_core_news_sm"):
            try:
                _SPACY_NLP = spacy.load(model, exclude=["ner", "tagger", "lemmatizer"])
                # Asegurar sentencizer si el modelo no trae parser
                if "senter" not in _SPACY_NLP.pipe_names and "parser" not in _SPACY_NLP.pipe_names:
                    senter = _SPACY_NLP.add_pipe("senter", first=True)
                return _SPACY_NLP
            except Exception:
                continue
    except Exception:
        pass
    return None

def _try_load_nltk():
    global _NLTK_SENT, _NLTK_SPAN
    if _NLTK_SENT is not None and _NLTK_SPAN is not None:
        return _NLTK_SENT, _NLTK_SPAN
    try:
        import nltk  # type: ignore
        from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters  # type: ignore

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        punkt_param = PunktParameters()

        # Abreviaturas frecuentes en textos jurídicos hispanos
        abbreviations = {
            "art", "arts", "pág", "págs", "pp", "cap", "caps",
            "inc", "incs", "ap", "aps", "dto", "dcto", "decr",
            "dnu", "dr", "dra", "sr", "sra", "sres", "ud",
            "nac", "prov", "cn", "csjn", "vgr", "ej", "etc",
            "op", "no", "nº", "núm", "num"
        }
        punkt_param.abbrev_types = abbreviations

        _NLTK_SENT = PunktSentenceTokenizer(punkt_param)
        _NLTK_SPAN = _NLTK_SENT.span_tokenize
        return _NLTK_SENT, _NLTK_SPAN
    except Exception:
        return None, None

# ----------------------------
# Preprocesamiento ligero
# ----------------------------

_HARD_BREAK = re.compile(r"(?:\r\n|\r|\n){2,}")  # párrafos
_SOFT_WS = re.compile(r"[ \t]+")
_MULTILINE_SPACES = re.compile(r"[ \t]+\n")
_HEADER_LINE = re.compile(r"^\s*(TÍTULO|SUMARIO|VISTO|CONSIDERANDO|RESUELVE|FALLO|RESULTA|HECHOS|FUNDAMENTOS)\s*:?\s*$", re.IGNORECASE | re.MULTILINE)

# Para evitar cortes tras dos puntos en encabezados/listas (e.g., "VISTO:", "1.")
_COLON_END = re.compile(r".+:$")

# Identificar comillas de apertura que sugieren mantener la próxima oración unida
_OPEN_QUOTE_END = re.compile(r"[“\"«]$")

def _normalize(text: str) -> str:
    # Convertir tabs a espacios, limpiar espacios múltiples y espacios antes de salto de línea
    t = text.replace("\t", " ")
    t = _SOFT_WS.sub(" ", t)
    t = _MULTILINE_SPACES.sub("\n", t)
    # Normalizar saltos de línea múltiples (dejar dobles para marcar párrafo)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

# ----------------------------
# Tokenización por oraciones
# ----------------------------

def _sentences_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """
    Devuelve lista de (oración, start_char, end_char).
    Usa spaCy si está disponible; de lo contrario, NLTK con spans.
    """
    nlp = _try_load_spacy()
    if nlp is not None:
        doc = nlp(text)
        sents = []
        for s in doc.sents:
            sents.append((text[s.start_char:s.end_char], s.start_char, s.end_char))
        if sents:
            return sents

    # Fallback NLTK
    sent_tok, span_tok = _try_load_nltk()
    if span_tok is not None:
        spans = list(span_tok(text))
        return [(text[a:b], a, b) for a, b in spans]

    # Fallback extremo: split ingenuo por punto + espacio
    crude = []
    cursor = 0
    for m in re.finditer(r".+?(?:\.\s+|\n\n|$)", text, flags=re.DOTALL):
        a, b = m.span()
        crude.append((text[a:b].strip(), a, b))
        cursor = b
    if not crude:
        crude = [(text, 0, len(text))]
    return crude

# ----------------------------
# Partición secundaria de oraciones muy largas
# ----------------------------

def _split_long_sentence(sentence: str, start_char: int, max_words: int) -> List[Tuple[str, int, int]]:
    """
    Divide oraciones muy largas por separadores suaves (punto y coma, dos puntos, comas).
    Conserva offsets aproximados a partir de start_char.
    """
    words = sentence.strip().split()
    if len(words) <= max_words:
        return [(sentence.strip(), start_char, start_char + len(sentence.strip()))]

    # Intentar cortes por ; luego : luego ,
    for sep in ["; ", ": ", ", "]:
        parts = sentence.split(sep)
        if len(parts) > 1:
            chunks = []
            offset = start_char
            rebuilt = ""
            for i, part in enumerate(parts):
                seg = (part if i == len(parts) - 1 else part + sep).strip()
                if not seg:
                    continue
                # Recalcular offset "aprox." recorriendo el original acumulado
                idx = sentence.find(seg, 0 if not rebuilt else len(rebuilt))
                if idx >= 0:
                    seg_start = start_char + idx
                else:
                    seg_start = offset
                seg_end = seg_start + len(seg)
                chunks.append((seg, seg_start, seg_end))
                rebuilt += seg
            # Si aún alguno excede, aplicar recursivamente
            final = []
            for s, a, b in chunks:
                final.extend(_split_long_sentence(s, a, max_words))
            return final

    # Último recurso: corte por ventana de palabras
    final = []
    idx = 0
    absolute = start_char
    while idx < len(words):
        window = words[idx: idx + max_words]
        seg = " ".join(window)
        seg = seg.strip()
        pos = sentence.find(seg)
        seg_start = start_char + (pos if pos >= 0 else 0)
        seg_end = seg_start + len(seg)
        final.append((seg, seg_start, seg_end))
        idx += max_words
    return final

# ----------------------------
# Reglas de agrupamiento
# ----------------------------

def _should_glue_next(prev_sentence: str) -> bool:
    prev = prev_sentence.strip()
    if not prev:
        return False
    # Si termina en dos puntos, es encabezado o enunciado que introduce lo que sigue
    if _COLON_END.search(prev):
        return True
    # Si termina con comilla de apertura (caso de comillas mal balanceadas)
    if _OPEN_QUOTE_END.search(prev):
        return True
    return False

# ----------------------------
# Chunks
# ----------------------------

def sentences_to_chunks(
    sentences: List[Tuple[str, int, int]],
    min_words: int = 60,
    target_words: int = 100,
    max_words: int = 140,
) -> List[Chunk]:
    """
    Agrupa oraciones en chunks luego de tener (oración, start, end).
    """
    chunks: List[Chunk] = []
    buf_text: List[str] = []
    buf_start: Optional[int] = None
    buf_end: Optional[int] = None
    buf_words = 0
    buf_sents = 0
    pending_glue = False

    def flush(notes: Optional[str] = None):
        nonlocal buf_text, buf_start, buf_end, buf_words, buf_sents
        if buf_text:
            text = " ".join([t.strip() for t in buf_text]).strip()
            # Limpieza final ligera
            text = re.sub(r"[ \t]+", " ", text)
            chunks.append(
                Chunk(
                    text=text,
                    start_char=buf_start if buf_start is not None else 0,
                    end_char=buf_end if buf_end is not None else (buf_start if buf_start else 0) + len(text),
                    word_count=len(text.split()),
                    sentence_count=buf_sents,
                    notes=notes,
                )
            )
        buf_text = []
        buf_start = None
        buf_end = None
        buf_words = 0
        buf_sents = 0

    for sent, a, b in sentences:
        # Saltar líneas de encabezados solas (pero permitir que peguen la próxima)
        if _HEADER_LINE.fullmatch(sent.strip()):
            if buf_text and buf_words >= min_words:
                flush()
            pending_glue = True
            buf_text.append(sent.strip())
            if buf_start is None:
                buf_start = a
            buf_end = b
            buf_words += len(sent.split())
            buf_sents += 1
            continue

        # Si la oración es excesivamente larga, dividirla
        words_in_sent = len(sent.split())
        sent_parts = []
        if words_in_sent > max_words:
            sent_parts = _split_long_sentence(sent, a, max_words)
        else:
            sent_parts = [(sent, a, b)]

        for sub_sent, sa, sb in sent_parts:
            w = len(sub_sent.split())
            if not buf_text:
                buf_text = [sub_sent.strip()]
                buf_start = sa
                buf_end = sb
                buf_words = w
                buf_sents = 1
                pending_glue = _should_glue_next(sub_sent)
                continue

            # Regla de pegado por encabezado/introducción
            if pending_glue:
                buf_text.append(sub_sent.strip())
                buf_end = sb
                buf_words += w
                buf_sents += 1
                pending_glue = _should_glue_next(sub_sent)
                continue

            # Agregar si no superamos max_words o aún no llegamos al mínimo
            if (buf_words + w) <= max_words or buf_words < min_words:
                buf_text.append(sub_sent.strip())
                buf_end = sb
                buf_words += w
                buf_sents += 1
                pending_glue = _should_glue_next(sub_sent)
                # Si pasamos target y ya no hay obligación de pegar, volcar
                if buf_words >= target_words and not pending_glue:
                    flush()
                continue

            # Si agregar supera y ya estamos razonablemente grandes, volcar y empezar nuevo
            flush()
            buf_text = [sub_sent.strip()]
            buf_start = sa
            buf_end = sb
            buf_words = w
            buf_sents = 1
            pending_glue = _should_glue_next(sub_sent)

    # Volcar resto
    if buf_text:
        flush()

    return chunks

# ----------------------------
# API pública
# ----------------------------

def chunk_text(
    text: str,
    min_words: int = 60,
    target_words: int = 100,
    max_words: int = 140,
    normalize: bool = True,
) -> List[Chunk]:
    """
    Segmenta texto en chunks coherentes para indexación RAG.
    """
    if not isinstance(text, str):
        raise TypeError("text debe ser str")
    if normalize:
        text = _normalize(text)

    # Obtener oraciones con offsets
    sents = _sentences_with_spans(text)

    # Ajuste adicional: unir líneas sueltas muy cortas (listas) con la siguiente oración
    merged = []
    skip_next = False
    for i, (s, a, b) in enumerate(sents):
        if skip_next:
            skip_next = False
            continue
        st = s.strip()
        # Línea de lista muy corta que termina en ":" o es numeración (e.g., "1.")
        if len(st.split()) <= 4 and (_COLON_END.search(st) or re.match(r"^\d+[\.\)]$", st)):
            if i + 1 < len(sents):
                next_s, na, nb = sents[i + 1]
                new_text = (st + " " + next_s).strip()
                merged.append((new_text, a, nb))
                skip_next = True
                continue
        merged.append((s, a, b))

    if merged:
        sents = merged

    # Construcción de chunks
    chunks = sentences_to_chunks(
        sents,
        min_words=min_words,
        target_words=target_words,
        max_words=max_words,
    )
    # Pegado del último chunk si quedó muy corto
    merge_last_under = 55  # palabras
    if len(chunks) >= 2 and chunks[-1].word_count < merge_last_under:
        prev = chunks[-2]
        last = chunks[-1]
        merged_text = (prev.text.rstrip() + " " + last.text.lstrip()).strip()
        chunks[-2] = Chunk(
            text=merged_text,
            start_char=prev.start_char,
            end_char=last.end_char,
            word_count=len(merged_text.split()),
            sentence_count=prev.sentence_count + last.sentence_count,
            notes="merged_tail"
        )
        chunks.pop()
    return chunks

# ----------------------------
# CLI
# ----------------------------

def _cli():
    parser = argparse.ArgumentParser(description="Chunker jurídico para Sistema 28")
    parser.add_argument("path", type=str, help="Ruta a archivo de texto (.txt)")
    parser.add_argument("--min", type=int, default=60, help="Mínimo de palabras por chunk")
    parser.add_argument("--target", type=int, default=100, help="Objetivo de palabras por chunk")
    parser.add_argument("--max", type=int, default=140, help="Máximo de palabras por chunk")
    parser.add_argument("--no-normalize", action="store_true", help="No normalizar espacios/saltos")
    parser.add_argument("--print", action="store_true", help="Imprimir chunks por stdout")
    args = parser.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        raw = f.read()

    chunks = chunk_text(
        raw,
        min_words=args.min,
        target_words=args.target,
        max_words=args.max,
        normalize=not args.no_normalize,
    )

    # Salida simple en JSONL (un chunk por línea)
    import json
    for ch in chunks:
        line = json.dumps(ch.asdict(), ensure_ascii=False)
        print(line)

    if args.print:
        print("\n" + "=" * 80)
        for i, ch in enumerate(chunks, 1):
            print(f"[{i}] ({ch.word_count} palabras, {ch.sentence_count} oraciones) [{ch.start_char}:{ch.end_char}]")
            print(ch.text)
            print("-" * 80)

if __name__ == "__main__":
    _cli()
