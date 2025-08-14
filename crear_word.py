"""
crear_word.py — Módulo de exportación de dictámenes en DOCX usando una plantilla.

Requisitos:
  pip install docxtpl

Uso típico desde generar_dictamen.py:
  from crear_word import exportar_dictamen_docx
  exportar_dictamen_docx(contexto, "dictamenes/2025-08-12-dictamen.docx")

La plantilla esperada (por defecto): templates/Dictamen_modelo_1.docx
Debe incluir placeholders Jinja2 como {{titulo}}, {{fecha}}, {{referencias}},
{{consulta_concreta}}, {{palabras_clave}}, {{cuerpo}}, {{firmante}}, {{organismo}}.

Si tu plantilla tiene secciones (Antecedentes/Análisis/Conclusión) podés usar
{{antecedentes}}, {{analisis}}, {{conclusion}} dentro de {{cuerpo}} o por separado.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional, Sequence
import os

try:
    from docxtpl import DocxTemplate
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Falta docxtpl. Instalá: pip install docxtpl"
    ) from e


DEFAULT_TEMPLATE = os.getenv("DICTAMEN_TEMPLATE", "templates/Dictamen_modelo_1.docx")


@dataclass
class DictamenContexto:
    titulo: str
    organismo: str = ""
    fecha: str = date.today().strftime("%d/%m/%Y")
    referencias: str = ""
    consulta_concreta: str = ""
    palabras_clave: Sequence[str] | str = ()
    cuerpo: str = ""  # Texto completo (puede concatenar secciones)
    antecedentes: str = ""
    analisis: str = ""
    conclusion: str = ""
    firmante: str = "Rolando Keumurdji Rizzuti"

    def to_mapping(self) -> Dict[str, str]:
        """Convierte a dict listo para docxtpl, normalizando claves y listas."""
        palabras = self.palabras_clave
        if isinstance(palabras, (list, tuple)):
            palabras = ", ".join([p for p in palabras if p])
        return {
            "titulo": self.titulo or "Dictamen Jurídico",
            "fecha": self.fecha or date.today().strftime("%d/%m/%Y"),
            "organismo": self.organismo or "",
            "referencias": self.referencias or "",
            "consulta_concreta": self.consulta_concreta or "",
            "palabras_clave": palabras or "",
            "cuerpo": self.cuerpo or _unir_secciones(self.antecedentes, self.analisis, self.conclusion),
            "antecedentes": self.antecedentes or "",
            "analisis": self.analisis or "",
            "conclusion": self.conclusion or "",
            "firmante": self.firmante or "",
        }


def _unir_secciones(antecedentes: str, analisis: str, conclusion: str) -> str:
    partes = []
    if antecedentes:
        partes.append("I. ANTECEDENTES\n" + antecedentes.strip())
    if analisis:
        partes.append("II. ANÁLISIS\n" + analisis.strip())
    if conclusion:
        partes.append("III. CONCLUSIÓN\n" + conclusion.strip())
    return "\n\n".join(partes)


def exportar_dictamen_docx(
    contexto: Dict[str, str] | DictamenContexto,
    salida: str | os.PathLike,
    template_path: Optional[str | os.PathLike] = None,
) -> Path:
    """
    Renderiza la plantilla y guarda el DOCX en `salida`.

    - `contexto` puede ser `DictamenContexto` o un `dict` compatible.
    - `template_path` por defecto toma DEFAULT_TEMPLATE.

    El template debe contener los placeholders Jinja2 que quieras usar.
    Placeholders soportados por este módulo:
      {{titulo}}, {{fecha}}, {{organismo}}, {{referencias}},
      {{consulta_concreta}}, {{palabras_clave}}, {{cuerpo}},
      {{antecedentes}}, {{analisis}}, {{conclusion}}, {{firmante}}
    """
    tpl_path = Path(template_path or DEFAULT_TEMPLATE)
    if not tpl_path.exists():
        raise FileNotFoundError(
            f"No se encontró la plantilla DOCX en: {tpl_path}. "
            "Asegurate de colocar tu modelo en esa ruta o pasar template_path explícito."
        )

    if isinstance(contexto, DictamenContexto):
        ctx = contexto.to_mapping()
    elif isinstance(contexto, dict):
        # normalizar llaves faltantes para evitar KeyError en el template
        base = DictamenContexto(titulo=contexto.get("titulo", "Dictamen Jurídico"))
        base.organismo = contexto.get("organismo", base.organismo)
        base.fecha = contexto.get("fecha", base.fecha)
        base.referencias = contexto.get("referencias", base.referencias)
        base.consulta_concreta = contexto.get("consulta_concreta", base.consulta_concreta)
        base.palabras_clave = contexto.get("palabras_clave", base.palabras_clave)
        base.cuerpo = contexto.get("cuerpo", base.cuerpo)
        base.antecedentes = contexto.get("antecedentes", base.antecedentes)
        base.analisis = contexto.get("analisis", base.analisis)
        base.conclusion = contexto.get("conclusion", base.conclusion)
        base.firmante = contexto.get("firmante", base.firmante)
        ctx = base.to_mapping()
    else:  # pragma: no cover
        raise TypeError("contexto debe ser DictamenContexto o dict")

    tpl = DocxTemplate(str(tpl_path))
    tpl.render(ctx)

    out_path = Path(salida)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tpl.save(str(out_path))
    return out_path


# --- utilidad opcional para generar un nombre de archivo limpio ---
import re

def nombre_archivo_dictamen(titulo: str, fecha: Optional[datetime] = None, carpeta: str = "dictamenes") -> Path:
    fecha = fecha or datetime.now()
    base = re.sub(r"[^\w\-]+", "_", titulo.strip(), flags=re.UNICODE)
    base = re.sub(r"_+", "_", base).strip("_")
    return Path(carpeta) / f"{fecha.strftime('%Y-%m-%d')}-{base or 'Dictamen'}.docx"


if __name__ == "__main__":  # pequeña prueba manual
    ctx = DictamenContexto(
        titulo="Dictamen Jurídico – Ejemplo",
        organismo="Ministerio X",
        referencias="Expte. 123/2025",
        consulta_concreta="Naturaleza de la oferta.",
        palabras_clave=["Oferta", "Licitación Pública", "Contratación Pública"],
        antecedentes="Descripción breve de antecedentes…",
        analisis="Análisis del caso…",
        conclusion="Conclusiones y recomendaciones…",
    )
    salida = nombre_archivo_dictamen(ctx.titulo)
    ruta = exportar_dictamen_docx(ctx, salida)
    print(f"Dictamen generado en: {ruta}")
