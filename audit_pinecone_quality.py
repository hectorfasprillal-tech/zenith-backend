import os
import re
import json
import argparse
from typing import Dict, List, Optional, Tuple

try:
    from pinecone import Pinecone
except Exception:
    Pinecone = None  # type: ignore


def _connect_via_semantic_search() -> Tuple[Optional[object], Optional[int]]:
    try:
        from models.semantic_search import is_ready, index  # type: ignore
        if is_ready():
            return index, 384
    except Exception:
        pass
    return None, None


def _connect_direct(index_name: str, api_key: Optional[str]) -> Tuple[object, int]:
    if Pinecone is None:
        raise RuntimeError("Dependencia 'pinecone' no disponible. Instala el SDK nuevo: pip install pinecone")
    key = api_key or os.getenv("PINECONE_API_KEY")
    if not key:
        raise RuntimeError("PINECONE_API_KEY no configurado en entorno")
    pc = Pinecone(api_key=key)
    idx = pc.Index(index_name)
    dim = 384
    try:
        info = pc.describe_index(index_name)
        if isinstance(info, dict):
            dim = int(info.get("dimension", dim))
        else:
            dim = int(getattr(info, "dimension", dim))
    except Exception:
        pass
    return idx, dim


def _get_total_count(idx: object) -> Optional[int]:
    try:
        stats = idx.describe_index_stats()  # type: ignore
        if isinstance(stats, dict):
            return int(stats.get("total_vector_count") or stats.get("total_count") or 0)
        return int(getattr(stats, "total_vector_count", 0) or getattr(stats, "total_count", 0))
    except Exception:
        return None


def _sample_records(idx: object, dim: int, sample_size: int) -> List[Dict]:
    if sample_size <= 0:
        sample_size = int(_get_total_count(idx) or 2000)
    r = idx.query(vector=[0.0] * dim, top_k=int(sample_size), include_metadata=True)  # type: ignore
    matches = r.get("matches") if isinstance(r, dict) else getattr(r, "matches", [])
    return matches


# --- Heurísticas de calidad ---
HEX_RE = re.compile(r"[0-9A-Fa-f]{16,}")
RTF_ARTIFACTS = ("{\\rtf", "\\par", "EMF", "PNG", "IHDR", "PLTE")
PLACEHOLDERS = {"N/A", "", "M E", "ME", "N A", "S/N", "NA", "NO APLICA", "SIN RESUMEN", "SIN INFORMACION"}
MIN_STRICT_CHARS = 50
MIN_STRICT_WORDS = 8


def _ratio_digits(text: str) -> float:
    if not text:
        return 1.0
    digits = sum(c.isdigit() for c in text)
    return digits / max(1, len(text))


def _ratio_non_letters(text: str) -> float:
    if not text:
        return 1.0
    non_letters = sum(not (c.isalpha() or c.isspace()) for c in text)
    return non_letters / max(1, len(text))


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def _clean_filename(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return os.path.splitext(name)[0]


def audit_metadata(md: Dict) -> List[str]:
    reasons: List[str] = []
    resumen = str(md.get("Resumen", "") or "")
    summary_extract = str(md.get("summary_extract", "") or "")
    informacion = str(md.get("Información", "") or md.get("Informacion", "") or "")
    providencia = str(md.get("Providencia", "") or "")
    filename = _clean_filename(md.get("filename")) or ""
    ident = str(md.get("ID", "") or "")

    # Falta de información
    if not resumen and not summary_extract:
        reasons.append("sin_resumen")
    if not informacion:
        reasons.append("sin_informacion")

    # Demasiado cortos
    if resumen and len(resumen.strip()) < 15:
        reasons.append("resumen_corto")
    if summary_extract and len(summary_extract.strip()) < 15:
        reasons.append("summary_extract_corto")
    if informacion and len(informacion.strip()) < 15:
        reasons.append("informacion_corta")

    # Muy cortos (más estrictos): pocas palabras o pocos caracteres
    if resumen and (len(resumen.strip()) < MIN_STRICT_CHARS or _word_count(resumen) < MIN_STRICT_WORDS):
        reasons.append("resumen_muy_corto")
    if summary_extract and (len(summary_extract.strip()) < MIN_STRICT_CHARS or _word_count(summary_extract) < MIN_STRICT_WORDS):
        reasons.append("summary_extract_muy_corto")
    if informacion and (len(informacion.strip()) < MIN_STRICT_CHARS or _word_count(informacion) < MIN_STRICT_WORDS):
        reasons.append("informacion_muy_corto")

    # Placeholders / basura binaria/hex
    blob_checks = []
    for field_name, text in ("resumen", resumen), ("summary_extract", summary_extract), ("informacion", informacion):
        if text in PLACEHOLDERS:
            blob_checks.append(f"{field_name}_placeholder")
        if HEX_RE.search(text):
            blob_checks.append(f"{field_name}_hex")
        if any(tag in text for tag in RTF_ARTIFACTS):
            blob_checks.append(f"{field_name}_rtf_artifacts")
        # Predominio de números y baja proporción alfabética
        if _ratio_digits(text) > 0.35:
            blob_checks.append(f"{field_name}_predominio_numeros")
        if _ratio_non_letters(text) > 0.5:
            blob_checks.append(f"{field_name}_poco_alfabetico")
        if _ratio_digits(text) > 0.5 or _ratio_non_letters(text) > 0.7:
            if len(text) >= 40:
                blob_checks.append(f"{field_name}_no_alfabetico")
    reasons.extend(blob_checks)

    # Duplicados con título/ID/filename
    for text, name in ((resumen, "resumen"), (summary_extract, "summary_extract")):
        if text:
            if providencia and text.strip() == providencia.strip():
                reasons.append(f"{name}_igual_providencia")
            if filename and text.strip() == filename.strip():
                reasons.append(f"{name}_igual_filename")
            if ident and text.strip() == ident.strip():
                reasons.append(f"{name}_igual_id")
            tema = str(md.get("Tema", "") or "")
            tema_sub = str(md.get("Tema - subtema", "") or "")
            tipo = str(md.get("Tipo", "") or "")
            if tema and text.strip() == tema.strip():
                reasons.append(f"{name}_igual_tema")
            if tema_sub and text.strip() == tema_sub.strip():
                reasons.append(f"{name}_igual_tema_subtema")
            if tipo and text.strip() == tipo.strip():
                reasons.append(f"{name}_igual_tipo")

    # Resumen igual a Información y entre sí
    if resumen and informacion and resumen.strip() == informacion.strip():
        reasons.append("resumen_igual_informacion")
    if summary_extract and informacion and summary_extract.strip() == informacion.strip():
        reasons.append("summary_extract_igual_informacion")
    if resumen and summary_extract and resumen.strip() == summary_extract.strip():
        reasons.append("resumen_igual_summary_extract")

    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita calidad de metadatos (resúmenes) en Pinecone sin modificar BD.")
    parser.add_argument("--index-name", default=os.getenv("PINECONE_INDEX_NAME", "relatoria-emebeddings"))
    parser.add_argument("--sample-size", type=int, default=0, help="0 = todo el índice si es posible")
    parser.add_argument("--use-semantic", action="store_true", help="Usar conexión ya inicializada en models.semantic_search")
    parser.add_argument("--max-examples", type=int, default=25, help="Cantidad de ejemplos problemáticos a mostrar")
    parser.add_argument("--output", default="", help="Ruta opcional para guardar JSON con resultados")
    args = parser.parse_args()

    idx = None
    dim = None
    if args.use_semantic:
        idx, dim = _connect_via_semantic_search()
    if idx is None or dim is None:
        idx, dim = _connect_direct(args.index_name, None)

    matches = _sample_records(idx, dim, args.sample_size)
    total = len(matches)
    reason_counts: Dict[str, int] = {}
    bad_examples: List[Dict] = []
    bad_total = 0

    for m in matches:
        md = m.get("metadata", {})
        reasons = audit_metadata(md)
        if reasons:
            bad_total += 1
            for r in reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1
            if len(bad_examples) < args.max_examples:
                bad_examples.append({
                    "id": m.get("id"),
                    "filename": md.get("filename"),
                    "Providencia": md.get("Providencia"),
                    "Tipo": md.get("Tipo"),
                    "anio": md.get("anio"),
                    "reasons": reasons,
                    "Resumen_len": len(str(md.get("Resumen", ""))),
                    "summary_extract_len": len(str(md.get("summary_extract", ""))),
                    "Informacion_len": len(str(md.get("Información", md.get("Informacion", "")))),
                })

    result = {
        "index_name": args.index_name,
        "dimension": dim,
        "total_records": total,
        "flagged_records": bad_total,
        "flag_rate": round((bad_total / max(1, total)) * 100, 2),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda x: (-x[1], x[0]))),
        "examples": bad_examples,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()