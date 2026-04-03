import os
import re
import json
import argparse
from typing import Dict, List, Optional, Tuple
from datetime import date

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


# --- Utilidades de fechas ---
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_anio(value) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return int(value)
        s = str(value).strip()
        return int(float(s))
    except Exception:
        return None


def _parse_fecha_number(value) -> Optional[date]:
    if value is None:
        return None
    try:
        n = None
        if isinstance(value, (int, float)):
            n = int(value)
        else:
            n = int(str(value))
        y = n // 10000
        m = (n // 100) % 100
        d = n % 100
        return date(y, m, d)
    except Exception:
        return None


def _parse_fecha_sentencia(value) -> Optional[date]:
    if not value:
        return None
    s = str(value).strip()
    try:
        if ISO_DATE_RE.match(s):
            y, m, d = s.split("-")
            return date(int(y), int(m), int(d))
    except Exception:
        pass
    # No se intenta parsear formatos complejos en español aquí
    return None


def _median(nums: List[int]) -> Optional[float]:
    if not nums:
        return None
    nums = sorted(nums)
    n = len(nums)
    mid = n // 2
    if n % 2 == 1:
        return float(nums[mid])
    return (nums[mid - 1] + nums[mid]) / 2.0


def _mean(nums: List[int]) -> Optional[float]:
    if not nums:
        return None
    return sum(nums) / float(len(nums))


def _percentile(nums: List[int], p: float) -> Optional[int]:
    if not nums:
        return None
    nums = sorted(nums)
    if p <= 0:
        return nums[0]
    if p >= 100:
        return nums[-1]
    k = int(round((p / 100.0) * (len(nums) - 1)))
    return nums[k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analiza años y fechas en metadatos de Pinecone (no modifica BD)")
    parser.add_argument("--index-name", default=os.getenv("PINECONE_INDEX_NAME", "relatoria-emebeddings"))
    parser.add_argument("--sample-size", type=int, default=0, help="0 = todo el índice si es posible")
    parser.add_argument("--use-semantic", action="store_true", help="Usar conexión inicializada en models.semantic_search")
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

    anios: List[int] = []
    fechas_num: List[date] = []
    fechas_sent: List[date] = []
    missing_anio = 0
    invalid_anio = 0
    missing_fecha_num = 0
    invalid_fecha_num = 0
    missing_fecha_sent = 0

    counts_por_anio: Dict[int, int] = {}
    counts_por_tipo: Dict[str, int] = {}
    counts_por_anio_tipo: Dict[int, Dict[str, int]] = {}

    mismatches_anio_vs_fecha_num = 0
    mismatch_examples: List[Dict] = []

    oldest_examples: List[Dict] = []
    newest_examples: List[Dict] = []

    # Recorremos todos los registros
    for m in matches:
        md = m.get("metadata", {})
        anio_val = _parse_anio(md.get("anio"))
        fecha_num_val = _parse_fecha_number(md.get("fecha_number"))
        fecha_sent_val = _parse_fecha_sentencia(md.get("Fecha Sentencia"))

        tipo = str(md.get("Tipo", "") or "")
        providencia = md.get("Providencia")
        filename = md.get("filename")

        if anio_val is None:
            missing_anio += 1
        else:
            anios.append(anio_val)
            counts_por_anio[anio_val] = counts_por_anio.get(anio_val, 0) + 1
            if tipo:
                counts_por_tipo[tipo] = counts_por_tipo.get(tipo, 0) + 1
                counts_por_anio_tipo.setdefault(anio_val, {})
                counts_por_anio_tipo[anio_val][tipo] = counts_por_anio_tipo[anio_val].get(tipo, 0) + 1

        if fecha_num_val is None:
            if md.get("fecha_number") is None:
                missing_fecha_num += 1
            else:
                invalid_fecha_num += 1
        else:
            fechas_num.append(fecha_num_val)

        if fecha_sent_val is None:
            if md.get("Fecha Sentencia"):
                # si existe pero no parsea, lo contamos como missing no estricto
                missing_fecha_sent += 1
            else:
                missing_fecha_sent += 1
        else:
            fechas_sent.append(fecha_sent_val)

        # Mismatch entre anio y fecha_number (si ambos existen)
        if anio_val is not None and fecha_num_val is not None:
            if anio_val != fecha_num_val.year:
                mismatches_anio_vs_fecha_num += 1
                if len(mismatch_examples) < 15:
                    mismatch_examples.append({
                        "id": m.get("id"),
                        "Providencia": providencia,
                        "Tipo": tipo,
                        "anio": anio_val,
                        "fecha_number": md.get("fecha_number"),
                        "fecha_number_parsed": fecha_num_val.isoformat(),
                        "filename": filename,
                    })

    # Estadísticas principales para anio
    anio_min = min(anios) if anios else None
    anio_max = max(anios) if anios else None
    anio_mean = _mean(anios)
    anio_median = _median(anios)
    anio_p10 = _percentile(anios, 10.0)
    anio_p90 = _percentile(anios, 90.0)

    # Fechas number
    fechas_num_sorted = sorted(fechas_num)
    fecha_num_min = fechas_num_sorted[0].isoformat() if fechas_num_sorted else None
    fecha_num_max = fechas_num_sorted[-1].isoformat() if fechas_num_sorted else None

    # Fechas sentencia (ISO, si disponibles)
    fechas_sent_sorted = sorted(fechas_sent)
    fecha_sent_min = fechas_sent_sorted[0].isoformat() if fechas_sent_sorted else None
    fecha_sent_max = fechas_sent_sorted[-1].isoformat() if fechas_sent_sorted else None

    # Ejemplos extremos (usando anio cuando disponible, si no fecha_number)
    def _make_example(m: Dict) -> Dict:
        md = m.get("metadata", {})
        return {
            "id": m.get("id"),
            "Providencia": md.get("Providencia"),
            "Tipo": md.get("Tipo"),
            "anio": md.get("anio"),
            "fecha_number": md.get("fecha_number"),
            "Fecha Sentencia": md.get("Fecha Sentencia"),
            "filename": md.get("filename"),
        }

    # Selección aproximada de ejemplos:
    # tomamos los primeros que coincidan con anio_min y anio_max
    if anio_min is not None:
        for m in matches:
            if _parse_anio(m.get("metadata", {}).get("anio")) == anio_min:
                oldest_examples.append(_make_example(m))
                if len(oldest_examples) >= 5:
                    break
    if anio_max is not None:
        for m in matches:
            if _parse_anio(m.get("metadata", {}).get("anio")) == anio_max:
                newest_examples.append(_make_example(m))
                if len(newest_examples) >= 5:
                    break

    result = {
        "index_name": args.index_name,
        "dimension": dim,
        "total_records": total,
        "anio_stats": {
            "min": anio_min,
            "max": anio_max,
            "mean": round(anio_mean or 0, 2) if anio_mean is not None else None,
            "median": anio_median,
            "p10": anio_p10,
            "p90": anio_p90,
            "missing": missing_anio,
            "invalid": invalid_anio,
            "counts_por_anio": dict(sorted(counts_por_anio.items()))
        },
        "fecha_number_stats": {
            "min": fecha_num_min,
            "max": fecha_num_max,
            "missing": missing_fecha_num,
            "invalid": invalid_fecha_num
        },
        "fecha_sentencia_stats": {
            "min": fecha_sent_min,
            "max": fecha_sent_max,
            "missing": missing_fecha_sent
        },
        "counts_por_tipo": dict(sorted(counts_por_tipo.items(), key=lambda x: (-x[1], x[0]))),
        "counts_por_anio_tipo": {str(k): v for k, v in sorted(counts_por_anio_tipo.items())},
        "mismatches_anio_vs_fecha_number": {
            "count": mismatches_anio_vs_fecha_num,
            "examples": mismatch_examples
        },
        "oldest_examples": oldest_examples,
        "newest_examples": newest_examples,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()