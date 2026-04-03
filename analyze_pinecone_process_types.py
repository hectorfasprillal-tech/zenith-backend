import os
import json
import argparse
from collections import Counter
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


def _sample_metadata(idx: object, dim: int, sample_size: int) -> List[Dict]:
    # Si sample_size <= 0, intentamos obtener todo el conteo del índice
    if sample_size <= 0:
        total = _get_total_count(idx) or 2000
        sample_size = int(total)
    # Consultamos metadatos sin valores para reducir carga
    r = idx.query(vector=[0.0] * dim, top_k=int(sample_size), include_metadata=True)  # type: ignore
    matches = r.get("matches") if isinstance(r, dict) else getattr(r, "matches", [])
    return [m.get("metadata", {}) for m in matches]


def _summarize_field(metadata_list: List[Dict], field: str) -> Dict[str, int]:
    c = Counter()
    for md in metadata_list:
        val = md.get(field)
        if val is None and field.lower() == "tipo":
            val = md.get("tipo")
        if val is None:
            val = "N/A"
        if isinstance(val, list):
            for v in val:
                c[str(v)] += 1
        else:
            c[str(val)] += 1
    return dict(sorted(c.items(), key=lambda x: (-x[1], x[0])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analiza la distribución de tipos de proceso en Pinecone sin modificar BD.")
    parser.add_argument("--index-name", default=os.getenv("PINECONE_INDEX_NAME", "relatoria-emebeddings"))
    parser.add_argument("--sample-size", type=int, default=0, help="Cantidad a muestrear (0 = todo el índice si es posible)")
    parser.add_argument("--field", default="Tipo", help="Campo de metadatos a resumir (por defecto 'Tipo')")
    parser.add_argument("--output", default="", help="Ruta opcional para guardar JSON de resultados")
    parser.add_argument("--use-semantic", action="store_true", help="Usar conexión ya inicializada en models.semantic_search")
    args = parser.parse_args()

    idx = None
    dim = None

    if args.use_semantic:
        idx, dim = _connect_via_semantic_search()

    if idx is None or dim is None:
        idx, dim = _connect_direct(args.index_name, None)

    md_list = _sample_metadata(idx, dim, args.sample_size)
    distribution = _summarize_field(md_list, args.field)

    result = {
        "index_name": args.index_name,
        "dimension": dim,
        "sample_size": len(md_list),
        "field": args.field,
        "distribution": distribution,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()