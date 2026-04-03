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


# --- Helpers de normalización y split ---
SEP_RE = re.compile(r"[;,\|/]+")


def _canon(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = str(s).strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" .;:,-")
    return t or None


def _split_values(value: Optional[str]) -> List[str]:
    if not value:
        return []
    s = str(value)
    parts = [p.strip() for p in SEP_RE.split(s) if p.strip()]
    return parts if parts else ([s.strip()] if s.strip() else [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analiza 'Tema' y 'Tema - subtema' en Pinecone")
    parser.add_argument("--index-name", default=os.getenv("PINECONE_INDEX_NAME", "relatoria-emebeddings"))
    parser.add_argument("--sample-size", type=int, default=0, help="0 = todo el índice si es posible")
    parser.add_argument("--use-semantic", action="store_true", help="Usar conexión inicializada en models.semantic_search")
    parser.add_argument("--top-k", type=int, default=30, help="Cuántos temas/subtemas principales devolver")
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

    tema_counts: Dict[str, int] = {}
    subtema_counts: Dict[str, int] = {}
    tema_to_subtemas: Dict[str, Dict[str, int]] = {}
    tipo_por_tema: Dict[str, Dict[str, int]] = {}

    missing_tema = 0
    missing_subtema = 0

    for m in matches:
        md = m.get("metadata", {})
        temas_raw = _split_values(str(md.get("Tema", "") or ""))
        subtemas_raw = _split_values(str(md.get("Tema - subtema", "") or ""))
        tipo = str(md.get("Tipo", "") or "")

        temas = [_canon(t) for t in temas_raw]
        temas = [t for t in temas if t]
        subtemas = [_canon(st) for st in subtemas_raw]
        subtemas = [st for st in subtemas if st]

        if not temas:
            missing_tema += 1
        if not subtemas:
            missing_subtema += 1

        for t in temas:
            tema_counts[t] = tema_counts.get(t, 0) + 1
            if tipo:
                tipo_por_tema.setdefault(t, {})
                tipo_por_tema[t][tipo] = tipo_por_tema[t].get(tipo, 0) + 1
            if subtemas:
                tema_to_subtemas.setdefault(t, {})
                for st in subtemas:
                    tema_to_subtemas[t][st] = tema_to_subtemas[t].get(st, 0) + 1

        for st in subtemas:
            subtema_counts[st] = subtema_counts.get(st, 0) + 1

    def _top_counts(d: Dict[str, int], top_k: int) -> List[Dict]:
        total_local = sum(d.values()) or 1
        items = sorted(d.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        return [{"value": k, "count": v, "percent": round((v / total_local) * 100, 2)} for k, v in items]

    def _map_top_subtemas(tts: Dict[str, Dict[str, int]], top_k: int) -> Dict[str, List[Dict]]:
        out: Dict[str, List[Dict]] = {}
        for t, sub_map in tts.items():
            out[t] = _top_counts(sub_map, top_k)
        return out

    result = {
        "index_name": args.index_name,
        "dimension": dim,
        "total_records": total,
        "tema_stats": {
            "missing": missing_tema,
            "unique": len(tema_counts),
            "top": _top_counts(tema_counts, args.top_k)
        },
        "subtema_stats": {
            "missing": missing_subtema,
            "unique": len(subtema_counts),
            "top": _top_counts(subtema_counts, args.top_k)
        },
        "tema_to_subtemas_top": _map_top_subtemas(tema_to_subtemas, 10),
        "tipo_por_tema": {t: dict(sorted(tp.items(), key=lambda x: (-x[1], x[0]))) for t, tp in tipo_por_tema.items()},
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()