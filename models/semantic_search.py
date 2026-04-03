import os
import re
import unicodedata
from typing import List, Dict, Any
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

_ready = False
try:
    _api_key = os.getenv('PINECONE_API_KEY') or 'pcsk_4T7tGp_NhkTJjhUG6459Ekg2MhZwKbY1wtGGAFuy8H7FeCc8ScWV2fLAzWmaMJ8vm4viM9'
    pc = Pinecone(api_key=_api_key)
    index = pc.Index("relatoria-emebeddings")
    _ready = True
except Exception as e:
    print(f"⚠️ SEMANTIC_SEARCH WARN: Pinecone no inicializado: {str(e)}")
    pc = None
    index = None
    _ready = False

def is_ready():
    return _ready and index is not None

def _strip_accents(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def _normalize(text: str) -> str:
    # Minúsculas + remover tildes + colapsar espacios
    t = _strip_accents(text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _expand_query(text: str) -> List[str]:
    """Expansión ligera de consulta con sinónimos legales comunes.
    No modifica Pinecone; sólo crea variantes de texto para la query.
    """
    base = text.strip()
    norm = _normalize(base)

    expansions: List[str] = [base]

    synonyms: Dict[str, List[str]] = {
        # IVE / aborto
        "interrupcion voluntaria de embarazo": ["aborto", "ive", "interrupcion del embarazo", "terminacion voluntaria del embarazo", "salud sexual y reproductiva", "derechos reproductivos"],
        "interrupcion voluntaria del embarazo": ["aborto", "ive", "interrupcion de embarazo", "terminacion voluntaria del embarazo", "salud sexual y reproductiva", "derechos reproductivos"],
        "aborto": ["interrupcion voluntaria de embarazo", "ive", "terminacion voluntaria del embarazo", "salud sexual y reproductiva", "derechos reproductivos"],
        "ive": ["aborto", "interrupcion voluntaria de embarazo", "salud sexual y reproductiva", "derechos reproductivos"],
        # Libertad condicional
        "libertad condicional": ["beneficio de libertad", "sustitucion de la pena", "redencion de pena"],
        # Tutela
        "accion de tutela": ["tutela"],
        "tutela": ["accion de tutela"],
        # Salud
        "derecho a la salud": ["eps", "servicios de salud", "plan obligatorio de salud", "pos", "acceso a la salud"],
        "eps": ["derecho a la salud", "servicios de salud"],
        # Debido proceso
        "debido proceso": ["garantias procesales", "derecho de defensa", "imparcialidad"],
        # Habeas Corpus
        "habeas corpus": ["libertad personal"],
        # Igualdad
        "igualdad": ["no discriminacion", "equidad"],
        # Educacion
        "educacion": ["acceso a la educacion", "matricula", "colegio", "universidad"],
        # Muerte digna
        "muerte digna": ["eutanasia", "morir dignamente", "voluntad anticipada"],
    }

    # Detectar claves presentes en el texto normalizado
    matched_keys = []
    for key in synonyms.keys():
        if key in norm:
            matched_keys.append(key)

    # Construir texto reforzado con sinónimos
    boosted_terms: List[str] = []
    for k in matched_keys:
        boosted_terms.extend(synonyms.get(k, []))

    if boosted_terms:
        # Variante con los sinónimos concatenados para sesgo semántico
        expansions.append(f"{base} " + " ".join(sorted(set(boosted_terms))))
        # Variantes cortas con cada sinónimo (máximo 3 para limitar consultas)
        for term in sorted(set(boosted_terms))[:3]:
            expansions.append(term)

    # Devolver únicas manteniendo orden
    seen = set()
    unique_expansions = []
    for e in expansions:
        if e not in seen:
            seen.add(e)
            unique_expansions.append(e)
    return unique_expansions

def _build_filter(payload: Dict[str, Any]):
    flt = []
    if payload.get('tipo'):
        flt.append({'Tipo': {'$eq': payload['tipo']}})
    if payload.get('anio'):
        try:
            flt.append({'anio': {'$eq': int(payload['anio'])}})
        except Exception:
            pass
    if payload.get('fecha_inicio'):
        try:
            flt.append({'fecha_number': {'$gte': int(str(payload['fecha_inicio']).replace('-', ''))}})
        except Exception:
            pass
    if payload.get('fecha_fin'):
        try:
            flt.append({'fecha_number': {'$lte': int(str(payload['fecha_fin']).replace('-', ''))}})
        except Exception:
            pass
    if not flt:
        return None
    return {'$and': flt}

def search(json):
    if not is_ready():
        raise RuntimeError("Pinecone no inicializado o índice inaccesible.")

    text = json.get('text', '') or ''
    expansions = _expand_query(text)
    flt = _build_filter(json)
    print('filtros: ', flt)

    # top_k más generoso por defecto para mejorar recall
    top_k = int(json.get('top_k', 20))

    # Ejecutar una o varias consultas y fusionar resultados sin duplicar IDs
    combined: Dict[str, Any] = {}
    for i, qtext in enumerate(expansions[:4]):  # límite duro de 4 consultas
        qvec = model.encode(qtext).tolist()
        try:
            res = index.query(vector=qvec, top_k=top_k, include_metadata=True, filter=flt)
        except Exception as e:
            print(f"⚠️ SEMANTIC_SEARCH WARN: Error en consulta {i}: {e}")
            continue
        matches = res.get('matches') if isinstance(res, dict) else getattr(res, 'matches', [])
        for m in matches or []:
            mid = m.get('id') if isinstance(m, dict) else getattr(m, 'id', None)
            if not mid:
                # Conservador: si no hay id, sólo añadimos
                combined[str(id(m))] = m
                continue
            # Mantener el de mayor score si se repite
            prev = combined.get(mid)
            if prev is None:
                combined[mid] = m
            else:
                prev_score = prev.get('score') if isinstance(prev, dict) else getattr(prev, 'score', 0.0)
                new_score = m.get('score') if isinstance(m, dict) else getattr(m, 'score', 0.0)
                if new_score > float(prev_score or 0.0):
                    combined[mid] = m

    # Ordenar por score descendente y recortar al top_k
    combined_list = list(combined.values())
    combined_list.sort(key=lambda m: (m.get('score') if isinstance(m, dict) else getattr(m, 'score', 0.0)), reverse=True)
    combined_list = combined_list[:top_k]

    # Devolver estructura compatible con el servidor y tests
    return { 'matches': combined_list }
