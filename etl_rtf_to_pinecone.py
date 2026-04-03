#!/usr/bin/env python3
"""
ETL directo: leer RTFs, generar resumen de relatoria y subir a Pinecone.

Fuente de datos:
- Carpeta local de sentencias RTF: por defecto `/Users/juancamilocristanchogomez/Documents/GitHub/zenith_ai/PNL_Zenith/PNL_Zenith/Data/downloaded_judgments/`

Salida:
- Vectores en Pinecone (modelo `all-MiniLM-L6-v2`, 384 dims) con metadatos:
  - `Providencia` (id del fallo, ej. `T-123-24`)
  - `Tipo` ("Tutela" | "Constitucionalidad" | "Auto" | "Sentencia de Unificación" | "otro")
  - `anio` (int)
  - `fecha_number` (int formato `YYYYMMDD` aproximado)
  - `summary_extract` (string, resumen de IA o fallback)
  - `Tema` y `Tema - subtema` (string, heurística simple)
  - `filename` y otros campos útiles para UI

Resumen de IA:
- Selección automática: usa OpenAI si `OPENAI_API_KEY` está definido; si no, usa HuggingFace API si `HUGGINGFACEHUB_API_TOKEN` está definido; si no, usa un resumen simple (primeros caracteres).

Requisitos:
- Variables de entorno: `PINECONE_API_KEY` (obligatoria)
- Opcional: `OPENAI_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`

Uso:
  python etl_rtf_to_pinecone.py \
    --data-dir "/ruta/a/downloaded_judgments" \
    --index-name "relatoria-emebeddings" \
    --batch-size 100 \
    --limit 0
"""

import os
import re
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from striprtf.striprtf import rtf_to_text

# Dependencias opcionales para resumen con APIs
try:
    from langchain_community.chat_models import ChatOpenAI
    from langchain_community.llms import HuggingFaceHub
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate
    from langchain.chains.summarize import load_summarize_chain
except Exception:
    ChatOpenAI = None
    HuggingFaceHub = None
    RecursiveCharacterTextSplitter = None
    PromptTemplate = None
    load_summarize_chain = None


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("etl_rtf_to_pinecone")


DEFAULT_DATA_DIR = \
    "/Users/juancamilocristanchogomez/Documents/GitHub/zenith_ai/PNL_Zenith/PNL_Zenith/Data/downloaded_judgments/"
DEFAULT_INDEX_NAME = "relatoria-emebeddings"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def clean_text(text: str) -> str:
    if not text:
        return ""
    # Normaliza espacios y quita caracteres no útiles
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\,\;\:\!\?\-\(\)]", " ", text)
    return text.strip()


def extract_metadata_from_filename(filename: str) -> Dict:
    base = filename.replace(".rtf", "")
    # Tipo (mapeo compatible con frontend/backend)
    if base.startswith("T-") or base.startswith("T"):
        tipo = "Tutela"
    elif base.startswith("C-") or base.startswith("C"):
        tipo = "Constitucionalidad"
    elif base.startswith("A-") or base.startswith("A"):
        tipo = "Auto"
    elif base.startswith("SU-") or base.startswith("SU"):
        tipo = "Sentencia de Unificación"
    else:
        tipo = "otro"

    # Año (dos dígitos al final del id)
    m = re.search(r"-(\d{2})$", base)
    if m:
        suf = int(m.group(1))
        anio = 2000 + suf if suf <= 30 else 1900 + suf
    else:
        anio = 2024

    # fecha_number aproximada al 1 de enero
    fecha_number = anio * 10000 + 101

    return {
        "providencia": base,
        "tipo": tipo,
        "anio": anio,
        "fecha_number": fecha_number,
        "filename": filename,
    }


def detect_tema_principal(text: str, default: str) -> str:
    temas_keywords = {
        "Derechos fundamentales": ["derecho fundamental", "derechos fundamentales", "fundamental"],
        "Debido proceso": ["debido proceso", "proceso debido"],
        "Igualdad": ["igualdad", "discriminación", "equidad"],
        "Libertad": ["libertad", "libre desarrollo"],
        "Tutela": ["acción de tutela", "tutela"],
        "Habeas corpus": ["habeas corpus", "libertad personal"],
        "Salud": ["derecho a la salud", "salud", "eps"],
        "Educación": ["derecho a la educación", "educación"],
        "Trabajo": ["derecho al trabajo", "trabajo", "laboral"],
    }
    found = []
    tl = text.lower()
    for tema, kws in temas_keywords.items():
        if any(k in tl for k in kws):
            found.append(tema)
    return found[0] if found else default


def split_docs_for_summary(text: str, model_name: str = "openai"):
    if not RecursiveCharacterTextSplitter:
        return [text]
    mapper_chunk_size = {"openai": 10000, "huggingface_api": 5000}
    chunk = mapper_chunk_size.get(model_name, 5000)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n"],
        chunk_size=chunk,
        chunk_overlap=int(chunk * 0.05),
        add_start_index=True,
    )
    return splitter.create_documents([text])


def summarize_with_openai(text: str) -> Optional[str]:
    if not ChatOpenAI or not load_summarize_chain or os.getenv("OPENAI_API_KEY") is None:
        return None
    try:
        llm = ChatOpenAI(temperature=0)
        docs = split_docs_for_summary(text, model_name="openai")
        map_prompt = """Your final answer must be in Spanish.\nWrite a concise summary of the following:\n{text}\nRESUMEN CONCISO:"""
        combine_prompt = """The answer must be in Spanish.\nWrite a concise summary of the following:\n{text}\nReturn your answer which cover the key points of the text.\nSUMMARY IN SPANISH WITH NUMERALS:"""
        prompt_map = PromptTemplate.from_template(map_prompt)
        prompt_combine = PromptTemplate.from_template(combine_prompt)
        chain = load_summarize_chain(llm=llm, chain_type="map_reduce", map_prompt=prompt_map, combine_prompt=prompt_combine, verbose=False)
        return chain.run(docs).strip()
    except Exception as e:
        logger.warning(f"OpenAI summarization failed: {e}")
        return None


def summarize_with_hf_api(text: str, model_name: str = "HuggingFaceH4/zephyr-7b-alpha") -> Optional[str]:
    if not HuggingFaceHub or not load_summarize_chain or os.getenv("HUGGINGFACEHUB_API_TOKEN") is None:
        return None
    try:
        llm = HuggingFaceHub(task="text-generation", repo_id=model_name, verbose=False, model_kwargs={"temperature": 0.001, "max_new_tokens": 600})
        docs = split_docs_for_summary(text, model_name="huggingface_api")
        map_prompt = """Your final answer must be in Spanish.\nWrite a concise summary of the following:\n{text}\nRESUMEN CONCISO:"""
        combine_prompt = """The answer must be in Spanish.\nWrite a concise summary of the following:\n{text}\nReturn your answer which cover the key points of the text.\nSUMMARY IN SPANISH WITH NUMERALS:"""
        prompt_map = PromptTemplate.from_template(map_prompt)
        prompt_combine = PromptTemplate.from_template(combine_prompt)
        chain = load_summarize_chain(llm=llm, chain_type="map_reduce", map_prompt=prompt_map, combine_prompt=prompt_combine, verbose=False)
        return chain.run(docs).strip()
    except Exception as e:
        logger.warning(f"HF API summarization failed: {e}")
        return None


def generate_summary(text: str) -> str:
    # Intenta OpenAI, luego HF API, luego fallback simple
    summary = summarize_with_openai(text)
    if not summary:
        summary = summarize_with_hf_api(text)
    if not summary:
        # Fallback: recorte simple
        summary = (text[:800] + "...") if len(text) > 800 else text
    return summary.strip()


def upload_batch_with_retry(index, vectors_batch, max_retries: int = 3) -> bool:
    for attempt in range(max_retries):
        try:
            index.upsert(vectors_batch)
            return True
        except Exception as e:
            logger.warning(f"Upsert intento {attempt + 1} falló: {e}")
            if attempt == max_retries - 1:
                logger.error("Fallo definitivo al subir lote")
                return False
            time.sleep(2 ** attempt)
    return False


def process_rtf_directory(data_dir: str, index_name: str, batch_size: int = 100, limit: int = 0, order: str = "asc") -> bool:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        logger.error("PINECONE_API_KEY no configurado en entorno")
        return False

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    folder = Path(data_dir)
    if not folder.exists():
        logger.error(f"Directorio no existe: {data_dir}")
        return False

    rtf_files = sorted([p for p in folder.glob("*.rtf")])
    # Permite procesar los últimos primero (desc)
    if order.lower() == "desc":
        rtf_files = list(reversed(rtf_files))
    total = len(rtf_files)
    if total == 0:
        logger.warning("No se encontraron archivos RTF")
        return False

    if limit and limit > 0:
        rtf_files = rtf_files[:limit]
    logger.info(f"Procesando {len(rtf_files)}/{total} archivos RTF")

    batch: List[Tuple] = []
    processed = 0
    errors = 0

    for i, rtf_path in enumerate(rtf_files, 1):
        try:
            logger.info(f"[{i}/{len(rtf_files)}] {rtf_path.name}")
            content = rtf_path.read_text(encoding="utf-8", errors="ignore")
            raw_text = clean_text(rtf_to_text(content))
            if not raw_text or len(raw_text) < 100:
                logger.warning(f"Texto insuficiente en {rtf_path.name}")
                continue

            meta_base = extract_metadata_from_filename(rtf_path.name)
            tema_principal = detect_tema_principal(raw_text, default=meta_base["tipo"])  # usa tipo como default
            summary = generate_summary(raw_text)

            # Texto para embedding (combina resumen y metadata para mejor señal semántica)
            text_to_embed = f"""
            {summary}\nTipo: {meta_base['tipo']}\nTema: {tema_principal}\nAño: {meta_base['anio']}
            """.strip()
            vector = model.encode(text_to_embed).tolist()

            metadata = {
                "Providencia": meta_base["providencia"],
                "Información": summary[:500],
                "Demandante": "Corte Constitucional",
                "Tema": tema_principal,
                "Tema - subtema": tema_principal,
                "Resumen": summary[:300] + ("..." if len(summary) > 300 else ""),
                "summary_extract": summary,
                "Tipo": meta_base["tipo"],
                "Fecha Sentencia": f"{meta_base['anio']}-01-01",
                "anio": meta_base["anio"],
                "fecha_number": meta_base["fecha_number"],
                "filename": meta_base["filename"],
            }

            batch.append((meta_base["providencia"], vector, metadata))

            if len(batch) >= batch_size:
                ok = upload_batch_with_retry(index, batch)
                if ok:
                    processed += len(batch)
                    logger.info(f"Subido lote. Procesados: {processed}")
                else:
                    errors += len(batch)
                batch = []

        except Exception as e:
            logger.error(f"Error procesando {rtf_path.name}: {e}")
            errors += 1

    if batch:
        ok = upload_batch_with_retry(index, batch)
        if ok:
            processed += len(batch)
        else:
            errors += len(batch)

    stats = index.describe_index_stats()
    logger.info(f"Finalizado. Exitosos: {processed}, Errores: {errors}, Total en índice: {stats.get('total_vector_count')}")
    return processed > 0


def main():
    parser = argparse.ArgumentParser(description="ETL de RTFs a Pinecone con resumen de relatoria")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Directorio de RTFs")
    parser.add_argument("--index-name", type=str, default=DEFAULT_INDEX_NAME, help="Nombre del índice de Pinecone")
    parser.add_argument("--batch-size", type=int, default=100, help="Tamaño del lote de upsert")
    parser.add_argument("--limit", type=int, default=0, help="Límite de archivos a procesar (0 = todos)")
    parser.add_argument("--order", type=str, default="asc", choices=["asc", "desc"], help="Orden de procesamiento de archivos (asc|desc)")

    args = parser.parse_args()

    success = process_rtf_directory(
        data_dir=args.data_dir,
        index_name=args.index_name,
        batch_size=args.batch_size,
        limit=args.limit,
        order=args.order,
    )
    if success:
        logger.info("Carga completada exitosamente")
    else:
        logger.error("La carga no procesó elementos")


if __name__ == "__main__":
    main()