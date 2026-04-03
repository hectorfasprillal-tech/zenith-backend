#!/usr/bin/env python3
"""
Script para sincronizar datos de MongoDB a Pinecone
Este script implementa el puente faltante entre MongoDB y Pinecone
Optimizado para manejar grandes volúmenes de datos (7GB+)
"""

import os
import json
from pymongo import MongoClient
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import logging
import time

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración
PINECONE_API_KEY = 'pcsk_4T7tGp_NhkTJjhUG6459Ekg2MhZwKbY1wtGGAFuy8H7FeCc8ScWV2fLAzWmaMJ8vm4viM9'
PINECONE_INDEX = "relatoria-emebeddings"

# Configuraciones de MongoDB para probar
MONGO_CONFIGS = [
    {
        'uri': 'mongodb://localhost:27017/',
        'db': 'zenith',
        'description': 'Local sin credenciales'
    },
    {
        'uri': 'mongodb://admin:zenith123@localhost:27017/?authSource=admin',
        'db': 'zenith',
        'description': 'Local con credenciales admin'
    },
    {
        'uri': 'mongodb://root:MiSuperClave123@localhost:27017/?authSource=admin',
        'db': 'zenith', 
        'description': 'Local con credenciales root'
    }
]

def test_mongo_connection():
    """Prueba diferentes configuraciones de MongoDB"""
    for config in MONGO_CONFIGS:
        try:
            logger.info(f"Probando conexión: {config['description']}")
            client = MongoClient(config['uri'], serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            db = client[config['db']]
            collections = db.list_collection_names()
            logger.info(f"✅ Conexión exitosa. Colecciones: {collections}")
            return client, db
        except Exception as e:
            logger.warning(f"❌ Falló {config['description']}: {e}")
    
    return None, None

def extract_case_type_from_mongo(document):
    """Extrae el tipo de caso desde MongoDB"""
    # Usar el case_type que ya está en MongoDB
    return document.get('case_type', 'otro')

def upload_batch_to_pinecone(index, vectors_batch):
    """Sube un lote de vectores a Pinecone con reintentos"""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            index.upsert(vectors_batch)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Intento {attempt + 1} falló: {e}")
            if attempt == max_retries - 1:
                logger.error(f"❌ Falló después de {max_retries} intentos")
                return False
            time.sleep(2 ** attempt)  # Backoff exponencial
    
    return False

def sync_mongo_to_pinecone(limit=None, batch_size=100):
    """Sincroniza datos de MongoDB a Pinecone con procesamiento por lotes"""
    logger.info("🔄 Iniciando sincronización MongoDB → Pinecone")
    
    # Conectar a MongoDB
    mongo_client, db = test_mongo_connection()
    if not mongo_client:
        logger.error("❌ No se pudo conectar a MongoDB")
        return False
    
    # Conectar a Pinecone
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("✅ Conectado a Pinecone")
    except Exception as e:
        logger.error(f"❌ Error conectando a Pinecone: {e}")
        return False
    
    # Buscar colección de documentos
    collections = db.list_collection_names()
    target_collection = None
    
    for coll_name in ['judgments', 'sentencias', 'documents']:
        if coll_name in collections:
            target_collection = db[coll_name]
            logger.info(f"📁 Usando colección: {coll_name}")
            break
    
    if not target_collection:
        logger.error(f"❌ No se encontró colección de documentos. Disponibles: {collections}")
        return False
    
    # Contar documentos totales
    total_docs = target_collection.count_documents({})
    logger.info(f"📄 Total documentos en MongoDB: {total_docs}")
    
    # Aplicar límite si se especifica
    if limit:
        process_count = min(limit, total_docs)
        logger.info(f"🎯 Procesando {process_count} documentos (límite aplicado)")
    else:
        process_count = total_docs
        logger.info(f"🎯 Procesando todos los {process_count} documentos")
    
    # Procesar en lotes
    processed_count = 0
    error_count = 0
    
    # Cursor para procesar documentos
    cursor = target_collection.find().limit(process_count) if limit else target_collection.find()
    
    vectors_batch = []
    
    for doc in cursor:
        try:
            # Extraer información
            judgment_id = doc.get('id_judgment', str(doc.get('_id')))
            raw_text = doc.get('raw_text', '')
            
            if not raw_text or len(raw_text) < 100:
                logger.warning(f"⚠️ Texto insuficiente para {judgment_id}")
                continue
            
            # Verificar si ya existe en Pinecone
            try:
                existing = index.fetch(ids=[judgment_id])
                if existing['vectors']:
                    logger.info(f"⏭️ Ya existe en Pinecone: {judgment_id}")
                    continue
            except:
                pass  # Si no existe, continuar
            
            # Generar embedding (usar más texto para mejor calidad)
            text_for_embedding = raw_text[:2000]  # Usar más texto
            vector = model.encode(text_for_embedding).tolist()
            
            # Extraer tipo correcto desde MongoDB
            case_type = doc.get('case_type', 'otro')
            
            # Preparar metadata completa
            metadata = {
                'Providencia': judgment_id,
                'Información': raw_text[:500],
                'Tipo': case_type,
                'anio': doc.get('year', 2024),
                'fecha_number': int(str(doc.get('year', 2024)) + '0101'),
                'Demandante': doc.get('demandante', ''),
                'Tema': case_type,
                'Resumen': raw_text[:300],
                'text_length': doc.get('text_length', len(raw_text)),
                'filename': doc.get('filename', '')
            }
            
            vectors_batch.append((judgment_id, vector, metadata))
            logger.info(f"✅ Preparado: {judgment_id} - Tipo: {case_type}")
            
            # Subir lote cuando alcance el tamaño
            if len(vectors_batch) >= batch_size:
                success = upload_batch_to_pinecone(index, vectors_batch)
                if success:
                    processed_count += len(vectors_batch)
                    logger.info(f"🚀 Lote subido: {processed_count}/{process_count}")
                else:
                    error_count += len(vectors_batch)
                vectors_batch = []
            
        except Exception as e:
            logger.error(f"❌ Error procesando documento {judgment_id}: {e}")
            error_count += 1
    
    # Subir último lote si queda algo
    if vectors_batch:
        success = upload_batch_to_pinecone(index, vectors_batch)
        if success:
            processed_count += len(vectors_batch)
        else:
            error_count += len(vectors_batch)
    
    # Estadísticas finales
    stats = index.describe_index_stats()
    logger.info(f"📊 Sincronización completada:")
    logger.info(f"   - Procesados exitosamente: {processed_count}")
    logger.info(f"   - Errores: {error_count}")
    logger.info(f"   - Total vectores en Pinecone: {stats['total_vector_count']}")
    
    return processed_count > 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sincronizar MongoDB a Pinecone')
    parser.add_argument('--limit', type=int, help='Límite de documentos a procesar')
    parser.add_argument('--batch-size', type=int, default=50, help='Tamaño del lote')
    
    args = parser.parse_args()
    
    success = sync_mongo_to_pinecone(limit=args.limit, batch_size=args.batch_size)
    if success:
        logger.info("🎉 Sincronización completada exitosamente")
    else:
        logger.error("💥 Sincronización falló")