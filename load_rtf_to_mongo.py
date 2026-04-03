#!/usr/bin/env python3
"""
Script para cargar archivos RTF a MongoDB
Este es el primer paso del flujo correcto: RTF → MongoDB → Pinecone
"""

import os
import json
import re
from pathlib import Path
from pymongo import MongoClient
from striprtf.striprtf import rtf_to_text
import logging
from datetime import datetime

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Ruta de los archivos RTF
RTF_FOLDER = "/Users/tuka/Documents/GitHub/zenith_ai/PNL_Zenith/PNL_Zenith/Data/downloaded_judgments"

def test_mongo_connection():
    """Prueba diferentes configuraciones de MongoDB"""
    for config in MONGO_CONFIGS:
        try:
            logger.info(f"Probando conexión: {config['description']}")
            client = MongoClient(config['uri'], serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            db = client[config['db']]
            
            # Probar operaciones reales de base de datos
            collection = db['judgments']
            collection.count_documents({})  # Esta operación requiere autenticación
            
            logger.info(f"✅ Conexión exitosa con {config['description']}")
            return client, db, config
        except Exception as e:
            logger.warning(f"❌ Falló {config['description']}: {e}")
    
    return None, None, None

def extract_judgment_info(filename):
    """Extrae información del nombre del archivo RTF"""
    # Remover extensión
    base_name = filename.replace('.rtf', '')
    
    # Extraer tipo basado en prefijo
    if base_name.startswith('T'):
        case_type = 'tutela'
    elif base_name.startswith('C'):
        case_type = 'constitucionalidad'
    elif base_name.startswith('A'):
        case_type = 'auto'
    elif base_name.startswith('SU'):
        case_type = 'sentencia_unificacion'
    else:
        case_type = 'otro'
    
    # Extraer año del nombre del archivo
    year_match = re.search(r'-(\d{2})$', base_name)
    if year_match:
        year_suffix = int(year_match.group(1))
        # Convertir año de 2 dígitos a 4 dígitos
        if year_suffix >= 90:
            year = 1900 + year_suffix
        else:
            year = 2000 + year_suffix
    else:
        year = 2024  # Año por defecto
    
    return {
        'id_judgment': base_name,
        'case_type': case_type,
        'year': year,
        'filename': filename
    }

def clean_rtf_text(raw_text):
    """Limpia el texto extraído del RTF"""
    if not raw_text:
        return ""
    
    # Remover caracteres de control y espacios excesivos
    cleaned = re.sub(r'\s+', ' ', raw_text)
    cleaned = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'áéíóúñüÁÉÍÓÚÑÜ]', ' ', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned

def load_rtf_to_mongo():
    """Carga archivos RTF a MongoDB"""
    logger.info("🏛️ Iniciando carga de archivos RTF a MongoDB")
    
    # Conectar a MongoDB
    mongo_client, db, config = test_mongo_connection()
    if not mongo_client:
        logger.error("❌ No se pudo conectar a MongoDB")
        return False
    
    logger.info(f"📡 Usando configuración: {config['description']}")
    
    # Crear colección
    collection = db['judgments']
    logger.info(f"📁 Usando colección: judgments")
    
    # Verificar directorio RTF
    if not os.path.exists(RTF_FOLDER):
        logger.error(f"❌ Directorio RTF no existe: {RTF_FOLDER}")
        return False
    
    # Obtener archivos RTF
    rtf_files = [f for f in os.listdir(RTF_FOLDER) if f.endswith('.rtf')]
    logger.info(f"📄 Encontrados {len(rtf_files)} archivos RTF")
    
    if not rtf_files:
        logger.warning("⚠️ No se encontraron archivos RTF")
        return False
    
    # Procesar archivos (limitar para prueba inicial)
    processed_count = 0
    error_count = 0
    # Procesar todos los archivos RTF
    
    for i, filename in enumerate(rtf_files):
        try:
            logger.info(f"📄 Procesando {i+1}/{len(rtf_files)}: {filename}")
            
            # Extraer información del archivo
            judgment_info = extract_judgment_info(filename)
            
            # Verificar si ya existe
            existing = collection.find_one({'id_judgment': judgment_info['id_judgment']})
            if existing:
                logger.info(f"⏭️ Ya existe: {judgment_info['id_judgment']}")
                continue
            
            # Leer archivo RTF
            file_path = os.path.join(RTF_FOLDER, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                rtf_content = f.read()
            
            # Convertir RTF a texto
            try:
                raw_text = rtf_to_text(rtf_content)
                cleaned_text = clean_rtf_text(raw_text)
            except Exception as e:
                logger.warning(f"⚠️ Error convirtiendo RTF {filename}: {e}")
                cleaned_text = rtf_content  # Usar contenido crudo si falla
            
            if not cleaned_text or len(cleaned_text) < 100:
                logger.warning(f"⚠️ Texto muy corto en {filename}")
                continue
            
            # Preparar documento para MongoDB
            document = {
                'id_judgment': judgment_info['id_judgment'],
                'filename': judgment_info['filename'],
                'case_type': judgment_info['case_type'],
                'year': judgment_info['year'],
                'raw_text': cleaned_text,
                'text_length': len(cleaned_text),
                'processed_date': datetime.now(),
                'status': 'loaded'
            }
            
            # Insertar en MongoDB
            result = collection.insert_one(document)
            logger.info(f"✅ Insertado: {judgment_info['id_judgment']} - Tipo: {judgment_info['case_type']}")
            processed_count += 1
            
        except Exception as e:
            logger.error(f"❌ Error procesando {filename}: {e}")
            error_count += 1
    
    # Estadísticas finales
    total_docs = collection.count_documents({})
    logger.info(f"📊 Estadísticas finales:")
    logger.info(f"   - Procesados en esta sesión: {processed_count}")
    logger.info(f"   - Errores: {error_count}")
    logger.info(f"   - Total documentos en MongoDB: {total_docs}")
    
    # Verificar tipos de documentos
    pipeline = [
        {"$group": {"_id": "$case_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    types_stats = list(collection.aggregate(pipeline))
    logger.info(f"📈 Distribución por tipos:")
    for stat in types_stats:
        logger.info(f"   - {stat['_id']}: {stat['count']} documentos")
    
    return processed_count > 0

if __name__ == "__main__":
    success = load_rtf_to_mongo()
    if success:
        logger.info("🎉 Carga a MongoDB completada exitosamente")
    else:
        logger.error("💥 Carga a MongoDB falló")