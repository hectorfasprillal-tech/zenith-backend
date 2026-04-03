#!/usr/bin/env python3
"""
Script para corregir los tipos de casos en MongoDB
"""

import re
from pymongo import MongoClient
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración de MongoDB
MONGO_URI = 'mongodb://admin:zenith123@localhost:27017/?authSource=admin'
DB_NAME = 'zenith'
COLLECTION_NAME = 'judgments'

def extract_correct_case_type(id_judgment):
    """Extrae el tipo correcto basado en el ID del juicio"""
    if not id_judgment or id_judgment == 'INIT-000':
        return None
        
    if id_judgment.startswith('T'):
        return 'tutela'
    elif id_judgment.startswith('C'):
        return 'constitucionalidad'
    elif id_judgment.startswith('A'):
        return 'auto'
    elif id_judgment.startswith('SU'):
        return 'sentencia_unificacion'
    else:
        return 'otro'

def fix_case_types():
    """Corrige los tipos de casos en MongoDB"""
    try:
        # Conectar a MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        logger.info("🔧 Iniciando corrección de tipos de casos...")
        
        # Obtener todos los documentos
        documents = collection.find({})
        
        updated_count = 0
        for doc in documents:
            id_judgment = doc.get('id_judgment')
            current_type = doc.get('case_type')
            correct_type = extract_correct_case_type(id_judgment)
            
            if correct_type and current_type != correct_type:
                # Actualizar el documento
                result = collection.update_one(
                    {'_id': doc['_id']},
                    {'$set': {'case_type': correct_type}}
                )
                
                if result.modified_count > 0:
                    logger.info(f"✅ Actualizado {id_judgment}: {current_type} → {correct_type}")
                    updated_count += 1
        
        logger.info(f"🎉 Corrección completada. {updated_count} documentos actualizados.")
        
        # Mostrar estadísticas finales
        tipos = collection.aggregate([
            {'$group': {'_id': '$case_type', 'count': {'$sum': 1}}}
        ])
        
        logger.info("📊 Distribución final por tipos:")
        for tipo in tipos:
            logger.info(f"   - {tipo['_id']}: {tipo['count']} documentos")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    success = fix_case_types()
    if success:
        logger.info("🎉 Corrección de tipos completada exitosamente")
    else:
        logger.error("💥 Corrección de tipos falló")