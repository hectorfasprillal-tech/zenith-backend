#!/usr/bin/env python3
"""
Script para verificar los tipos de datos en MongoDB
"""

import sys
import os
from pymongo import MongoClient
from collections import Counter

# Configuración de MongoDB - probando diferentes configuraciones
MONGO_CONFIGS = [
    {
        "uri": "mongodb://localhost:27017/",
        "db": "zenith",
        "collection": "judgments"
    },
    {
        "uri": "mongodb://root:MiSuperClave123@localhost:27017/?authSource=admin",
        "db": "zenith", 
        "collection": "judgments"
    },
    {
        "uri": "mongodb+srv://juancamilocristanchogomez:Zenith2024@cluster0.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
        "db": "zenith_db",
        "collection": "judgments"
    }
]

def check_mongo_tipos():
    print("🔍 VERIFICANDO TIPOS EN MONGODB")
    print("=" * 50)
    
    working_config = None
    
    # Probar diferentes configuraciones
    for i, config in enumerate(MONGO_CONFIGS, 1):
        print(f"\n📡 Probando configuración {i}: {config['uri'][:50]}...")
        
        try:
            client = MongoClient(config['uri'], serverSelectionTimeoutMS=5000)
            db = client[config['db']]
            collection = db[config['collection']]
            
            # Verificar conexión con ping
            client.admin.command('ping')
            
            # Contar documentos totales
            total_docs = collection.count_documents({})
            print(f"✅ Conectado a la base de datos: {config['db']}")
            print(f"📊 Total de documentos: {total_docs}")
            
            if total_docs > 0:
                working_config = config
                break
            else:
                print("⚠️ Conexión exitosa pero no hay documentos")
                client.close()
                
        except Exception as e:
            print(f"❌ Error con configuración {i}: {e}")
            continue
    
    if not working_config:
        print("\n❌ No se pudo conectar a ninguna configuración de MongoDB")
        return
    
    print(f"\n🎉 Usando configuración exitosa: {working_config['db']}")
    
    # Continuar con el análisis usando la configuración que funciona
    try:
        client = MongoClient(working_config['uri'])
        db = client[working_config['db']]
        collection = db[working_config['collection']]
        total_docs = collection.count_documents({})
        
        # Obtener una muestra de documentos para ver la estructura
        print("\n🔍 ESTRUCTURA DE DOCUMENTOS:")
        print("-" * 30)
        sample_doc = collection.find_one()
        if sample_doc:
            print("Campos disponibles:")
            for key in sample_doc.keys():
                print(f"  - {key}")
        
        # Buscar campos relacionados con 'tipo'
        print("\n🎯 BUSCANDO CAMPOS DE TIPO:")
        print("-" * 30)
        
        # Verificar diferentes posibles nombres de campo
        possible_tipo_fields = ['Tipo', 'tipo', 'Type', 'type', 'case_type', 'Tipo de proceso']
        
        for field in possible_tipo_fields:
            # Contar valores únicos para este campo
            pipeline = [
                {"$group": {"_id": f"${field}", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            
            try:
                results = list(collection.aggregate(pipeline))
                if results and results[0]['_id'] is not None:
                    print(f"\n✅ Campo '{field}' encontrado:")
                    for result in results[:10]:  # Mostrar top 10
                        print(f"   '{result['_id']}': {result['count']} documentos")
            except Exception as e:
                continue
        
        # Buscar documentos que contengan 'Tutela', 'Auto', 'Constitucionalidad'
        print("\n🔍 BUSCANDO PATRONES EN PROVIDENCIAS:")
        print("-" * 40)
        
        patterns = {
            'T-': 'Tutela',
            'C-': 'Constitucionalidad', 
            'A-': 'Auto',
            'SU-': 'Sentencia de Unificación'
        }
        
        for pattern, tipo_esperado in patterns.items():
            count = collection.count_documents({"Providencia": {"$regex": f"^{pattern}"}})
            print(f"   Providencias que empiezan con '{pattern}' ({tipo_esperado}): {count}")
        
        # Verificar algunos ejemplos específicos
        print("\n📋 EJEMPLOS DE DOCUMENTOS:")
        print("-" * 25)
        
        for pattern, tipo_esperado in patterns.items():
            doc = collection.find_one({"Providencia": {"$regex": f"^{pattern}"}})
            if doc:
                providencia = doc.get('Providencia', 'N/A')
                tipo_actual = doc.get('Tipo', 'N/A')
                tipo_proceso = doc.get('Tipo de proceso', 'N/A')
                print(f"   {providencia}: Tipo='{tipo_actual}', Tipo de proceso='{tipo_proceso}'")
        
        # Verificar si hay un campo que contenga los valores correctos
        print("\n🎯 VERIFICANDO VALORES ESPERADOS:")
        print("-" * 35)
        
        expected_values = ['Tutela', 'Auto', 'Constitucionalidad']
        for value in expected_values:
            count_tipo = collection.count_documents({"Tipo": value})
            count_proceso = collection.count_documents({"Tipo de proceso": {"$regex": value, "$options": "i"}})
            print(f"   '{value}' en campo 'Tipo': {count_tipo}")
            print(f"   '{value}' en campo 'Tipo de proceso': {count_proceso}")
        
        client.close()
        print("\n✅ Análisis completado")
        
    except Exception as e:
        print(f"❌ Error conectando a MongoDB: {e}")

if __name__ == "__main__":
    check_mongo_tipos()