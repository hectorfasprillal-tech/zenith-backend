#!/usr/bin/env python3
"""
Script para investigar qué valores de 'Tipo' existen realmente en Pinecone
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.semantic_search import search

def investigate_tipos():
    print("🔍 INVESTIGANDO VALORES DE 'TIPO' EN PINECONE")
    print("=" * 60)
    
    # Hacer una búsqueda amplia para obtener una muestra de datos
    request = {
        "text": "constitucional",  # Término amplio
        "tipo": "",
        "anio": "",
        "fecha_inicio": "",
        "fecha_fin": "",
        "top_k": 50  # Obtener más resultados para ver variedad
    }
    
    try:
        result = search(request)
        print(f"✅ Obtenidos {len(result['matches'])} resultados para análisis")
        
        # Recopilar todos los valores únicos de 'Tipo'
        tipos_encontrados = set()
        ejemplos_por_tipo = {}
        
        for match in result['matches']:
            metadata = match['metadata']
            tipo = metadata.get('Tipo', 'N/A')
            tipos_encontrados.add(tipo)
            
            # Guardar un ejemplo de cada tipo
            if tipo not in ejemplos_por_tipo:
                ejemplos_por_tipo[tipo] = {
                    'providencia': metadata.get('Providencia', 'N/A'),
                    'anio': metadata.get('anio', 'N/A'),
                    'fecha': metadata.get('Fecha Sentencia', 'N/A')
                }
        
        print(f"\n📊 VALORES ÚNICOS DE 'TIPO' ENCONTRADOS: {len(tipos_encontrados)}")
        print("-" * 40)
        
        for i, tipo in enumerate(sorted(tipos_encontrados), 1):
            ejemplo = ejemplos_por_tipo[tipo]
            print(f"{i:2d}. '{tipo}'")
            print(f"    Ejemplo: {ejemplo['providencia']} ({ejemplo['anio']}) - {ejemplo['fecha']}")
        
        print("\n" + "=" * 60)
        print("🎯 COMPARACIÓN CON OPCIONES DEL FRONTEND:")
        print("-" * 40)
        
        frontend_options = ["Constitucionalidad", "Auto", "Tutela"]
        
        for option in frontend_options:
            if option in tipos_encontrados:
                print(f"✅ '{option}' - EXISTE en los datos")
            else:
                print(f"❌ '{option}' - NO EXISTE en los datos")
                # Buscar similares
                similares = [t for t in tipos_encontrados if option.lower() in t.lower() or t.lower() in option.lower()]
                if similares:
                    print(f"   🔍 Posibles similares: {similares}")
        
        print(f"\n💡 RECOMENDACIÓN:")
        print("   El frontend debe usar los valores exactos que existen en Pinecone.")
        print("   Considera actualizar las opciones del dropdown en el frontend.")
        
    except Exception as e:
        print(f"❌ Error en la investigación: {e}")

if __name__ == "__main__":
    investigate_tipos()