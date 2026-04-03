#!/usr/bin/env python3
"""
Script para probar y debuggear los filtros del buscador
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.semantic_search import search

def test_filters():
    print("🔍 TESTING SEARCH FILTERS")
    print("=" * 50)
    
    # Test 1: Búsqueda sin filtros
    print("\n1️⃣ TEST: Búsqueda sin filtros")
    request_no_filters = {
        "text": "libertad condicional",
        "tipo": "",
        "anio": "",
        "fecha_inicio": "",
        "fecha_fin": "",
        "top_k": 5
    }
    
    try:
        result = search(request_no_filters)
        print(f"✅ Resultados sin filtros: {len(result['matches'])} encontrados")
        if result['matches']:
            print(f"   Primer resultado: {result['matches'][0]['metadata'].get('Providencia', 'N/A')}")
            print(f"   Tipo: {result['matches'][0]['metadata'].get('Tipo', 'N/A')}")
            print(f"   Año: {result['matches'][0]['metadata'].get('anio', 'N/A')}")
    except Exception as e:
        print(f"❌ Error en búsqueda sin filtros: {e}")
    
    # Test 2: Filtro por tipo
    print("\n2️⃣ TEST: Filtro por tipo 'Tutela'")
    request_tipo = {
        "text": "libertad condicional",
        "tipo": "Tutela",
        "anio": "",
        "fecha_inicio": "",
        "fecha_fin": "",
        "top_k": 5
    }
    
    try:
        result = search(request_tipo)
        print(f"✅ Resultados con filtro Tutela: {len(result['matches'])} encontrados")
        if result['matches']:
            for i, match in enumerate(result['matches'][:3]):
                print(f"   Resultado {i+1}: {match['metadata'].get('Providencia', 'N/A')} - Tipo: {match['metadata'].get('Tipo', 'N/A')}")
    except Exception as e:
        print(f"❌ Error en filtro por tipo: {e}")
    
    # Test 3: Filtro por año
    print("\n3️⃣ TEST: Filtro por año '2023'")
    request_anio = {
        "text": "libertad condicional",
        "tipo": "",
        "anio": "2023",
        "fecha_inicio": "",
        "fecha_fin": "",
        "top_k": 5
    }
    
    try:
        result = search(request_anio)
        print(f"✅ Resultados con filtro año 2023: {len(result['matches'])} encontrados")
        if result['matches']:
            for i, match in enumerate(result['matches'][:3]):
                print(f"   Resultado {i+1}: {match['metadata'].get('Providencia', 'N/A')} - Año: {match['metadata'].get('anio', 'N/A')}")
    except Exception as e:
        print(f"❌ Error en filtro por año: {e}")
    
    # Test 4: Filtro por fechas
    print("\n4️⃣ TEST: Filtro por fechas (2023-01-01 a 2023-06-30)")
    request_fechas = {
        "text": "libertad condicional",
        "tipo": "",
        "anio": "",
        "fecha_inicio": "2023-01-01",
        "fecha_fin": "2023-06-30",
        "top_k": 5
    }
    
    try:
        result = search(request_fechas)
        print(f"✅ Resultados con filtro de fechas: {len(result['matches'])} encontrados")
        if result['matches']:
            for i, match in enumerate(result['matches'][:3]):
                fecha_sentencia = match['metadata'].get('Fecha Sentencia', 'N/A')
                fecha_number = match['metadata'].get('fecha_number', 'N/A')
                print(f"   Resultado {i+1}: {match['metadata'].get('Providencia', 'N/A')} - Fecha: {fecha_sentencia} (Number: {fecha_number})")
    except Exception as e:
        print(f"❌ Error en filtro por fechas: {e}")
    
    # Test 5: Múltiples filtros
    print("\n5️⃣ TEST: Múltiples filtros (Tutela + 2023)")
    request_multiple = {
        "text": "libertad condicional",
        "tipo": "Tutela",
        "anio": "2023",
        "fecha_inicio": "",
        "fecha_fin": "",
        "top_k": 5
    }
    
    try:
        result = search(request_multiple)
        print(f"✅ Resultados con múltiples filtros: {len(result['matches'])} encontrados")
        if result['matches']:
            for i, match in enumerate(result['matches'][:3]):
                print(f"   Resultado {i+1}: {match['metadata'].get('Providencia', 'N/A')} - Tipo: {match['metadata'].get('Tipo', 'N/A')} - Año: {match['metadata'].get('anio', 'N/A')}")
    except Exception as e:
        print(f"❌ Error en múltiples filtros: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 TESTS COMPLETADOS")

if __name__ == "__main__":
    test_filters()