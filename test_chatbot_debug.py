#!/usr/bin/env python3
"""
Script de prueba para debuggear el chatbot
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🧪 INICIANDO PRUEBA DEL CHATBOT...")

# Probar la búsqueda primero
try:
    from models.semantic_search import search
    print("✅ Importación de semantic_search exitosa")
    
    # Probar búsqueda
    test_query = "libertad condicional"
    print(f"🔍 Probando búsqueda con: '{test_query}'")
    
    result = search({
        'text': test_query,
        'top_k': 3,
    })
    
    print(f"📊 Resultados de búsqueda: {len(result.get('matches', []))} encontrados")
    
    if result.get('matches'):
        print("✅ La búsqueda funciona correctamente")
        for i, match in enumerate(result['matches'][:2]):
            print(f"  Resultado {i+1}: Score {match.get('score', 'N/A')}")
            print(f"  Metadata keys: {list(match.get('metadata', {}).keys())}")
    else:
        print("❌ No se encontraron resultados en la búsqueda")
        
except Exception as e:
    print(f"❌ Error en búsqueda: {str(e)}")

print("\n" + "="*50)

# Probar el chatbot
try:
    from models.rag import chatbot
    print("✅ Importación de chatbot exitosa")
    
    test_question = "¿Qué dice sobre libertad condicional?"
    print(f"🤖 Probando chatbot con: '{test_question}'")
    
    response = chatbot(test_question)
    print(f"📝 Respuesta del chatbot: {response}")
    
except Exception as e:
    print(f"❌ Error en chatbot: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n🏁 PRUEBA COMPLETADA")