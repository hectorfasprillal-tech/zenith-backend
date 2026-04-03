#!/usr/bin/env python3
"""
Pruebas de búsquedas semánticas (consultas abiertas y en forma de pregunta).
No modifica Pinecone; sólo llama al buscador y muestra resultados.
"""

import sys
import os

# Permite importar modelos desde este directorio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.semantic_search import search


QUERIES = [
    # IVE / aborto
    "interrupción voluntaria de embarazo",
    "¿Qué dice la Corte sobre interrupción voluntaria del embarazo?",
    "aborto",
    # Libertad condicional
    "libertad condicional",
    "¿Qué dice sobre libertad condicional?",
    # Salud / EPS
    "derecho a la salud",
    "EPS no autoriza tratamiento",
    # Muerte digna / eutanasia
    "muerte digna",
    "¿Qué dice sobre eutanasia?",
    # Otros temas frecuentes
    "habeas corpus",
    "debido proceso",
    "igualdad y no discriminación",
    "educación universitaria",
]


def run_query(q: str, top_k: int = 10):
    try:
        res = search({'text': q, 'top_k': top_k})
        matches = res.get('matches', [])
        tutela_count = 0
        print(f"\n🔍 {q} → {len(matches)} resultados")
        for m in matches[:5]:
            md = m.get('metadata', {})
            prov = md.get('Providencia', 'N/A')
            tipo = md.get('Tipo', 'N/A')
            anio = md.get('anio', 'N/A')
            tema = md.get('Tema - subtema', 'N/A')
            if (tipo or '').lower() == 'tutela':
                tutela_count += 1
            print(f"   - {prov} | {tipo} | {anio} | Tema: {tema}")
        print(f"   ⚖️ Tutelas en top {min(5, len(matches))}: {tutela_count}")
    except Exception as e:
        print(f"❌ Error en '{q}': {e}")


def main():
    print("🧪 PRUEBAS DE BÚSQUEDA SEMÁNTICA")
    print("=" * 50)
    for q in QUERIES:
        run_query(q)
    print("\n🏁 PRUEBAS COMPLETADAS")


if __name__ == "__main__":
    main()