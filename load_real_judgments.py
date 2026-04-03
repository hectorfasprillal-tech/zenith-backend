import os
import json
import re
from pathlib import Path
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from striprtf.striprtf import rtf_to_text
import uuid

# Configuración
API_KEY = 'pcsk_4T7tGp_NhkTJjhUG6459Ekg2MhZwKbY1wtGGAFuy8H7FeCc8ScWV2fLAzWmaMJ8vm4viM9'
INDEX_NAME = "relatoria-emebeddings"

# Rutas
RTF_FOLDER = "/Users/tuka/Documents/GitHub/zenith_ai/PNL_Zenith/PNL_Zenith/Data/downloaded_judgments"

# Inicializar Pinecone y modelo
pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("🏛️ Cargando sentencias reales de la Corte Constitucional...")

def clean_text(text):
    """Limpia y normaliza el texto"""
    if not text:
        return ""
    
    # Remover caracteres especiales y normalizar espacios
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
    return text.strip()

def extract_judgment_info(rtf_content, filename):
    """Extrae información clave de la sentencia"""
    try:
        # Convertir RTF a texto plano
        text = rtf_to_text(rtf_content)
        text = clean_text(text)
        
        # Extraer información básica del nombre del archivo
        judgment_id = filename.replace('.rtf', '')
        
        # Extraer año del nombre del archivo (últimos 2 dígitos)
        year_match = re.search(r'-(\d{2})$', judgment_id)
        if year_match:
            year_suffix = year_match.group(1)
            # Convertir a año completo (asumiendo 1900-2099)
            year = 2000 + int(year_suffix) if int(year_suffix) <= 30 else 1900 + int(year_suffix)
        else:
            year = 2023  # Año por defecto
        
        # Extraer tipo de sentencia del ID - usando valores que coincidan con el frontend
        if judgment_id.startswith('T-'):
            case_type = 'Tutela'
        elif judgment_id.startswith('C-'):
            case_type = 'Constitucionalidad'
        elif judgment_id.startswith('A-'):
            case_type = 'Auto'
        elif judgment_id.startswith('SU-'):
            case_type = 'Sentencia de Unificación'
        else:
            case_type = 'otro'
        
        # Extraer resumen (primeros 500 caracteres del texto)
        summary = text[:500] + "..." if len(text) > 500 else text
        
        # Buscar temas clave en el texto
        temas_keywords = {
            'derechos fundamentales': ['derecho fundamental', 'derechos fundamentales', 'fundamental'],
            'debido proceso': ['debido proceso', 'proceso debido'],
            'igualdad': ['igualdad', 'discriminación', 'equidad'],
            'libertad': ['libertad', 'libre desarrollo'],
            'tutela': ['acción de tutela', 'tutela'],
            'habeas corpus': ['habeas corpus', 'libertad personal'],
            'salud': ['derecho a la salud', 'salud', 'eps'],
            'educación': ['derecho a la educación', 'educación'],
            'trabajo': ['derecho al trabajo', 'trabajo', 'laboral']
        }
        
        temas_encontrados = []
        text_lower = text.lower()
        for tema, keywords in temas_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                temas_encontrados.append(tema)
        
        tema_principal = temas_encontrados[0] if temas_encontrados else case_type
        
        return {
            'id': judgment_id,
            'year': year,
            'case_type': case_type,
            'summary': summary,
            'full_text': text,
            'tema_principal': tema_principal,
            'temas': temas_encontrados,
            'filename': filename
        }
        
    except Exception as e:
        print(f"❌ Error procesando {filename}: {e}")
        return None

def load_judgments_from_rtf():
    """Carga todas las sentencias RTF disponibles"""
    rtf_folder = Path(RTF_FOLDER)
    
    if not rtf_folder.exists():
        print(f"❌ Carpeta no encontrada: {RTF_FOLDER}")
        return []
    
    rtf_files = list(rtf_folder.glob("*.rtf"))
    print(f"📁 Encontrados {len(rtf_files)} archivos RTF")
    
    judgments = []
    
    for i, rtf_file in enumerate(rtf_files, 1):  # Procesar TODAS las sentencias
        print(f"📄 Procesando {i}/{len(rtf_files)}: {rtf_file.name}")
        
        try:
            with open(rtf_file, 'r', encoding='utf-8', errors='ignore') as f:
                rtf_content = f.read()
            
            judgment_info = extract_judgment_info(rtf_content, rtf_file.name)
            if judgment_info:
                judgments.append(judgment_info)
                
        except Exception as e:
            print(f"❌ Error leyendo {rtf_file.name}: {e}")
            continue
    
    print(f"✅ Procesadas {len(judgments)} sentencias exitosamente")
    return judgments

# Limpiar índice existente
print("🧹 Limpiando datos anteriores...")
try:
    # Obtener todos los IDs existentes
    stats = index.describe_index_stats()
    if stats['total_vector_count'] > 0:
        # Eliminar todos los vectores existentes
        index.delete(delete_all=True)
        print("✅ Datos anteriores eliminados")
except Exception as e:
    print(f"⚠️ Error limpiando datos: {e}")

# Cargar sentencias reales
judgments = load_judgments_from_rtf()

if not judgments:
    print("❌ No se pudieron cargar sentencias")
    exit(1)

# Preparar vectores para insertar
print("🔄 Generando embeddings...")
vectors_to_upsert = []

for i, judgment in enumerate(judgments, 1):
    print(f"🧠 Generando embedding {i}/{len(judgments)}: {judgment['id']}")
    
    # Crear texto para vectorizar (combinando información clave)
    text_to_embed = f"""
    {judgment['summary']}
    Tipo: {judgment['case_type']}
    Tema: {judgment['tema_principal']}
    Año: {judgment['year']}
    Temas relacionados: {', '.join(judgment['temas'])}
    """.strip()
    
    # Generar embedding
    vector = model.encode(text_to_embed).tolist()
    
    # Preparar metadata rica compatible con el frontend
    metadata = {
        'Providencia': judgment['id'],
        'Información': judgment['summary'],
        'demandante': f"Corte Constitucional",  # minúscula para el frontend
        'Demandante': f"Corte Constitucional",  # mayúscula por compatibilidad
        'Tema': judgment['tema_principal'].title(),
        'Tema - subtema': judgment['tema_principal'].title(),  # Campo que espera el frontend
        'Resumen': judgment['summary'][:200] + "..." if len(judgment['summary']) > 200 else judgment['summary'],
        'summary_extract': judgment['summary'][:200] + "..." if len(judgment['summary']) > 200 else judgment['summary'],  # Campo que espera el frontend
        'Tipo': judgment['case_type'],
        'Fecha Sentencia': f"{judgment['year']}-01-01",  # Fecha aproximada que espera el frontend
        'Tipo de proceso': judgment['case_type'],  # Campo adicional que puede esperar el frontend
        'anio': judgment['year'],
        'fecha_number': judgment['year'] * 10000 + 101,  # Formato YYYYMMDD aproximado
        'temas_relacionados': ', '.join(judgment['temas']),
        'filename': judgment['filename'],
        'texto_completo': judgment['full_text'][:1000]  # Primeros 1000 caracteres
    }
    
    # Crear vector para insertar
    vector_data = {
        'id': judgment['id'],
        'values': vector,
        'metadata': metadata
    }
    
    vectors_to_upsert.append(vector_data)

# Insertar en Pinecone en lotes
print(f"📤 Insertando {len(vectors_to_upsert)} vectores en Pinecone...")

batch_size = 100
for i in range(0, len(vectors_to_upsert), batch_size):
    batch = vectors_to_upsert[i:i + batch_size]
    
    try:
        upsert_response = index.upsert(vectors=batch)
        print(f"✅ Lote {i//batch_size + 1}: {upsert_response['upserted_count']} vectores insertados")
        
    except Exception as e:
        print(f"❌ Error en lote {i//batch_size + 1}: {e}")

# Verificar resultados
try:
    stats = index.describe_index_stats()
    print(f"\n📊 Estadísticas finales del índice:")
    print(f"   - Vectores totales: {stats['total_vector_count']}")
    print(f"   - Dimensiones: {stats.get('dimension', 'N/A')}")
    
except Exception as e:
    print(f"❌ Error obteniendo estadísticas: {e}")

print("\n🎉 ¡Carga de sentencias reales completada!")
print("🔍 Ahora puedes buscar en más de 50 sentencias reales de la Corte Constitucional")
print("💡 Prueba buscar: 'tutela', 'derechos fundamentales', 'debido proceso', etc.")