import json
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import uuid

# Configuración
API_KEY = 'pcsk_4T7tGp_NhkTJjhUG6459Ekg2MhZwKbY1wtGGAFuy8H7FeCc8ScWV2fLAzWmaMJ8vm4viM9'
INDEX_NAME = "relatoria-emebeddings"

# Inicializar Pinecone y modelo
pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("📊 Cargando datos de prueba en Pinecone...")

# Cargar datos de prueba
with open('/Users/tuka/Documents/GitHub/zenith_ai/PNL_Zenith/PNL_Zenith/data/judgments_test.json', 'r', encoding='utf-8') as f:
    judgments = json.load(f)

# Preparar vectores para insertar
vectors_to_upsert = []

for judgment in judgments:
    # Crear texto para vectorizar
    text_to_embed = f"{judgment['description']} {judgment['case_type']} {judgment['court']}"
    
    # Generar embedding
    vector = model.encode(text_to_embed).tolist()
    
    # Preparar metadata
    metadata = {
        'Providencia': judgment['id_judgment'],
        'Información': judgment['description'],
        'Demandante': f"Caso {judgment['id_judgment']}",
        'Tema': judgment['case_type'].title(),
        'Resumen': f"Sentencia de {judgment['case_type']} de la Corte Constitucional del año {judgment['date']}",
        'Tipo': judgment['case_type'],
        'anio': int(judgment['date']),
        'fecha_number': int(judgment['date'] + '0101'),  # Fecha aproximada
        'url': judgment['url']
    }
    
    # Crear vector para insertar
    vector_data = {
        'id': judgment['id_judgment'],
        'values': vector,
        'metadata': metadata
    }
    
    vectors_to_upsert.append(vector_data)

# Insertar vectores en Pinecone
try:
    upsert_response = index.upsert(vectors=vectors_to_upsert)
    print(f"✅ Insertados {upsert_response['upserted_count']} vectores exitosamente!")
    
    # Verificar estadísticas
    stats = index.describe_index_stats()
    print(f"📊 Estadísticas actualizadas del índice:")
    print(f"   - Vectores totales: {stats['total_vector_count']}")
    
except Exception as e:
    print(f"❌ Error al insertar datos: {e}")

print("\n🎉 Carga de datos completada!")