from pinecone import Pinecone, ServerlessSpec
import time

# Configuración
API_KEY = 'pcsk_4T7tGp_NhkTJjhUG6459Ekg2MhZwKbY1wtGGAFuy8H7FeCc8ScWV2fLAzWmaMJ8vm4viM9'
INDEX_NAME = "relatoria-emebeddings"

# Inicializar Pinecone
pc = Pinecone(api_key=API_KEY)

print("🔧 Configurando Pinecone...")

# Verificar si el índice ya existe
existing_indexes = pc.list_indexes()
index_names = [index['name'] for index in existing_indexes]

if INDEX_NAME in index_names:
    print(f"✅ El índice '{INDEX_NAME}' ya existe.")
else:
    print(f"📝 Creando índice '{INDEX_NAME}'...")
    
    # Crear el índice
    # Dimensión 384 para el modelo 'sentence-transformers/all-MiniLM-L6-v2'
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    
    # Esperar a que el índice esté listo
    print("⏳ Esperando a que el índice esté listo...")
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)
    
    print(f"✅ Índice '{INDEX_NAME}' creado exitosamente!")

# Verificar conexión
try:
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    print(f"📊 Estadísticas del índice:")
    print(f"   - Vectores totales: {stats['total_vector_count']}")
    print(f"   - Dimensión: {stats['dimension']}")
    print("✅ Conexión exitosa con Pinecone!")
    
except Exception as e:
    print(f"❌ Error al conectar con el índice: {e}")

print("\n🎉 Configuración de Pinecone completada!")