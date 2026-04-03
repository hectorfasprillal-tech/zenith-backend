import json
from flask import Flask, request, jsonify
from flask_cors import CORS

from models.semantic_search import search, is_ready
#from models.finetunned_model import chatbot
try:
    from models.rag import chatbot
except Exception as e:
    chatbot = None
    print(f"⚠️ SERVER WARN: Chatbot deshabilitado: {str(e)}")

# flash app
app = Flask(__name__)
CORS(app)


@app.route('/search', methods=['POST'])
def search_route():
    try:
        payload = request.json or {}
        results = search(payload).get('matches', [])
        def _to_dict(item):
            return getattr(item, "_data_store", item)
        return jsonify([_to_dict(item) for item in results])
    except Exception as e:
        return jsonify({"error": "Search failed", "detail": str(e)}), 500

@app.route('/health')
def health_route():
    return jsonify({"status": "ok", "pinecone_ready": is_ready()})

@app.route('/chatbot')
def chatbot_route():
    if chatbot is None:
        return jsonify(dict(
            response="Chatbot no disponible en este entorno"
        ))
    query = request.args['query']
    print(f"\n🌐 SERVER DEBUG: Recibida consulta en /chatbot: {query}")
    
    try:
        response = chatbot(query)
        print(f"🌐 SERVER DEBUG: Respuesta del chatbot: {response}")
        return jsonify(dict(
            response=response
        ))
    except Exception as e:
        print(f"❌ SERVER ERROR: {str(e)}")
        return jsonify(dict(
            response=f"Error del servidor: {str(e)}"
        ))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
