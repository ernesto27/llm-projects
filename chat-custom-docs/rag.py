from flask import Flask, request, jsonify, Response, stream_with_context, render_template
import chromadb
import requests

# Connect to ChromaDB running in Docker
CHROMA_HOST = "http://localhost:8000"  # Docker container exposes port 8000
chroma_client = chromadb.PersistentClient(path=CHROMA_HOST)

# Create or get the collection
collection = chroma_client.get_or_create_collection(name="documents")

app = Flask(__name__)

# Insert a document into ChromaDB
def insert_document(doc_id, content):
    collection.add(ids=[doc_id], documents=[content])

# Query the vector database
def query_vector_db(question):
    results = collection.query(query_texts=[question], n_results=1)
    print(f"Query results: {results}")
    if results["documents"]:
        return results["documents"][0][0]  # Return most relevant document
    return "No relevant documents found."

# API Route for Chatbot

def ollama_api_request(model, prompt):
    # # Save prompt to debug filele
    with open('debug_prompts.txt', 'a') as f:
        f.write(f"\n--- {model} ---\n{prompt}\n")
    
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": True}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers, stream=True)
    
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                yield line + b"\n"

@app.route("/")
def chat_interface():
    return render_template('chat.html')

@app.route("/chat", methods=["POST"])
def chat_handler():
    data = request.get_json()
    question = data.get("question", "")
    
    doc_content = query_vector_db(question)
    # model = "llama3.2:3b"
    model = "mistral:latest"
    
    prompt = """Eres un asistente muy estricto que ÚNICAMENTE puede responder usando la información que se encuentra en el siguiente texto delimitado por doble comillas. 

REGLAS IMPORTANTES:
1. Si la información NO está explícitamente en el texto, debes responder "No puedo responder esa pregunta porque la información no se encuentra en los documentos proporcionados."
2. NO debes inferir, suponer, ni agregar información externa bajo ninguna circunstancia
3. NO debes usar conocimiento general o previo
4. Tu respuesta debe estar basada ÚNICAMENTE en el texto proporcionado
5. Si la pregunta no está directamente relacionada con el texto, responde que no puedes ayudar

Pregunta: {question}
Texto de referencia:
""
{doc_content}
"" 
"""

    return Response(
        ollama_api_request(model, prompt.format(doc_content=doc_content, question=question)),
        content_type='text/event-stream'
    )

def read_data_file(filepath):
    try:
        with open(filepath, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Warning: {filepath} not found")
        return "No content available"

if __name__ == "__main__":
    # Read and insert document from data.txt
    content = read_data_file("docs/casos-exito.txt")
    print(content)
    insert_document("1", content)

    content2 = read_data_file("docs/cultura.txt")
    insert_document("2", content2)


    content3 = read_data_file("docs/servicios.txt")
    insert_document("3", content3)




    print("Server running on port 8081...")
    app.run(host="0.0.0.0", port=8081, debug=True, use_reloader=False)
