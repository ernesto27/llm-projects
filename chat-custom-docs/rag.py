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
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json().get("response", "")
    return "Error generating response."

@app.route("/")
def chat_interface():
    return render_template('chat.html')

@app.route("/chat", methods=["POST"])
def chat_handler():
    data = request.get_json()
    question = data.get("question", "")
    
    doc_content = query_vector_db(question)
    print(f"Document content: {doc_content}")
    #model = "deepseek-r1:latest"
    #model = "mistral"
    model = "llama3.2:3b"
    
    response = ollama_api_request(model, f"Answer based on the document: {doc_content}")
    
    return jsonify({"response": response})

def read_data_file(filepath):
    try:
        with open(filepath, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Warning: {filepath} not found")
        return "No content available"

if __name__ == "__main__":
    # Read and insert document from data.txt
    content = read_data_file("data.txt")
    print(content)
    insert_document("1", content)


    print("Server running on port 8081...")
    app.run(host="0.0.0.0", port=8081, debug=True)
