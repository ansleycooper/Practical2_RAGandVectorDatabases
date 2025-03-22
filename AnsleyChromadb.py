import chromadb
import numpy as np
import json
import os
import ollama
from sentence_transformers import SentenceTransformer

VECTOR_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "nomic-embed-text-v1": 768,
}
DISTANCE_METRIC = "COSINE"

# List your embedding files
EMBEDDING_FILES = {
    "all-MiniLM-L6-v2": "embeddings/all-MiniLM-L6-v2.json",
    "all-mpnet-base-v2": "embeddings/all-mpnet-base-v2.json",
    "nomic-embed-text-v1": "embeddings/nomic-embed-text-v1.json",
}

def clear_chroma_db():
    """Delete all collections from ChromaDB."""
    db = chromadb.PersistentClient(path="chroma_db")  # Adjust path if needed
    collection_names = db.list_collections()  # Returns a list of names (strings)
    
    for name in collection_names:
        db.delete_collection(name)  # Directly delete by name
        print(f"Deleted collection: {name}")
    
    print("cleared.")

def load_embeddings(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def store_embeddings_in_chroma(db, data, collection_name, model_name="all-MiniLM-L6-v2"):
    # Initialize Chroma client and model
    collection = db.get_or_create_collection(collection_name)
    
    # Store the embeddings and metadata in ChromaDB
    for idx, item in enumerate(data):
        chunk_id = f"doc:{collection_name}:{idx}"
        collection.add(
            ids=[chunk_id],  # Unique id
            embeddings=[item["embedding"]],  # The embedding vector
            metadatas=[{"chunk_idx": idx, "text": item.get("text", "")}]  # Metadata with chunk index and original text
        )
    print(f"Stored {len(data)} embeddings in ChromaDB collection: {collection_name}")


def get_query_embedding(text, model):
    if model == "nomic-embed-text-v1":
        """Get text embedding using Ollama"""
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]
    elif model == "all-MiniLM-L6-v2" or model == "all-mpnet-base-v2":
        # Load the model once (do this globally to avoid reloading on every call)
        embedding_model = SentenceTransformer(model)
        return embedding_model.encode(text).tolist()
    else:
        print(f"Model '{model}' not supported.")


def search_embeddings_in_chroma(db, query, collection_name, top_n=3):
    """Search for relevant embeddings in the given ChromaDB collection."""
    collection = db.get_collection(collection_name)
    query_embedding = get_query_embedding(query, collection_name)
    
    # Query the collection for the most similar embeddings
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n  # Get the top N results
    )
    print(f"Found {len(results['ids'][0])} results in collection '{collection_name}'")  # Debugging print
    # Extract the most relevant texts and similarities based on the query result
    context_results = []
    for idx, ids in enumerate(results["ids"][0]):  # Access the nested list of documents
        context_results.append({
            "chunk": results["metadatas"][0][0]["text"],  # Access the metadata's text field
            "similarity": results["distances"][0][idx]  # Access the corresponding similarity score
        })
    
    return context_results



def generate_rag_response(query, context_results):
    """Generates a response using retrieved context and Ollama LLM."""
    context_str = "\n".join(
        [f"Chunk: {r['chunk']} (Similarity: {float(r['similarity']):.2f})" for r in context_results]
    )
    print(context_str)  # Debugging print
    prompt = f"""You are a helpful AI assistant. 
        Use the following context to answer the query accurately. If the context is not relevant, say 'I don't know'.
    Context:
    {context_str}

    Query: {query}

    Answer:"""

    response = ollama.chat(model="mistral:latest", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def interactive_search(db):
    """Interactive search across all embedding models."""
    print("🔍 Multi-Model RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your query: ")

        if query.lower() == "exit":
            break

        print(f"Searching for: {query}")  # Debugging print

        best_response = None
        best_similarity = float("inf")

        for model_name in EMBEDDING_FILES.keys():
            print(f"Checking index: {model_name}")  # Debugging print

            context_results = search_embeddings_in_chroma(db, query, model_name)

            if not context_results:
                print(f"⚠️ No results found in index '{model_name}'.")
                continue  # Skip if no results

            print(f"Results from {model_name}: {context_results}")  # Debugging print

            # Get the best response based on the highest similarity
            if float(context_results[0]["similarity"]) < best_similarity:
                best_response = generate_rag_response(query, context_results)
                best_similarity = context_results[0]["similarity"]

        print("\n--- Best Response ---")
        print(best_response if best_response else "No relevant information found.")

def main():
    db = chromadb.PersistentClient(path="chroma_db")
    clear_chroma_db()

    embedding_files = [
        "embeddings/all-MiniLM-L6-v2.json",
        "embeddings/all-mpnet-base-v2.json",
        "embeddings/nomic-embed-text-v1.json"
    ]

    for file_path in embedding_files:
        if os.path.exists(file_path):
            collection_name = os.path.basename(file_path).replace(".json", "")
            embeddings = load_embeddings(file_path)
            store_embeddings_in_chroma(db, embeddings, collection_name)
        else:
            print(f"File not found: {file_path}")
    interactive_search(db)

if __name__ == "__main__":
    main()
