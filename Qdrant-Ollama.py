import json
import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from sentence_transformers import SentenceTransformer  # For generating embeddings with transformers
import ollama

VECTOR_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "nomic-embed-text-v1": 768,
}
DISTANCE_METRIC = "Cosine"  # This can be 'Cosine' or 'Euclidean' depending on your preference

# List your embedding files
EMBEDDING_FILES = {
    "all-MiniLM-L6-v2": "embeddings/all-MiniLM-L6-v2.json",
    "all-mpnet-base-v2": "embeddings/all-mpnet-base-v2.json",
    "nomic-embed-text-v1": "embeddings/nomic-embed-text-v1.json",
}

def load_embeddings(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint, VectorParams

from qdrant_client.models import PointStruct, VectorParams

def store_embeddings_in_qdrant(client, data, collection_name):
    # Create collection if it does not exist
    print(f'VECTOR_DIMS[collection_name]: {VECTOR_DIMS[collection_name]}')
    try:
        client.get_collection(collection_name)
        # Delete the existing collection if needed
        client.delete_collection(collection_name=collection_name)
        # If collection doesn't exist, create it
        print(f"Collection '{collection_name}' not found, creating it...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_DIMS[collection_name], distance=DISTANCE_METRIC)
        )
    except Exception:
        # If collection doesn't exist, create it
        print(f"Collection '{collection_name}' not found, creating it...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_DIMS[collection_name], distance=DISTANCE_METRIC)
        )


    # Store the embeddings and metadata in Qdrant
    points = []
    for idx, item in enumerate(data):
        # Create a chunk ID as an integer (this is now just the index)
        print(f'embedding length: {len(item["embedding"])}')
        points.append(
            PointStruct(
                id=idx,  # Use the index as the integer-based point ID
                vector=item["embedding"],  # Ensure the embedding is a list of floats
                payload={"chunk_idx": idx, "text": item.get("text", "")}  # Add metadata
            )
        )
        
    # Upsert the points into the Qdrant collection
    client.upsert(collection_name=collection_name, points=points)
    print(f"Stored {len(data)} embeddings in Qdrant collection: {collection_name}")


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

def search_embeddings_in_qdrant(client, query, collection_name, top_n=3):
    """Search for relevant embeddings in the given Qdrant collection."""
    query_embedding = get_query_embedding(query, collection_name)
    
    # Perform search in the collection
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_n  # Number of similar results to return
    )
    
    print(f"Found {len(results)} results in collection '{collection_name}'")
    
    # Extract the most relevant texts and similarities based on the query result
    context_results = []
    for result in results:
        context_results.append({
            "chunk": result.payload["text"],
            "similarity": result.score
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

def interactive_search(client):
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

            context_results = search_embeddings_in_qdrant(client, query, model_name)

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
    # Initialize Qdrant client
    client = QdrantClient(host="localhost", port=6333)
    print("Connected to Qdrant", client)

    embedding_files = [
        "embeddings/all-MiniLM-L6-v2.json",
        "embeddings/all-mpnet-base-v2.json",
        "embeddings/nomic-embed-text-v1.json"
    ]

    for file_path in embedding_files:
        if os.path.exists(file_path):
            collection_name = os.path.basename(file_path).replace(".json", "")
            embeddings = load_embeddings(file_path)
            store_embeddings_in_qdrant(client, embeddings, collection_name)
        else:
            print(f"File not found: {file_path}")
    
    interactive_search(client)

if __name__ == "__main__":
    main()
