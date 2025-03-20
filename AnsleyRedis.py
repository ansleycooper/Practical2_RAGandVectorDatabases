import redis
import json
import numpy as np
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import ollama

# Initialize Redis
redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=False)

VECTOR_DIM = 768  # Ensure this matches the embeddings in your JSON files
DISTANCE_METRIC = "COSINE"

# List your embedding files
EMBEDDING_FILES = {
    "all-MiniLM-L6-v2": "embeddings/all-MiniLM-L6-v2.json",
    "all-mpnet-base-v2": "embeddings/all-mpnet-base-v2.json",
    "nomic-embed-text": "embeddings/nomic-embed-text-v1.json",
}

def create_redis_index(index_name, vector_dim):
    """Creates a new index for a specific embedding model."""
    try:
        redis_client.ft(index_name).info()
        print(f"Index '{index_name}' already exists.")
        return
    except:
        print(f"Creating index '{index_name}'...")

    schema = [
        TextField("chunk"),
        NumericField("chunk_idx"),
        VectorField("embedding", "HNSW", {
            "TYPE": "FLOAT32",
            "DIM": vector_dim,
            "DISTANCE_METRIC": DISTANCE_METRIC,
        }),
    ]

    redis_client.ft(index_name).create_index(
        schema, definition=IndexDefinition(prefix=[f"{index_name}:"])
    )
    print(index_name)

def store_embeddings(index_name, file_path):
    """Stores embeddings from a JSON file into Redis."""
    with open(file_path, "r") as f:
        embeddings = json.load(f)

    for entry in embeddings:
        redis_key = f"{index_name}:{entry['chunk_idx']}"
        redis_client.hset(redis_key, mapping={
            "chunk": entry["chunk_idx"],
            "embedding": np.array(entry["embedding"], dtype=np.float32).tobytes(),
        })

    print(f"Stored {len(embeddings)} embeddings in '{index_name}'.")

# Create indexes and store embeddings
for model_name, file_path in EMBEDDING_FILES.items():
    create_redis_index(model_name, VECTOR_DIM)
    store_embeddings(model_name, file_path)
    
def get_embedding(text, model="nomic-embed-text"):
    """Get text embedding using Ollama"""
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def search_embeddings(query, index_name, top_k=3):
    """Search for similar embeddings in a given index."""
    query_embedding = get_embedding(query)

    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("chunk", "vector_distance")
    )

    try:
        results = redis_client.ft(index_name).search(q, query_params={"vec": query_vector})
        return [{"chunk": r.chunk, "similarity": r.vector_distance} for r in results.docs]
    except Exception as e:
        print(f"Error searching index {index_name}: {e}")
        return []

def generate_rag_response(query, context_results):
    """Generates a response using retrieved context and Ollama LLM."""
    context_str = "\n".join(
        [f"Chunk: {r['chunk']} (Similarity: {float(r['similarity']):.2f})" for r in context_results]

    )

    prompt = f"""You are a helpful AI assistant. 
Use the following context to answer the query accurately. If the context is not relevant, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    response = ollama.chat(model="mistral:latest", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def interactive_search():
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

            context_results = search_embeddings(query, model_name)

            if not context_results:
                print(f"⚠️ No results found in index '{model_name}'.")
                continue  # Skip if no results

            print(f"Results from {model_name}: {context_results}")  # Debugging print

            if float(context_results[0]["similarity"]) < best_similarity:
                best_response = generate_rag_response(query, context_results)
                best_similarity = context_results[0]["similarity"]

        print("\n--- Best Response ---")
        print(best_response if best_response else "No relevant information found.")

interactive_search()
