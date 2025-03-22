import redis
import json
import numpy as np
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import ollama
from sentence_transformers import SentenceTransformer



VECTOR_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "nomic-embed-text": 768,
}
DISTANCE_METRIC = "COSINE"

# List your embedding files
EMBEDDING_FILES = {
    "all-MiniLM-L6-v2": "embeddings/all-MiniLM-L6-v2.json",
    "all-mpnet-base-v2": "embeddings/all-mpnet-base-v2.json",
    "nomic-embed-text": "embeddings/nomic-embed-text-v1.json",
}

def create_redis_index(redis_client, index_name, vector_dim):
    print(vector_dim)
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

def store_embeddings(redis_client, index_name, file_path):
    """Stores embeddings from a JSON file into Redis."""
    with open(file_path, "r") as f:
        embeddings = json.load(f)

    for entry in embeddings:
        redis_key = f"{index_name}:{entry['chunk_idx']}"
        redis_client.hset(redis_key, mapping={
        "chunk": entry["chunk_idx"],
        "embedding": np.array(entry["embedding"], dtype=np.float32).tobytes(),
        "text": entry["text"]  # Store the original text metadata
        })


    print(f"Stored {len(embeddings)} embeddings in '{index_name}'.")
    
    
def get_embedding(text, model):
    if model == "nomic-embed-text-v1":
        """Get text embedding using Ollama"""
        response = ollama.embeddings(model='nomic-embed-text', prompt=text)
        return response["embedding"]
    elif model == "all-MiniLM-L6-v2" or model == "all-mpnet-base-v2":
        # Load the model once (do this globally to avoid reloading on every call)
        embedding_model = SentenceTransformer(model)
        return embedding_model.encode(text).tolist()
    else:
        print(f"Model '{model}' not supported.")

def search_embeddings(redis_client, query, index_name, top_k=3):
    """Search for similar embeddings in a given index."""
    query_embedding = get_embedding(query, model=index_name)

    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    q = (
        Query(f"*=>[KNN {top_k} @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("chunk", "vector_distance")
        .dialect(2)  # Ensure compatibility
    )

    try:
        results = redis_client.ft(index_name).search(q, query_params={"vec": query_vector})
        return [{"chunk": r.chunk, "similarity": r.vector_distance} for r in results.docs]
    except Exception as e:
        print(f"Error searching index {index_name}: {e}")
        return []

def main():
     # Initialize Redis
    redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=False)
    # Create indexes and store embeddings
    for model_name, file_path in EMBEDDING_FILES.items():
        create_redis_index(redis_client, model_name, VECTOR_DIMS[model_name])
        store_embeddings(redis_client, model_name, file_path)

if __name__ == "__main__":
    main()
