## DS 4300 Example - from docs

# inputs:
#   preprocessed text file

# outputs:
#   an embedded version of the text for each of the different embedders
#   a csv storing encoding times and storage

import redis
import numpy as np
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6380, db=0)

DOC_PREFIX = "doc:"
INDEX_NAME = "embedding_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list:

    embedding_model = SentenceTransformer(model, trust_remote_code=True)
    return embedding_model.encode(text).tolist()


# store the embedding in Redis
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(), 
        },
    )
    print(f"Stored embedding for: {chunk}")

def search_embeddings(query, top_k=3):

    query_embedding = get_embedding(query)

    # Convert embedding to bytes for Redis search
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        # Construct the vector similarity search query
        # Use a more standard RediSearch vector search syntax
        # q = Query("*").sort_by("embedding", query_vector)

        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        # Perform the search
        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        # Transform results into the expected format
        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ][:top_k]

        # Print results for debugging
        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []