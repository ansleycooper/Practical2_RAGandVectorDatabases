# input:
#   embedded text files

# outputs:
#   indexes a model for each of the different model types and tracks time/memory it takes for each 

import redis

INDEX_NAME = "embedding_index"
VECTOR_DIM = 384
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# used to clear the redis vector store
def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")

# Create an HNSW index in Redis
def create_hnsw_index(index_name=INDEX_NAME, doc_prefix=DOC_PREFIX, vector_dim=VECTOR_DIM, distance_metric=DISTANCE_METRIC):
    try:
        redis_client.execute_command(f"FT.DROPINDEX {index_name} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {index_name} ON HASH PREFIX 1 {doc_prefix}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {vector_dim} TYPE FLOAT32 DISTANCE_METRIC {distance_metric}
        """
    )
    print("Index created successfully.")