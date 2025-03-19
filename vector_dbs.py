## DS 4300 Example - from docs

# inputs:
#   preprocessed text file

# outputs:
#   an embedded version of the text for each of the different embedders
#   a csv storing encoding times and storage

import time
from datetime import datetime
import pandas as pd
import json
import chromadb
from chromadb import Settings
import redis
import numpy as np


def load_embeddings(filepath):
    data = []
    with open(filepath, "r") as f:
        data = json.load(f)

    ids = [str(datum['chunk_idx']) for datum in data]
    embeddings = [datum['embedding'] for datum in data]

    return ids, embeddings 



# used to clear the redis vector store
def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")

def clear_chroma_store():
    print("Clearing existing Chroma collections...")
    collections = chroma_client.list_collections()
    
    for collection in collections:
        chroma_client.delete_collection(collection)
    
    print("Chroma database cleared.")

# Create an HNSW index in Redis
def create_redis_index(index_name, doc_prefix="DOC:", vector_dim=384, distance_metric="COSINE"):
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

def index_embeddings_redis(ids: list, embeddings: list):
    for i in range(len(ids)):
        key = ids[i]
        redis_client.hset(
            key,
            mapping={
                "embedding": np.array(
                    embeddings[i], dtype=np.float32
                ).tobytes(),  
            },
        )
        #print(f"Stored embedding for: index {ids[i]}")

def test_redis(ids, embeddings):
    print(f'Testing redis...')
    start_time = time.time() 

    index_embeddings_redis(ids, embeddings)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f'Execution time: {execution_time}')

    pd.DataFrame([{'time':datetime.now(),
                   'db': 'Redis',
                   'execution_time':execution_time}]).to_csv('indexing_logging.csv', index=False, mode='a', header=False)


def index_embeddings_chroma(ids: list, embeddings) -> list:

    collection = chroma_client.get_or_create_collection(
        name="ds4300-rag"
    )

    collection.add(
        ids=ids,
        embeddings=embeddings
    )


def test_chroma(ids, embeddings):
    print(f'Testing chroma...')
    start_time = time.time() 

    index_embeddings_chroma(ids, embeddings)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f'Execution time: {execution_time}')

    pd.DataFrame([{'time':datetime.now(),
                   'db': 'Chroma',
                   'execution_time':execution_time}]).to_csv('indexing_logging.csv', index=False, mode='a', header=False)


if __name__ == '__main__':

    redis_client = redis.Redis(host="localhost", port=6379, db=0)
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)

    ids, embeddings = load_embeddings('embeddings/all-MiniLM-L6-v2.json')

    clear_redis_store()
    create_redis_index('ds4300-rag')
    test_redis(ids, embeddings)

    print()

    clear_chroma_store()
    test_chroma(ids, embeddings)
    