import json
import csv
import os
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama
import redis
import chromadb
from AnsleyChromadb import establish_chroma, search_embeddings_in_chroma
from AnsleyRedis import create_redis_index, store_embeddings, search_embeddings
from AnsleyQdrant import search_embeddings_in_qdrant, establish_qdrant

VECTOR_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "nomic-embed-text-v1": 768,
}
DISTANCE_METRIC = "Cosine"

EMBEDDING_FILES = {
    "all-MiniLM-L6-v2": "embeddings/all-MiniLM-L6-v2.json",
    "all-mpnet-base-v2": "embeddings/all-mpnet-base-v2.json",
    "nomic-embed-text-v1": "embeddings/nomic-embed-text-v1.json",
}

CLIENTS = {
    "redis": redis.StrictRedis(host="localhost", port=6379, decode_responses=False),
    "chromadb": chromadb.PersistentClient(path="chroma_db"),  
    "QdrantClient": QdrantClient(host="localhost", port=6333)
}

# Predefined queries
QUERIES_DICT = {
    1: "Do Transactions obtain locks on data when they read or write?",
    2: "What is the difference between a list where memory is contiguously allocated and a list where linked structures are used?",
    3: "What data types can be stored as values in Redis?",
    4: "Create a redis database that is able to pass its keys as tokens into an Ollama RAG system.",
    5: "Why does the CAP principle not make sense when applied to a single-node MongoDB instance?",
    6: "Insert the following values into a binary search tree and then provide a post order traversal: 23 17 20 42 31 50.",
    7: "When was Booker T. Washington’s birthday?",
    8: "who am I?",
    9: "Who was the most streamed artist on Spotify in 2015?",
}

def get_query_embedding(text, model):
    if model == "nomic-embed-text-v1":
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]
    elif model in ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]:
        embedding_model = SentenceTransformer(model)
        return embedding_model.encode(text).tolist()
    else:
        raise ValueError(f"Unsupported model: {model}")

def generate_rag_response(query, context_results):
    """Generate a response using retrieved context and the Ollama LLM."""
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

def search_and_record():
    """Search embeddings in each vector database and record results for all queries."""
    results = []

    for query_id, query_text in QUERIES_DICT.items():
        print(f"\n\nProcessing query {query_id}: {query_text}")

        for client_name, client in CLIENTS.items():
            for model_name in EMBEDDING_FILES.keys():
                print(f"🔎 Searching {client_name} with model {model_name}...")

                # Retrieve context based on the database type
                if client_name == "redis":
                    context_results = search_embeddings(client, query_text, model_name)
                elif client_name == "chromadb":
                    context_results = search_embeddings_in_chroma(client, query_text, model_name)
                elif client_name == "QdrantClient":
                    context_results = search_embeddings_in_qdrant(client, query_text, model_name)

                if not context_results:
                    print(f"      ⚠️ No results found in {client_name} for {model_name}.")
                    continue

                # Generate response
                response = generate_rag_response(query_text, context_results)
                best_similarity = float(context_results[0]["similarity"])

                # Append results
                results.append([query_text, client_name, model_name, response, best_similarity])

    # Write results to CSV
    with open("ollama_rag_results.tsv", mode="a", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerows(results)

    print('\nresults recorded in ollama_rag_results.tsv')

def main():
    for client_name, client in CLIENTS.items():
        if client_name == "redis":
            for model_name, file_path in EMBEDDING_FILES.items():
                create_redis_index(client, model_name, VECTOR_DIMS[model_name])
                store_embeddings(client, model_name, file_path)
        elif client_name == "chromadb":
            establish_chroma()
        elif client_name == "QdrantClient":
            establish_qdrant()

    # make CSV file and add headers i doesn't exist
    if not os.path.exists("ollama_rag_results.tsv"):
        with open("ollama_rag_results.tsv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Query", "Database", "Embedding Model", "Generated Response", "Best Similarity Score"])

    search_and_record()

if __name__ == "__main__":
    main()
