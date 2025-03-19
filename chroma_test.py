import chromadb
import json

with open("embeddings/all-MiniLM-L6-v2.json", "r") as f:
    embeddings = json.load(f)

chunk_idxs = [str(embedding['chunk_idx']) for embedding in embeddings]
embeddings_list = [embedding['embedding'] for embedding in embeddings]

client = chromadb.HttpClient(host="localhost", port=8000)

collection = client.get_or_create_collection(
    name="ds4300-test"
)

collection.add(
    embeddings=embeddings_list,
    ids=chunk_idxs
)



