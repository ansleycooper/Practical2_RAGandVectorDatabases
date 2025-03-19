from pymilvus import MilvusClient, Collection, FieldSchema, DataType
import json

with open("embeddings/all-MiniLM-L6-v2.json", "r") as f:
    embeddings = json.load(f)

chunk_idxs = [str(embedding['chunk_idx']) for embedding in embeddings]
embeddings_list = [embedding['embedding'] for embedding in embeddings]

client = MilvusClient(host='localhost', port='19530')

if client.has_collection(collection_name="ds4300_test"):
    client.drop_collection(collection_name="ds4300_test")
client.create_collection(
    collection_name="ds4300_test",
    dimension=384, 
)

entities = {"id": list(range(len(embeddings))), "embedding": embeddings_list}
client.insert(collection_name="ds4300_test", data=entities)
