## DS 4300 Example - from docs

# inputs:
#   preprocessed text file

# outputs:
#   an embedded version of the text for each of the different embedders
#   a csv storing encoding times and storage


import os
import fitz
from sentence_transformers import SentenceTransformer
import psutil
import time
from datetime import datetime
import pandas as pd
import json


def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page
def encode_data():
    chunks = []
    for file_name in os.listdir('data/'):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join('data/', file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks += split_text_into_chunks(text)

    with open('test.txt', 'w') as f:
        for chunk_index, chunk in enumerate(chunks):
            f.write(chunk + "\n")
                
    print(f" -----> Processed {file_name}")


def load_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(line.strip())
    return data


def get_embeddings(data: list, model) -> list:

    embedding_model = SentenceTransformer(model, trust_remote_code=True)

    embeddings = []
    for text in data:
        embeddings += [embedding_model.encode(text).tolist()]
    
    return embeddings


def test_embedding(data, embedding_model):
    print(f'Testing model {embedding_model}...')
    start_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024) 

    embeddings = get_embeddings(data, embedding_model)

    end_time = time.time()
    mem_after = process.memory_info().rss / (1024 * 1024) 
    execution_time = end_time - start_time
    memory_usage = mem_after - mem_before

    print(f'Execution time: {execution_time}')
    print(f'Memory usage: {memory_usage}')

    pd.DataFrame([{'time':datetime.now(), 
                   'model':embedding_model, 
                   'execution_time':execution_time,
                   'memory_usage':memory_usage}]).to_csv('embedding_logging.csv', index=False, mode='a', header=False)

    return embeddings

def save_embeddings(data, filepath):

    embeddings = [{'chunk_idx':embedding_idx, 'embedding':embedding} for embedding_idx, embedding in enumerate(data)]
    with open(filepath, "w") as f:
        json.dump(embeddings, f, indent=4)

if __name__ == '__main__':
    data = load_data('chunked_data/Practical 2_ Data.txt_size500_overlap100_lowercase_whitespace.txt')
    os.makedirs('embeddings', exist_ok=True)

    
    embeddings_1 = test_embedding(data, 'all-MiniLM-L6-v2')
    save_embeddings(embeddings_1, 'embeddings/all-MiniLM-L6-v2.json')

    embeddings_2 = test_embedding(data, 'all-mpnet-base-v2')
    save_embeddings(embeddings_2, 'embeddings/all-mpnet-base-v2.json')

    embeddings_3 = test_embedding(data, 'nomic-ai/nomic-embed-text-v1')
    save_embeddings(embeddings_3, 'embeddings/nomic-embed-text-v1.json')