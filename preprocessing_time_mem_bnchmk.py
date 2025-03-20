import os
import time
import re
import csv
import psutil
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Configuration
DATA_DIR = "data"
CHUNK_SIZES = [200, 500, 1000]
CHUNK_OVERLAPS = [0, 50, 100]
PREPROCESSING_STEPS = [
    "lowercase",
    "lowercase_whitespace",
    "lowercase_whitespace_punctuation",
    "lowercase_whitespace_punctuation_noise"
]

STOP_WORDS = set(stopwords.words('english'))

# Create output directory
OUTPUT_DIR = "chunked_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_text(file_path):
    """Load text file"""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
    
def preprocess_text(text, method):
    text = text.lower()
    if method == "lowercase_whitespace":
        text = " ".join(text.split())  # Normalize whitespace (remove extra spaces, tabs, newlines)
    if method in ["lowercase_whitespace_punctuation", "lowercase_whitespace_punctuation_noise"]:
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
    if method == "lowercase_whitespace_punctuation_noise":
        text = ' '.join([word for word in word_tokenize(text) if word not in STOP_WORDS])  # Remove stop words
    return text

def chunk_text(text, chunk_size, overlap):
    words = word_tokenize(text)
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(words[start:end])
        start += chunk_size - overlap if end < len(words) else len(words)
    return chunks

def measure_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def main():
    results = []
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_DIR, filename)
            dataset = load_text(file_path)
            
            for chunk_size in CHUNK_SIZES:
                for overlap in CHUNK_OVERLAPS:
                    for preprocessing in PREPROCESSING_STEPS:
                        start_time = time.time()
                        processed_text = preprocess_text(dataset, preprocessing)
                        chunks = chunk_text(processed_text, chunk_size, overlap)
                        processing_time = time.time() - start_time
                        memory_usage = measure_memory_usage()
                        
                        dataset_name = f"{filename}_size{chunk_size}_overlap{overlap}_{preprocessing}.txt"
                        dataset_path = os.path.join(OUTPUT_DIR, dataset_name)
                        with open(dataset_path, "w") as f:
                            for chunk in chunks:
                                f.write(" ".join(chunk) + "\n")
                        
                        # Save all required columns
                        results.append([filename, chunk_size, overlap, preprocessing, round(processing_time, 4), round(memory_usage, 2)])

    # Save results to CSV
    csv_path = os.path.join(OUTPUT_DIR, "dataset_processing_results.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset Name", "Chunk Size", "Chunk Overlap", "Preprocessing Strategy", "Time (s)", "Memory (MB)"])
        writer.writerows(results)

if __name__ == "__main__":
    main()
