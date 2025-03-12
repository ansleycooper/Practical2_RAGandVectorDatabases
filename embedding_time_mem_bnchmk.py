import time
import psutil
import csv
import os



# Output directory for indexing
OUTPUT_DIR = "embedded_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV log file for indexing benchmarks
csv_filename = "embedding_models_benchmark_log.csv"
csv_fields = ["Filename", "Time (s)", "Memory (MB)"]

# Retrieve all files from preprocessed_text directory, excluding DS_Store
files = [f for f in os.listdir("indexes/preprocessed_text") if os.path.isfile(os.path.join("indexes/preprocessed_text", f)) and f != ".DS_Store"]

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_fields)
    
    for filename in files:
        start_time = time.time()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        
        # Placeholder for indexing process
        # TODO: Insert code to build the indexers
        
        # Record time and memory usage
        end_time = time.time()
        mem_after = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        execution_time = end_time - start_time
        memory_usage = mem_after - mem_before
        
        # Write to CSV log
        writer.writerow([filename, round(execution_time, 4), round(memory_usage, 4)])

print(f"Embedding models benchmark log saved to {csv_filename}")
