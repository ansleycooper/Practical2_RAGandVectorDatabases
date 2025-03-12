import time
import psutil
import csv
import os
from itertools import product

# Define chunk sizes and overlap
CHUNK_SIZES = [200, 500, 1000]
CHUNK_OVERLAPS = [0, 50, 100]
PREPROCESSING_STRATEGIES = ["lowercase", "lowercase_whitespace_punctuation", "lowercase_whitespace_punctuation_noise"]

# Output directory
OUTPUT_DIR = "indexes/preprocessed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV log file
csv_filename = "benchmark_log.csv"
csv_fields = ["Dataset Name", "Chunk Size", "Chunk Overlap", "Preprocessing Strategy", "Time (s)", "Memory (MB)"]

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_fields)
    
    for chunk_size, overlap, strategy_name in product(CHUNK_SIZES, CHUNK_OVERLAPS, PREPROCESSING_STRATEGIES):
        start_time = time.time()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        
        # Simulate dataset creation (no actual preprocessing)
        dataset_filename = f"{OUTPUT_DIR}/{chunk_size}_{overlap}_{strategy_name}.txt"
        """
        
        DO PREPROCESSING HERE
        
        """
        
        # Record time and memory usage
        end_time = time.time()
        mem_after = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        execution_time = end_time - start_time
        memory_usage = mem_after - mem_before
        
        # Write to CSV log
        writer.writerow([dataset_filename, chunk_size, overlap, strategy_name, round(execution_time, 4), round(memory_usage, 4)])

print(f"Benchmark log saved to {csv_filename}")