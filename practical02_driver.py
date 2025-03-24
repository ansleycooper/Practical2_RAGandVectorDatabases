import os
from embeddings import make_embeddings
from mistralRAG import run_mistralRAG
from ollamaRAG import run_ollamaRAG


input_folder = "chunked_data" 
results_folder = "results"

# Ensure results folder exists
os.makedirs(results_folder, exist_ok=True)

# Iterate through each file in the input folder
for filename in os.listdir(input_folder):
    print(f"Processing {filename}...")
    if filename.endswith(".txt"):  # Ensure it's a text document
        file_path = os.path.join(input_folder, filename)
        subfolder_name = os.path.splitext(filename)[0]  # Use filename without extension
        chunked_data_str = "chunked_data/" + filename
        make_embeddings(chunked_data_str)
        run_ollamaRAG(subfolder_name)
        run_mistralRAG(subfolder_name)
        print(f"Processed {filename}")

print("✅ Processing complete. All results saved in the 'results' folder. ✅")
