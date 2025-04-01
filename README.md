# DS4300 Practical 2: RAG and Vector Databases
## Timothy Clay, Ansley Cooper, & Nidhi Hosamane

### Introduction
This project was developed in Spring 2025 for DS4300 at Northeastern University. The goal is to test different preprocessing, embedding models, vector databases, and large language models against eachother to see which combination results in the most accurate and robust RAG.

### Key Files and Folders
1. `data` Folder: holds the text file that is the compilation of all notes for the class from all three authors of this repository, which is the raw version of the context provided to the RAG
2. `chunked_data` Folder: holds all the output text files for all combinations of preprocessing: chunk size, chunk overlap, punctuation, whitespace, and stopwords
3. `embeddings` Folder: holds 3 JSON files, one for each embedding
4. `embeddings.py` File: contains sample code provided by Dr. Fontenot for embedding
5. `practical02_driver.py` File: this file should be run to generate final results 
6. `results` Folder: holds the results from each combination of experimental design elements in each subfolder

### Step 1: Set up your environment
We recommend setting up a new environment to run this repository in order to minimize any issues that may arise during to package versions or dependancies. Below are a few resources that may help in the creation of your virtual environment. 

Ensure that the environment is set up with Python version 3.10. For example, use `conda create -n [envName] python=3.10`.

- https://docs.python.org/3/library/venv.html
- https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

### Step 2: Run Docker containers
Using the Docker Desktop app, run containers for vector databases.

### Step 3: Install requirements
Clone the Github repository and navigate to it using terminal. Run it through terminal using the following command: `pip install -r requirements.txt`. Alternatively, you can manually install each of the packages using `pip install [package name]`.

### Step 4: Run code
Navigate to practical02_driver.py and run.

### Step 5: View results
Results will appear in the `results` folder, within the subfolder for each combination of our experimental design elements.
