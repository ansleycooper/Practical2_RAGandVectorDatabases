# DS4300 Practical 2: RAG and Vector Databases
## Timothy Clay, Ansley Cooper, & Nidhi Hosamane

### Introduction
This project was developed in Spring 2025 for DS4300 at Northeastern University. The goal is to test different preprocessing, embedding models, vector databases, and large language models against eachother to see which combination results in the most accurate and robust RAG ... 

### Key Files and Folders
1. `data` Folder: holds the text file that is the compilation of all notes for the class from all three authors of this repository, which is the raw version of the context provided to the RAG
2. `chunked_data` Folder: holds all the output text files for all combinations of preprocessing: chunk size, chunk overlap, punctuation, whitespace, and stopwords
3. `embeddings` Folder: holds 3 JSON files, one for each embedding
4. `embeddings.py` File: **finish**
5. `practical02_driver.py` File: **finish**
6. `results` Folder: **finish**

### Step 1: Set up your environment
We recommend setting up a new environment to run this repository in order to minimize any issues that may arise during to package versions or dependancies. Below are a few resources that may help in the creation of your virtual environment. 
- https://docs.python.org/3/library/venv.html
- https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

### Step 2: Install requirements
Navigate to the `requirements.txt` file using terminal. Run it through terminal using the follwing command: `pip install -r requirements.txt`. Alternatively, you can manually install each of the packages using `pip install [package name]`.

### Step 3: 
Navigate to the 
