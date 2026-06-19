
# Olympic History RAG Assistant

This project is a **Retrieval-Augmented Generation (RAG) system** that allows users to 
chat with a local AI about Olympic history. It uses a Wikipedia text corpus and a Kaggle 
dataset to provide accurate, context-based answers using **Ollama** and **LangChain Classic**.

---

## Prerequisites

Before setup, ensure you have the following installed:

- Python 3.10+
- Ollama Desktop (Download at https://ollama.com)
- Visual Studio Code (Recommended)

---

## Part 1 — Streamlit Chatbot Setup (Windows)

### 1. Create and activate virtual environment
```bash
python -m venv venv
.\venv\Scripts\Activate
```

### 2. Install dependencies
```bash
pip install streamlit pandas langchain-community langchain-huggingface langchain-chroma langchain-text-splitters langchain-ollama langchain-classic
```

### 3. Pull Ollama model
```bash
ollama pull qwen2.5:1.5b
```

### 4. Preprocess the data
```bash
python preprocess.py
```

### 5. Run the chatbot UI
```bash
streamlit run app.py
```
Click **Initialize / Sync Knowledge Base** in the sidebar on first run.  
This may take 5–15 minutes to build the vector database on first launch.  
Subsequent runs load instantly from disk.

### 6. Run evaluation (optional)
```bash
python evaluate.py
```
Runs 23 test questions through the RAG system and saves scores to `evaluation_results.csv`.

---

