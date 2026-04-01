# CST8507 Assignment 2

##  Olympic History RAG Assistant

This project is a **Retrieval-Augmented Generation (RAG) system** that allows users to chat with a local AI about Olympic history.  
It uses a Wikipedia text corpus and a Kaggle dataset to provide accurate, context-based answers using **Ollama** and **LangChain Classic**.

---

##  Prerequisites

Before setup, ensure you have the following installed:

- Python 3.10+
- Ollama Desktop (Download at https://ollama.com)
- Visual Studio Code (Recommended)

---

##  Setup Instructions


---

###  Virtual Environment & Dependencies

Open PowerShell and run:


### Create virtual environment
```bash
python -m venv venv
```

### Activate environment
```bash
\venv\Scripts\Activate
```

### Install dependencies
```bash
pip install streamlit pandas langchain-community langchain-huggingface langchain-chroma langchain-text-splitters langchain-ollama langchain-classic
```

### pull ollama
```bash
ollama pull qwen2.5:0.5b
```

### run front ui
```bash
streamlit run app.py
```
