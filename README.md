# CST8507 Assignment 2
## Olympic History RAG Assistant

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

## Part 2 — ROS2 Node Setup (Ubuntu / Loaner Laptop)

### 1. Copy files to the ROS2 workspace
```bash
cp ollama_publisher.py ~/aisd-vision-Bhagiyaraj99/aisd_hearing/aisd_hearing/
mkdir -p ~/aisd-vision-Bhagiyaraj99/knowledge
cp data/*_clean.* ~/aisd-vision-Bhagiyaraj99/knowledge/
```

### 2. Install dependencies
```bash
pip3 install langchain-community langchain-huggingface langchain-chroma langchain-text-splitters langchain-ollama langchain-classic sentence-transformers
```

### 3. Pull Ollama model
```bash
ollama pull qwen2.5:1.5b
```

### 4. Build the workspace
```bash
cd ~/aisd-vision-Bhagiyaraj99
colcon build
source install/setup.bash
```

### 5. Run all 5 nodes (each in a separate terminal)
```bash
# Terminal 1 — TTS Service
ros2 run aisd_speaking speak

# Terminal 2 — Microphone
ros2 run aisd_hearing recording_publisher 5

# Terminal 3 — Whisper STT
ros2 run aisd_hearing words_publisher

# Terminal 4 — RAG Node
ros2 run aisd_hearing ollama_publisher --ros-args -p model:=qwen2.5:1.5b

# Terminal 5 — Speak Client
ros2 run aisd_hearing speak_client
```

Speak an Olympic history question into the microphone.  
The robot will transcribe it, generate an answer using RAG, and speak it back.
