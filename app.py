import streamlit as st
import pandas as pd
import os

# Imports as per your environment
from langchain_community.document_loaders import TextLoader, DirectoryLoader, CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA

class OlympicRAGSystem:
    def __init__(self, data_dir="data/", model_name="qwen2.5:1.5b"):
        self.data_dir = data_dir
        self.model_name = model_name
        self.vector_db_path = "vector_db"
        
        # Initialize Models
        self.llm = OllamaLLM(model=self.model_name)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.qa_chain = None

    def initialize_system(self):
        """Builds or loads the vector database and sets up the RAG chain."""
        
        # 1. PERSISTENCE CHECK (Prevents "Forever" Loading)
        # If the vector_db folder exists and isn't empty, load it instead of re-processing
        if os.path.exists(self.vector_db_path) and os.listdir(self.vector_db_path):
            st.info("Loading existing Knowledge Base from disk...")
            vector_db = Chroma(
                persist_directory=self.vector_db_path, 
                embedding_function=self.embeddings
            )
        else:
            st.warning("No existing database found. Processing files (this may take a while)...")
            all_documents = []

            if not os.path.exists(self.data_dir):
                st.error(f"Data directory '{self.data_dir}' not found!")
                return

            # Load Text Files (Fixes the RuntimeError with encoding)
            txt_loader = DirectoryLoader(
                self.data_dir, 
                glob="*_clean.txt", 
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            all_documents.extend(txt_loader.load())

            # Load CSV Files (Kaggle Dataset)
            csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('_clean.csv')]
            for csv_file in csv_files:
                csv_path = os.path.join(self.data_dir, csv_file)
                loader = CSVLoader(file_path=csv_path, encoding="utf-8")
                all_documents.extend(loader.load())

            if not all_documents:
                st.error("No valid documents found in data directory.")
                return

            # Split Documents into Chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            chunks = text_splitter.split_documents(all_documents)

            # Create and Save Vector Store
            vector_db = Chroma.from_documents(
                documents=chunks, 
                embedding=self.embeddings,
                persist_directory=self.vector_db_path
            )
            st.success("Database created and saved to disk!")

        # 2. Create Retrieval Chain (using langchain_classic)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )

    def ask_question(self, query):
        if not self.qa_chain:
            return "System not initialized.", []
        
        # Using the standard invoke syntax
        response = self.qa_chain.invoke({"query": query})
        return response["result"], response["source_documents"]

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Olympic History AI", layout="wide")
    st.title("🏆 Olympic History RAG Assistant")

    # Initialize the class in session state
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = OlympicRAGSystem()

    # Sidebar for initialization control
    with st.sidebar:
        st.header("System Control")
        if st.button("Initialize / Sync Knowledge Base"):
            with st.spinner("Processing documents and generating embeddings..."):
                st.session_state.rag_system.initialize_system()
            st.rerun()

    # Chat History Setup
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask about Olympic history..."):
        if not st.session_state.rag_system.qa_chain:
            st.error("Please click 'Initialize / Sync Knowledge Base' in the sidebar first!")
        else:
            # User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Assistant Message
            with st.chat_message("assistant"):
                answer, docs = st.session_state.rag_system.ask_question(prompt)
                st.markdown(answer)
                
                # Expandable Source View
                with st.expander("🔍 View Source Documents"):
                    for i, doc in enumerate(docs):
                        src = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        st.write(f"**Source {i+1}:** {src}")
                        st.caption(doc.page_content[:250] + "...")

            st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()