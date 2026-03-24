from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import signal
import sys

def timeout_handler(signum, frame):
    raise TimeoutError("Initialization timeout")

df = pd.read_csv("realistic_restaurant_reviews.csv")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

retriever = None
embeddings = None

# Get Ollama base URL from environment variable, default to localhost
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

try:
    print("Initializing embeddings...")
    # Create embeddings with timeout
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url=OLLAMA_BASE_URL
    )
    
    print("Creating or loading Chroma vector store...")
    vector_store = Chroma(
        collection_name="restaurant_reviews",
        persist_directory=db_location,
        embedding_function=embeddings
    )

    if add_documents:
        print("Adding documents to vector store...")
        documents = []
        ids = []
        
        for i, row in df.iterrows():
            document = Document(
                page_content=row["Title"] + " " + row["Review"],
                metadata={"rating": row["Rating"], "date": row["Date"]},
                id=str(i)
            )
            ids.append(str(i))
            documents.append(document)
        
        vector_store.add_documents(documents=documents, ids=ids)
        print(f"Added {len(documents)} documents to vector store")
        
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    print("Vector store initialized successfully!")
    
except Exception as e:
    print(f"Warning: Error initializing vector store: {e}")
    print("Vector database will not be available until Ollama is running")