from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# Try to import optional dependencies
try:
    from langchain_ollama import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate
    from vector import retriever
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    retriever = None

app = FastAPI()

# Enable CORS - allow configurable origins for Docker/production
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Ollama model if available
model = None
chain = None
if HAS_LANGCHAIN:
    try:
        model = OllamaLLM(model="phi3")
        template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews:
{reviews}

Here is the question to answer:
{question}
"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
    except Exception as e:
        print(f"Warning: Error initializing Ollama model: {e}")


# Request model for the /chat endpoint
class ChatRequest(BaseModel):
    question: str


# Chat endpoint
@app.post("/chat")
def chat(req: ChatRequest):
    """
    Process a chat request and return an answer based on the RAG pipeline.
    
    Args:
        req: ChatRequest containing the user's question
        
    Returns:
        dict with "answer" key containing the model's response
    """
    if not HAS_LANGCHAIN:
        return {"answer": f"Demo mode: Received your question - '{req.question}'. (Full RAG features require LangChain installation)"}
    
    if not retriever:
        return {"error": "Vector database not available. Please check the Chroma database."}
    
    if not chain:
        return {"error": "Service not available. Please ensure Ollama is running with the phi3 model."}

    try:
        # Retrieve relevant reviews from the vector database
        reviews = retriever.invoke(req.question)
        
        # Generate response using the RAG pipeline
        result = chain.invoke({
            "reviews": reviews,
            "question": req.question
        })
        
        return {"answer": result}
    except Exception as e:
        return {"error": f"Error processing request: {str(e)}"}


# Health check endpoint
@app.get("/health")
def health():
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        dict with status
    """
    return {"status": "ok", "rag_enabled": HAS_LANGCHAIN}
