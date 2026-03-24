from fastapi import FastAPI
from pydantic import BaseModel

# Try to import optional LangChain dependencies
try:
    from langchain_ollama import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate
    from vector import retriever
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    retriever = None

app = FastAPI()

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


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
    if not HAS_LANGCHAIN:
        return {"answer": f"Demo mode: Received your question - '{req.question}'"}
    
    if not retriever or not chain:
        return {"error": "Service not available. Please ensure Ollama is running."}

    reviews = retriever.invoke(req.question)

    result = chain.invoke({
        "reviews": reviews,
        "question": req.question
    })

    return {"answer": result}
