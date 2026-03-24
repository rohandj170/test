@echo off
cd f:\Users\user\Desktop\RAG
echo Installing dependencies...
.venv\Scripts\pip install -r requirements.txt -q
echo.
echo Starting RAG Chat Application...
echo Backend running on: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
.venv\Scripts\python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
