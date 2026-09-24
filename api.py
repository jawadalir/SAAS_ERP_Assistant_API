"""
api.py
------
FastAPI wrapper around rag_engine.RAGEngine, so any frontend (PHP, JS,
mobile, etc.) can call the chatbot over HTTP instead of using Streamlit.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000

Then call it, e.g. from PHP/JS, as:
    POST http://your-server:8000/ask
    Body (JSON): {"question": "...", "length": "medium", "mode": "general"}
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_engine import RAGEngine

app = FastAPI(title="FAQ Assistant API", version="1.0")

# Allow your PHP website's domain to call this API from the browser (JS fetch).
# Replace "*" with your actual site domain(s) before going to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. ["https://yourwebsite.com"]
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Loaded once at startup, reused across all requests (same idea as
# Streamlit's @st.cache_resource).
engine = RAGEngine()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    length: str = Field("medium", pattern="^(brief|medium|detailed)$")
    mode: str = Field("general", pattern="^(general|strict)$")


class Source(BaseModel):
    source: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    """Simple check to confirm the API and engine are up."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    try:
        answer, sources = engine.ask(
            payload.question,
            length=payload.length,
            mode=payload.mode,
        )
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")