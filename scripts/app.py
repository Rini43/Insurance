from fastapi import FastAPI
from pydantic import BaseModel

from search_knowledge_base import search_knowledge_base


app = FastAPI(
    title="Insurance Knowledge Base API",
    version="1.0.0"
)


class SearchRequest(BaseModel):

    question: str
    top_k: int = 5
    company: str | None = None


@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "Insurance Knowledge Base"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/search")
def search(request: SearchRequest):

    results = search_knowledge_base(
        query=request.question,
        top_k=request.top_k,
        company=request.company
    )

    return {
        "question": request.question,
        "results": results
    }
