from ingestion.chunkers import Document
import pickle
import json 
from pathlib import Path

def sparse_search(query: str, n_results: int = 10) -> list[Document]:
    
    with Path("data/bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)
    
    chunks = []
    with Path("data/chunks/chunks.jsonl").open() as f:
        for line in f:
            chunks.append(Document(**json.loads(line)))
    
    tokenised = query.lower().split()

    scores = bm25.get_scores(tokenised)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

    return [chunks[i] for i in top_indices]
    