from ingestion.chunkers import Document
from retrieval.dense import dense_search
from retrieval.sparse import sparse_search
from collections import defaultdict

async def hybrid_search(query: str, n_results: int = 10) -> list[Document]:
    dense_result = await dense_search(query, n_results=20)
    sparse_result = sparse_search(query, n_results=20)

    scores = defaultdict(float)

    doc_map = {}

    for rank, doc in enumerate(dense_result, start=1):
        scores[doc.id] += 1/(60 + rank)
        doc_map[doc.id] = doc
        
    for rank, doc in enumerate(sparse_result, start=1):
        scores[doc.id] += 1/(60 + rank) 
        doc_map[doc.id] = doc

    sorted_ids = sorted(scores, key=lambda id: scores[id], reverse=True)[:n_results]
    return [doc_map[id] for id in sorted_ids]

