"""
retrieval/dense.py

Dense (semantic) retrieval using Chroma vector store.
Embeds the query with the same model used at ingestion time, then finds
the closest chunks by cosine similarity.

Why dense retrieval?
- Catches semantic matches even when exact words differ
- "handle errors" matches "raise HTTPException" because they're semantically close
- Misses exact identifiers (HTTPException, Depends) — sparse retrieval covers that
"""

from ingestion.chunkers import Document
import chromadb
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
_openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


async def dense_search(query: str, n_results: int = 10) -> list[Document]:
    """
    Embed the query and find the top-n most similar chunks in Chroma.

    Must use the same embedding model as ingestion (text-embedding-3-small) —
    if models differ, vectors live in different spaces and similarity is meaningless.
    """
    # Embed the query as a single string — input is a list because the API
    # supports batching, but we only need one embedding here.
    response = await _openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    # response.data is a list — take [0] since we only sent one input
    vector = response.data[0].embedding

    # Load the same Chroma collection written by embed_and_store.py
    # PersistentClient reads from disk — no server needed
    chroma_client = chromadb.PersistentClient(path="data/chroma")
    collection = chroma_client.get_or_create_collection("fastapi")

    # query_embeddings is a list of vectors (one per query)
    # Chroma computes cosine similarity and returns top n_results
    results = collection.query(
        query_embeddings=[vector],
        n_results=n_results
    )

    # Chroma returns parallel nested lists — [0] because we sent one query
    # ids[0], documents[0], metadatas[0] all correspond to the same result set
    docs = []
    for id, content, meta in zip(results["ids"][0], results["documents"][0], results["metadatas"][0]):
        docs.append(Document(
            id=id,
            content=content,
            type=meta["type"],
            source=meta["source"],
            parent_id=None,  # not stored in Chroma metadata — not needed for retrieval
            metadata=meta,
        ))
    return docs