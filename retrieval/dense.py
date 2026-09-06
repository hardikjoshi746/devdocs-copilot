from ingestion.chunkers import Document
import chromadb
import os
from openai import AsyncOpenAI

async def dense_search(query: str, n_results: int = 10) -> list[Document]:
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # 1. Embed the query
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    vector = response.data[0].embedding

    # 2. Loac Chroma and query
    chroma_client = chromadb.PersistentClient(path="data/chroma")
    collections = chroma_client.get_or_create_collection("fastapi")
    results = collections.query(
        query_embeddings=[vector],
        n_results=n_results
    )

    # 3. Reconstruct Document objects
    # results["ids"] = [["id1", "id2", ...]]  ← note: nested list, take [0]
    # results["documents"] = [["content1", "content2", ...]
    # results["metadatas"] = [["meta1", "meta2", ...]]

    docs = []
    for id, content, meta in zip(results["ids"][0], results["documents"][0], results["metadatas"][0]):
        docs.append(Document(
            id=id,
            content=content,
            type = meta["type"],
            source=meta["source"],
            parent_id= None,
            metadata = meta
        ))
    return docs

