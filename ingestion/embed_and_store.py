from dataclasses import asdict
from ingestion.chunkers import Document
import os
from openai import AsyncOpenAI
import chromadb
from rank_bm25 import BM25Okapi
from pathlib import Path
import pickle
import json


async def embed_and_store(documents: list[Document]) -> None:
    """
    Embed all Document chunks and persist them in three forms:
      1. Chroma vector store  — for dense (semantic) retrieval
      2. BM25 index (pickle)  — for sparse (keyword) retrieval
      3. chunks.jsonl         — raw chunks, so we can re-embed later without re-chunking

    Why three outputs?
    - Chroma + BM25 together enable hybrid retrieval (Phase 2). Dense alone misses
      exact API names like HTTPException; BM25 alone misses paraphrases. Both together
      cover each other's blind spots and are fused with RRF at query time.
    - chunks.jsonl is the checkpoint: if we change the embedding model we can
      re-embed from here without re-running the chunkers (expensive AST parsing).
    """
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # PersistentClient saves the vector store to disk automatically.
    # get_or_create_collection is idempotent — safe to re-run without duplicating data.
    chroma_client = chromadb.PersistentClient(path="data/chroma")
    collection = chroma_client.get_or_create_collection("fastapi")

    # Batch size of 100 — OpenAI accepts up to 2048 texts per request.
    # Batching reduces HTTP round trips from N (one per chunk) to N/100,
    # dramatically cutting latency and rate-limit risk. Cost is the same
    # (charged per token, not per request).
    batch_size = 100

    for i in range(0, len(documents), batch_size):
        batch = documents[i: i + batch_size]

        # Single API call for the whole batch — returns one embedding per input.
        # text-embedding-3-small produces 1536-dimensional vectors.
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=[doc.content for doc in batch],
        )

        # response.data[i].embedding corresponds to batch[i] — order is preserved.
        vectors = [item.embedding for item in response.data]

        # Store in Chroma: ids for deduplication, documents (text) returned at query
        # time as context for the LLM, embeddings for similarity search, metadatas
        # for filtering (e.g. content_type="code" filter in the API).
        collection.add(
            ids=[doc.id for doc in batch],
            documents=[doc.content for doc in batch],
            embeddings=vectors,
            metadatas=[{**doc.metadata, "type": doc.type, "source": doc.source} for doc in batch],
        )

    # BM25 — keyword-based sparse retrieval index.
    # Tokenize by lowercasing and splitting on whitespace. Simple but effective
    # for code identifiers (HTTPException, status_code) that dense search misses.
    # BM25Okapi is the standard variant — ranks by term frequency + inverse doc frequency.
    tokenized = [doc.content.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized)

    # Pickle serializes the BM25 object to bytes so it can be loaded instantly
    # on startup without rebuilding from scratch. "wb" = write binary (pickle output).
    bm25_path = Path("data/bm25.pkl")
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with bm25_path.open("wb") as f:
        pickle.dump(bm25, f)

    # Save raw chunks as JSONL (one JSON object per line).
    # JSONL chosen over a single JSON array because:
    # - Each line is independently readable (grep-friendly)
    # - Re-embeddable: pull this file from S3, skip re-chunking entirely
    # asdict() converts the Document dataclass to a plain dict for json.dumps().
    chunk_path = Path("data/chunks/chunks.jsonl")
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    with chunk_path.open("w") as f:
        for doc in documents:
            f.write(json.dumps(asdict(doc)) + "\n")



