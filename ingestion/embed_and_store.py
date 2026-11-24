from dataclasses import asdict
from ingestion.chunkers import Document
import os
import re
import asyncio
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import chromadb
from rank_bm25 import BM25Okapi
from pathlib import Path
import json
import hashlib

# ---------------------------------------------------------------------------
# Ingestion-time injection scanner
#
# Flags chunks whose content contains known prompt injection patterns before
# they are embedded and stored. We warn rather than drop — dropping causes
# silent data loss and the runtime defenses (XML delimiters, canary token)
# handle poisoned chunks at query time. The warnings surface compromised
# repos or malicious PRs during the ingestion run so they can be investigated.
# ---------------------------------------------------------------------------

_CHUNK_INJECTION_RE = re.compile(
    r"ignore\s+(previous|all|prior|above)(\s+\w+)?\s+(instructions?|prompts?|context|rules?|system)|"
    r"note\s+to\s+(ai|claude|assistant|language\s+model)|"
    r"dear\s+(claude|ai|assistant)|"
    r"attention\s+(language\s+model|ai\s+assistant|claude)|"
    r"PROMPT_OVERRIDE|AI_TASK=|AI_INSTRUCTION:|AI_DIRECTIVE|"
    r"<\/?(system|instruction)\s*>|"
    r"\[INST\]|<<SYS>>",
    re.IGNORECASE,
)


def _flag_suspicious_chunks(documents: list[Document]) -> None:
    """
    Scan all chunks for injection patterns and print warnings.
    Does not drop or modify chunks — runtime defenses remain the enforcement layer.
    """
    flagged = [doc for doc in documents if _CHUNK_INJECTION_RE.search(doc.content)]
    if flagged:
        print(f"\nWARNING: {len(flagged)} chunk(s) contain potential injection patterns:")
        for doc in flagged:
            print(f"  source={doc.source!r}  id={doc.id!r}")

def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()

async def embed_and_store(documents: list[Document]) -> bool:
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
    # Warn about chunks that contain injection patterns before storing them.
    _flag_suspicious_chunks(documents)

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # PersistentClient saves the vector store to disk automatically.
    # get_or_create_collection is idempotent — safe to re-run without duplicating data.
    chroma_client = chromadb.PersistentClient(path="data/chroma")
    collection = chroma_client.get_or_create_collection("devdocs")

    exsisting = collection.get(include=["metadatas"])
    exsisting_hashes = {
        id: meta.get("content_hash", "")
        for id, meta in zip(exsisting["ids"], exsisting["metadatas"])
    }

    new_ids = {doc.id for doc in documents}
    to_embed = [
        doc for doc in documents
        if doc.id not in exsisting_hashes
        or _content_hash(doc.content) != exsisting_hashes[doc.id]
    ]

    anthropic_client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    ctx_batch_size = 20
    contextisation = []
    for i in range(0, len(to_embed), ctx_batch_size):
        batch = to_embed[i: i + ctx_batch_size]
        results = await asyncio.gather(*[_contextualize_chunk(anthropic_client, doc) for doc in batch])
        contextisation.extend(results)

    to_delete = [
        id for id in exsisting_hashes
        if id not in new_ids
    ]

    # Batch size of 100 — OpenAI accepts up to 2048 texts per request.
    # Batching reduces HTTP round trips from N (one per chunk) to N/100,
    # dramatically cutting latency and rate-limit risk. Cost is the same
    # (charged per token, not per request).
    batch_size = 100

    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i: i + batch_size]
        # Enriched content used for embedding — context sentence + original content.
        # Original doc.content is stored in Chroma so the LLM sees clean code/docs.
        enriched_batch = contextisation[i: i + batch_size]

        # Single API call for the whole batch — returns one embedding per input.
        # text-embedding-3-small produces 1536-dimensional vectors.
        # Truncate to 30000 chars (~8000 tokens) — OpenAI's max is 8192 tokens.
        # Character-based truncation is imprecise but avoids adding tiktoken dependency.
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=[content[:30000] for content in enriched_batch],
        )

        # response.data[i].embedding corresponds to batch[i] — order is preserved.
        vectors = [item.embedding for item in response.data]

        # Store in Chroma: ids for deduplication, documents (text) returned at query
        # time as context for the LLM, embeddings for similarity search, metadatas
        # for filtering (e.g. content_type="code" filter in the API).
        collection.upsert(
            ids=[doc.id for doc in batch],
            documents=[doc.content for doc in batch],
            embeddings=vectors,
            metadatas=[
                {
                    **{k: ",".join(v) if isinstance(v, list) else v for k, v in doc.metadata.items()},
                    "type": doc.type,
                    "source": doc.source,
                    "content_hash": _content_hash(doc.content),
                }
                for doc in batch
            ],
        )

    if to_delete:
        collection.delete(ids=to_delete)
        print(f"Deleted {len(to_delete)} removed chunks")

    print(f"Embedded {len(to_embed)} chunks ({len(documents) - len(to_embed)} unchanged, {len(to_delete)} deleted)")

    # BM25 — keyword-based sparse retrieval index.
    # Tokenize by lowercasing and splitting on whitespace. Simple but effective
    # for code identifiers (HTTPException, status_code) that dense search misses.
    # BM25Okapi is the standard variant — ranks by term frequency + inverse doc frequency.
    tokenized = [doc.content.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized)

    # Pickle serializes the BM25 object to bytes so it can be loaded instantly
    # on startup without rebuilding from scratch. "wb" = write binary (pickle output).
    # Serialize the tokenized corpus as JSON instead of pickling the BM25 object.
    # pickle.load() on an untrusted file is an RCE vector — JSON is safe.
    # sparse.py rebuilds BM25Okapi(corpus) at load time; it's deterministic so
    # the resulting index is identical to what pickle would have restored.
    bm25_path = Path("data/bm25_corpus.json")
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with bm25_path.open("w") as f:
        json.dump(tokenized, f)

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

    return len(to_embed) > 0


async def _contextualize_chunk(client: AsyncAnthropic, doc: Document) -> str:
    """Returns enriched content = context sentence + original content."""
    # Build a type-specific prompt
    if doc.type == "code":
        prompt = (
            f"In one sentence, describe what this code chunk does. "
            f"Mention the repo '{doc.metadata.get('repo', '')}', "
            f"file '{doc.source}', and function/class name if present.\n\n"
            f"{doc.content[:1000]}"
        )
    elif doc.type == "doc":
        prompt = (
            f"In one sentence, describe what this documentation section explains. "
            f"The section heading is '{doc.metadata.get('heading', '')}' "
            f"from file '{doc.source}'.\n\n"
            f"{doc.content[:1000]}"
        )
    else: # issue
        prompt = (
            f"In one sentence, describe what this GitHub issue is about. "
            f"Issue number: {doc.metadata.get('issue_number', '')}.\n\n"
            f"{doc.content[:1000]}"
        )

    response = await client.messages.create(
        model = "claude-haiku-4-5-20251001",
        max_tokens = 100,
        messages = [{"role": "user", "content": prompt}],
    )

    context_sentence = response.content[0].text.strip()
    return f"{context_sentence}\n\n{doc.content}"