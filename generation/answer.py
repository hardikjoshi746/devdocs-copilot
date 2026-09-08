from ingestion.chunkers import Document
from anthropic import AsyncAnthropic
import os
from dotenv import load_dotenv

load_dotenv()
_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

async def generate(query: str, docs: list[Document]) -> dict:
    context = "\n\n".join(f"[{doc.id}]\n{doc.content}" for doc in docs)
    prompt = (
        f"Answer the question using only the provided context.\n"
        f"Be concise. Cite the source id when you use information from a chunk. \n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}"
    )
    response = await _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer" : response.content[0].text,
        "citations" : [{"source" : doc.source, "chunk_id" : doc.id} for doc in docs]
    }
    