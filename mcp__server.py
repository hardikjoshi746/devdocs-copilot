import httpx
from mcp.server.mcpserver import MCPServer

DEVDOCS_URL = "http://localhost:8000"

mcp = MCPServer("devdocs-copilot")


@mcp.tool()
async def query_codebase(question: str) -> str:
    """Answer questions about the indexed codebase. Use this whenever the developer asks how something works, where something is defined, or how two parts of the code connect."""
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            response = await client.post(
                f"{DEVDOCS_URL}/query",
                json={"question": question}
            )

            if response.status_code == 503:
                return "retrieval quality too low — could not find relevant context for this question."

            data = response.json()
            answer = data["answer"]
            citations = data["citations"]
            quality = data["retrieval_quality"]

            citation_text = "\n".join(
                f"- {c['chunk_id']} ({c['source']})" for c in citations
            )

            return f"{answer}\n\n**Sources ({quality}):**\n{citation_text}"

        except httpx.ConnectError:
            return "DevDocs Copilot API is not running. Start it with: uvicorn api.main:app --reload"


if __name__ == "__main__":
    mcp.run()
