import asyncio
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

DEVDOCS_URL = "http://localhost:8000"

app = Server("devdocs-copilot")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name= "query_codebase",
            description="Answer questions about the indexed codebase. Use this whenever the developer asks how something works, where something is defined, or how two parts of the code connect.",
            inputSchema={
                "type" : "object",
                "properties" : {
                    "question" : {
                        "type" : "string",
                        "description" : "The natural language question about the codebase"
                    }
                },
                "required" : ["question"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "query_codebase":
        question = arguments["question"]

        async with httpx.AsyncClient(timeout=60) as client:
            try:
                response = await client.post(
                    f"{DEVDOCS_URL}/query",
                    json={"question": question}
                )

                if response.status_code == 503:
                    return [types.TextContent(type="text", text="retrieval quality too low — could not find relevant context for this question.")]

                data = response.json()
                answer = data["answer"]
                citations = data["citations"]
                quality = data["retrieval_quality"]

                citation_text = "\n".join(
                    f"- {c['chunk_id']} ({c['source']})" for c in citations
                )

                result = f"{answer}\n\n**Sources ({quality}):**\n{citation_text}"
                return [types.TextContent(type="text", text=result)]

            except httpx.ConnectError:
                return [types.TextContent(type="text", text="DevDocs Copilot API is not running. Start it with: uvicorn api.main:app --reload")]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())