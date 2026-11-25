from anthropic import AsyncAnthropic

async def route(query: str) -> str:
    client = AsyncAnthropic()
    prompt = """
    Categories:
    - code: asks about implementation, functions, classes, or how something works in the code
    - doc: asks about documentation, README, setup, or configuration 
    - issue: asks about bugs, errors, or GitHub issues
    - general: anything else or spans multiple categories
"""
    response = await client.messages.create(
        model = "claude-haiku-4-5-20251001",
        max_tokens= 100,
        messages = [{"role": "user", "content": f"{prompt}\nQuestion: {query}\nReply with one word only."}]
    )
    result = response.content[0].text.strip().lower()
    return result if result in {"code", "doc", "issue", "general"} else "general"
    