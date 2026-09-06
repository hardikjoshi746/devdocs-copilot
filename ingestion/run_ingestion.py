from pathlib import Path
from dotenv import load_dotenv
import json
import asyncio

from ingestion.fetch_repo import clone_repo, Issue
from ingestion.chunkers import chunk_python_file, chunk_markdown_file, chunk_issue
from ingestion.embed_and_store import embed_and_store


load_dotenv()

async def main():
    REPO = "tiangolo/fastapi"
    REPO_DIR = Path("data/raw/repo")

    # 1. Clone
    clone_repo(REPO, REPO_DIR)

    # 2. Load issue from JSONL
    issues = []
    with Path("data/raw/issue.jsonl").open() as f:
        for line in f:
            issues.append(Issue(**json.loads(line)))


    # 3. Chunk Python file
    docs = []
    for path in (REPO_DIR / "fastapi").rglob("*.py"):
        docs += chunk_python_file(path.read_text(), str(path))

    # 4. Chunk Markdown file
    for path in (REPO_DIR / "docs").rglob("*.md"):
        docs += chunk_markdown_file(path.read_text(), str(path))

    # 5. Chunk issues
    docs += chunk_issue(issues)

    # 6. Embed and Store
    await embed_and_store(docs)
    print(f"Ingested {len(docs)} chunks")

asyncio.run(main())

