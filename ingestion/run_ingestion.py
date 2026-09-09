from pathlib import Path
from dotenv import load_dotenv
import json
import asyncio

from ingestion.fetch_repo import clone_repo, fetch_issues, Issue
from ingestion.chunkers import chunk_code_file, chunk_markdown_file, chunk_issue, detect_language
from ingestion.embed_and_store import embed_and_store

load_dotenv()

REPOS = [
    {
        "slug": "hardikjoshi746/job_scrapper",
        "name": "job_scrapper",
        "src_dirs": ["backend", "frontend/src"],
    },
]

CODE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".java"}


async def ingest_repo(slug: str, name: str, src_dirs: list[str], base_dir: Path) -> list:
    repo_dir = base_dir / name

    # 1. Clone (idempotent — skips if already present)
    clone_repo(slug, repo_dir)

    # 2. Fetch GitHub issues (idempotent — skips if file already exists)
    issues_path = base_dir / f"{name}_issues.jsonl"
    if not issues_path.exists():
        print(f"  [{name}] fetching issues...")
        await fetch_issues(repo=slug, max_issues=500, output_path=issues_path)

    issues = []
    with issues_path.open() as f:
        for line in f:
            issues.append(Issue(**json.loads(line)))

    docs = []

    # 3a. Always pick up markdown files from the repo root (e.g. README.md)
    for path in repo_dir.glob("*.md"):
        try:
            docs += chunk_markdown_file(path.read_text(errors="replace"), str(path))
        except Exception as e:
            print(f"  [{name}] skip {path.name}: {e}")

    # 3b. Chunk source files inside the specified dirs
    dirs_to_walk = [repo_dir / d for d in src_dirs] if src_dirs else [repo_dir]
    for walk_dir in dirs_to_walk:
        if not walk_dir.exists():
            print(f"  [{name}] WARNING: {walk_dir} not found, skipping")
            continue

        for path in walk_dir.rglob("*"):
            if not path.is_file():
                continue

            ext = path.suffix.lower()

            if ext == ".md":
                try:
                    docs += chunk_markdown_file(path.read_text(errors="replace"), str(path))
                except Exception as e:
                    print(f"  [{name}] skip {path.name}: {e}")

            elif ext in CODE_EXTENSIONS:
                try:
                    docs += chunk_code_file(path.read_text(errors="replace"), str(path), repo=name)
                except Exception as e:
                    print(f"  [{name}] skip {path.name}: {e}")

    # 4. Chunk issues
    docs += chunk_issue(issues)

    print(f"  [{name}] {len(docs)} chunks  ({len(issues)} issues)")
    return docs


async def main():
    base_dir = Path("data/raw/repos")
    base_dir.mkdir(parents=True, exist_ok=True)

    all_docs = []
    for repo in REPOS:
        print(f"Ingesting {repo['slug']}...")
        all_docs += await ingest_repo(
            slug=repo["slug"],
            name=repo["name"],
            src_dirs=repo["src_dirs"],
            base_dir=base_dir,
        )

    print(f"\nTotal: {len(all_docs)} chunks")
    await embed_and_store(all_docs)
    print("Done.")

asyncio.run(main())