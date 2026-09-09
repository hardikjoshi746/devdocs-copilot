from pathlib import Path
from dotenv import load_dotenv
import json
import os
import asyncio
import argparse

from redis.asyncio import Redis
from ingestion.fetch_repo import clone_repo, pull_repo, fetch_issues, Issue
from ingestion.chunkers import chunk_code_file, chunk_markdown_file, chunk_issue, detect_language
from ingestion.embed_and_store import embed_and_store

load_dotenv()

# REPOS is read from .env as a JSON array so you can add repos without editing source code.
# Format: [{"slug": "owner/repo", "name": "repo", "src_dirs": ["dir1", "dir2"]}]
_repos_env = os.environ.get("REPOS")
if not _repos_env:
    raise EnvironmentError("REPOS not set in .env — add a JSON array of repos to ingest")
REPOS: list[dict] = json.loads(_repos_env)

CODE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".java"}


async def ingest_repo(slug: str, name: str, src_dirs: list[str], base_dir: Path, pull: bool = False) -> list:
    repo_dir = base_dir / name

    # 1. Clone (idempotent — skips if already present), then optionally pull latest
    clone_repo(slug, repo_dir)
    if pull:
        print(f"  [{name}] pulling latest...")
        pull_repo(repo_dir)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--pull", action="store_true", help="git pull each repo before re-ingesting")
    args = parser.parse_args()

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
            pull=args.pull,
        )

    print(f"\nTotal: {len(all_docs)} chunks")
    changed = await embed_and_store(all_docs)

    # Flush Redis cache — cached answers are now stale since the corpus changed
    if changed:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        try:
            redis = Redis.from_url(redis_url)
            await redis.flushdb()
            await redis.aclose()
            print("Cache flushed.")
        except Exception:
            print("Redis not reachable — cache not flushed (start Redis before querying).")

    print("Done.")

asyncio.run(main())