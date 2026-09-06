"""
ingestion/fetch_repo.py

Fetches GitHub issues from a repo and saves them as JSONL.
Part of Phase 0 — Python Foundations (async, dataclasses, pathlib).
"""

from typing import Literal
from dataclasses import dataclass, asdict
import asyncio, httpx, os, json
from pathlib import Path
import subprocess

def clone_repo(repo: str, dest: Path) -> Path:
    if dest.exists():
        print("idempotent — safe to re-run")
        return dest
    url = f'https://github.com/{repo}.git'
    subprocess.run(["git", "clone", url, str(dest)], check=True) 
    return dest



# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    """
    One GitHub issue normalized to only the fields we need for RAG.

    @dataclass auto-generates __init__, __repr__, __eq__ — no boilerplate.
    Literal["open", "closed"] restricts state to exactly those two values.
    Any other value is caught by the type checker (mypy / IDE) at write time.
    """
    id: int
    number: int
    title: str
    body: str
    state: Literal["open", "closed"]
    url: str
    labels: list[str]

    def to_dict(self) -> dict:
        # asdict() recursively converts the dataclass to a plain dict.
        # Needed for json.dumps() which can't serialize dataclass objects directly.
        return asdict(self)


# ---------------------------------------------------------------------------
# Private helpers (leading _ = not meant to be called from outside this file)
# ---------------------------------------------------------------------------

async def _fetch_page(
    client: httpx.AsyncClient,
    url: str,
    params: dict,
    max_retries: int = 3,
) -> list[dict]:
    """
    Fetch one page of API results with exponential backoff on rate limiting.

    429 = too many requests (rate limited)
    403 = forbidden, often GitHub's secondary rate limit on bursts
    Both mean: slow down, wait, then retry.

    Backoff sequence: 2**0=1s, 2**1=2s, 2**2=4s
    After max_retries exhausted → raise so the caller knows it failed.
    """
    for attempt in range(max_retries):
        res = await client.get(url, params=params)

        if res.status_code in (429, 403):
            # Exponential backoff — each retry waits longer than the last
            wait = 2 ** attempt
            await asyncio.sleep(wait)   # yields control to event loop during the wait
        else:
            # raise_for_status() raises httpx.HTTPStatusError on any 4xx/5xx
            res.raise_for_status()
            return res.json()

    raise RuntimeError(f"Failed after {max_retries} retries")


async def _fetch_all_issues(
    client: httpx.AsyncClient,
    repo: str,
    state: Literal["open", "closed", "all"],
    max_pages: int,
) -> list[dict]:
    """
    Fetch multiple pages of issues concurrently using asyncio.gather.

    Why gather instead of sequential awaits?
    Sequential: fetch page1, wait, fetch page2, wait... → slow
    gather:     fire all page requests simultaneously → ~max_pages times faster

    Pages are independent of each other (page 2 doesn't need page 1's result),
    so concurrent fetching is safe here.
    """
    url = f"https://api.github.com/repos/{repo}/issues"

    # One params dict per page — each page is an independent request
    page_params = [
        {"state": state, "per_page": 100, "page": p}
        for p in range(1, max_pages + 1)
    ]

    # * unpacks the list — gather() needs individual coroutines, not a list
    # Results come back in the same order as inputs (page 1 first, etc.)
    pages = await asyncio.gather(*[
        _fetch_page(client, url, params)
        for params in page_params
    ])

    # pages is a list of lists — flatten into a single list
    return [issue for page in pages for issue in page]


def _parse_issue(req: dict) -> Issue | None:
    """
    Convert a raw GitHub API dict into a typed Issue, or None if it's a PR.

    GitHub's issues endpoint returns PRs mixed in with real issues.
    PRs have a "pull_request" key — we skip them since they're not issues.

    body can be null in the API (issue with no description) → coerce to "".
    labels is a list of dicts like {"name": "bug", ...} → extract just the name.
    html_url is the human-readable GitHub URL (not the API URL stored in "url").
    """
    if "pull_request" in req:
        return None

    return Issue(
        id=req["id"],
        number=req["number"],
        title=req["title"],
        body=req.get("body") or "",   # null-safe: None or "" both become ""
        state=req["state"],
        url=req["html_url"],          # html_url = github.com/..., url = api.github.com/...
        labels=[label["name"] for label in req.get("labels", [])],
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def fetch_issues(
    repo: str = "tiangolo/fastapi",
    max_issues: int = 500,
    output_path: Path = Path("data/raw/issues.jsonl"),
) -> list[Issue]:
    """
    Fetch up to max_issues GitHub issues and save as JSONL.

    Fetches open + closed issues concurrently (two gather calls in parallel),
    parses and deduplicates them, trims to max_issues, and writes to disk.

    JSONL (JSON Lines) = one JSON object per line.
    Chosen over a single JSON array because:
    - Each line is independently readable (grep-friendly)
    - Easy to append new issues without rewriting the whole file
    - Streamable — process line by line without loading everything into memory
    """
    # Read token from environment — never hardcode credentials in source
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise EnvironmentError("GITHUB_TOKEN not set — check your .env file")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",  # pin API version for stability
    }

    # max_issues split evenly between open + closed, 100 per page max
    page_needed = (max_issues // 2) // 100 + 1

    # AsyncClient opens a connection pool shared across all requests.
    # follow_redirects=True: GitHub sometimes redirects repo URLs (301) — follow them.
    # timeout=30: per-request timeout, not total — prevents hanging forever on slow responses.
    async with httpx.AsyncClient(headers=headers, timeout=30, follow_redirects=True) as client:

        # Fetch open and closed issues concurrently — two independent operations
        open_raw, close_raw = await asyncio.gather(
            _fetch_all_issues(client, repo=repo, state="open", max_pages=page_needed),
            _fetch_all_issues(client, repo=repo, state="closed", max_pages=page_needed),
        )

        # Combine both lists and parse into typed Issue objects, dropping PRs (None)
        all_raw = open_raw + close_raw
        issues = [parsed for raw in all_raw if (parsed := _parse_issue(raw)) is not None]

        # Deduplicate by issue number — the same issue can appear in both open + closed
        # if its state changed during our fetch window
        seen: set[int] = set()
        unique_issues: list[Issue] = []
        for issue in issues:
            if issue.number not in seen:
                seen.add(issue.number)
                unique_issues.append(issue)

        # Trim to requested limit
        unique_issues = unique_issues[:max_issues]

        # Write to JSONL — one issue per line
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            for issue in unique_issues:
                f.write(json.dumps(issue.to_dict()) + "\n")

        return unique_issues


# ---------------------------------------------------------------------------
# Run directly: python -m ingestion.fetch_repo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()  # reads .env into os.environ so GITHUB_TOKEN is available

    # asyncio.run() creates the event loop, runs the coroutine, then closes it.
    # It's the bridge between regular sync code (this __main__ block) and async functions.
    issues = asyncio.run(fetch_issues())
    print(f"Fetched {len(issues)} issues")
    print(f"Sample: #{issues[0].number} — {issues[0].title}")