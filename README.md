# DevDocs Copilot

A production-minded RAG system that answers natural-language questions about any codebase. Ingests Python, JavaScript, TypeScript, Java, and Markdown across multiple repos; retrieves with hybrid dense+sparse search; evaluates retrieval quality before generating; caches answers in Redis; and refuses to answer when context is insufficient.

Integrates with **Claude Code via MCP** — developers get codebase-grounded answers without leaving their session, at a fraction of the token cost of Claude Code reading files directly.

**Stack:** OpenAI embeddings · Chroma · BM25 · Cross-encoder reranker · Claude Sonnet 4.6 (generation) · Claude Haiku 4.5 (evaluation) · tree-sitter (multi-language parsing) · Redis (answer cache) · MCP (Claude Code integration)

---

## What Was Built

| Module | File(s) | Description |
|---|---|---|
| Ingestion | `ingestion/fetch_repo.py` | Async GitHub issues fetcher + `clone_repo()` |
| Ingestion | `ingestion/chunkers.py` | tree-sitter chunker (Python, JS, JSX, TS, TSX, Java), heading-aware Markdown chunker, fixed-size fallback, issue chunker |
| Ingestion | `ingestion/embed_and_store.py` | Batched embedding → Chroma (`devdocs` collection); BM25 index serialized to pickle; content-hash diffing skips unchanged chunks |
| Ingestion | `ingestion/run_ingestion.py` | Multi-repo orchestration: REPOS config list → clone → chunk → embed → store; `--pull` flag for incremental updates |
| Retrieval | `retrieval/dense.py` | Chroma cosine similarity search |
| Retrieval | `retrieval/sparse.py` | BM25 keyword search |
| Retrieval | `retrieval/hybrid.py` | Reciprocal Rank Fusion (RRF, k=60) over dense + sparse |
| Retrieval | `retrieval/reranker.py` | Cross-encoder reranking: top-20 → top-5 |
| Query | `query/rewriter.py` | HyDE query rewriter (generates fake answer → embed that) |
| Query | `query/pipeline.py` | Full retrieval pipeline: HyDE → hybrid → rerank |
| Evaluator | `evaluator/retrieval_evaluator.py` | Scores chunks, routes GOOD / EXPAND / ABSTAIN |
| Evaluator | `evaluator/faithfulness_check.py` | Post-generation grounding check; strips ungrounded claims |
| Generation | `generation/answer.py` | Claude Sonnet call with structured output + source citations |
| Cache | `api/cache.py` | Redis answer cache — SHA-256 keyed, 1hr TTL, skips full pipeline on hit |
| Monitoring | `monitoring/tracer.py` | Per-step spans with latency; emits to Arize Phoenix (dev) or X-Ray (prod) |
| Monitoring | `monitoring/logger.py` | Structured JSONL logs locally; CloudWatch Logs in production |
| Eval | `eval/dataset.py` | 50 hand-written Q&A pairs with ground-truth source IDs |
| Eval | `eval/metrics.py` | Recall@5, answer correctness, faithfulness, latency |
| Eval | `eval/run_ablations.py` | Runs all system variants, outputs comparison table |
| API | `api/main.py` | FastAPI service: `POST /query`, `GET /health` |
| MCP | `mcp__server.py` | MCP server exposing `query_codebase` tool to Claude Code |
| Tests | `tests/` | `test_chunkers.py`, `test_retrieval.py`, `test_api.py` |

**Not built (deferred):** `infra/` — EC2 deploy script and S3 sync script.

---

## Eval Results

Evaluated against 50 hand-written Q&A pairs, stratified across 4 question types.

### Overall

| Metric | Score | Target |
|---|---|---|
| Recall@5 | **0.98** | > 0.85 |
| Answer Correctness | **0.776** | > 0.70 |
| Faithfulness | **0.818** | > 0.80 |
| Latency p95 | **26s** | < 30s |
| ABSTAIN rate | **8%** | < 15% |

### Per Category

| Category | N | Recall@5 | Correctness | Faithfulness |
|---|---|---|---|---|
| Factual | 15 | 1.00 | 0.93 | 0.86 |
| Conceptual | 15 | 1.00 | 0.88 | 0.94 |
| Cross-source | 10 | 0.90 | 0.64 | 0.69 |
| Debug | 10 | 1.00 | 0.52 | 0.70 |

**Findings:**
- Retrieval is near-perfect (Recall@5 = 0.98) — HyDE + hybrid + reranking combination works
- Factual and conceptual questions answered well (correctness 0.88–0.93)
- Cross-source and debug categories are weaker — require synthesizing across docs, source, and issues simultaneously
- ABSTAIN rate dropped from 18% → 8% after tuning the retrieval evaluator prompt

Raw results are in `eval/ablation_results.json` and `eval/results_full.json`.

---

## System Design

```
  ┌─────────────────────────────────────────────┐
  │              REST API (FastAPI)              │
  │          POST /query   GET /health           │
  └───────────────────┬─────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │     Redis Cache       │ ◄── cache hit → return instantly ($0 LLM cost)
          └───────────┬───────────┘
                 cache miss
                      │
  ┌───────────────────▼─────────────────────────┐
  │                Query Pipeline               │
  │                                             │
  │  [1] HyDE Rewriter        (Haiku 4.5)       │
  │          │                                  │
  │  [2] Hybrid Retriever                       │
  │       Dense (Chroma) + Sparse (BM25) + RRF  │
  │          │                                  │
  │  [3] Cross-Encoder Reranker                 │
  │       top-20 → top-5                        │
  │          │                                  │
  │  [4] Retrieval Evaluator  (Haiku 4.5)       │◄── GOOD / EXPAND / ABSTAIN
  │          │                                  │
  │  [5] Claude Generator     (Sonnet 4.6)      │
  │       + Source Citations                    │
  │          │                                  │
  │  [6] Faithfulness Check   (Sonnet 4.6)      │
  └───────────────────┬─────────────────────────┘
                      │
            store in Redis cache
                      │
              ┌───────▼────────┐
              │  QueryResponse │
              └────────────────┘
```

### API Contract

```
POST /query
{
  "question": str
}

→ 200
{
  "answer": str,
  "citations": [{ "source": str, "chunk_id": str }],
  "retrieval_quality": "GOOD" | "EXPAND" | "ABSTAIN"
}

→ 503  { "error": "retrieval_quality_too_low" }
```

`retrieval_quality` is surfaced to callers so they know why an answer looks thin. `503` on ABSTAIN means low-confidence results are rejected rather than returned with hallucinated content.

### Retrieval Routing

After scoring the top-5 chunks against the query:

```
avg score >= 0.7  →  GOOD:    proceed to generation
avg score 0.4–0.7 →  EXPAND:  fetch parent chunks, broaden context, re-retrieve once
avg score < 0.4   →  ABSTAIN: return 503, log full trace
```

### Failure Modes

| Failure | Detection | Handling |
|---|---|---|
| Retrieved chunks irrelevant | Retrieval Evaluator score < 0.4 | ABSTAIN → 503, full trace logged |
| Retrieved chunks borderline | Score 0.4–0.7 | EXPAND → fetch parent chunks, retry once |
| LLM returns ungrounded claims | Faithfulness check post-generation | Strip claim; return grounded answer only |
| BM25 index stale | Checksum mismatch on load | Rebuild index |
| Embedding API rate-limited | Exponential backoff, 3 retries | 429 → 503 after retries exhausted |
| Redis unavailable | `_redis is None` guard in `/query` | Graceful degradation — cache skipped, pipeline runs normally |

---

## Design Decisions

### Multi-language chunking with tree-sitter

Replaced `ast.parse` (Python-only) with **tree-sitter**, a single parsing library with grammar packages for each language. The same parent-child chunking structure applies across all languages:

| Extension | Language | Parser |
|---|---|---|
| `.py` | Python | `tree-sitter-python` |
| `.js`, `.jsx` | JavaScript / React | `tree-sitter-javascript` |
| `.ts`, `.tsx` | TypeScript / React | `tree-sitter-typescript` |
| `.java` | Java | `tree-sitter-java` |
| anything else | — | Fixed 100-line chunks, 20-line overlap |

Every code chunk gets `language` and `repo` metadata fields for future query-time filtering.

Chunk IDs are prefixed with the repo name to prevent collisions across repos:
```
job_scrapper::backend/dependencies.py::get_current_user
job_scrapper::frontend/src/context/AuthContext.jsx::AuthContext
```

### Content-hash diffing (incremental re-ingestion)

On every ingestion run, `embed_and_store` fetches the `content_hash` stored in Chroma metadata for every existing chunk and compares it against `SHA-256(chunk.content)` for the current chunks. Only three categories of chunks touch the OpenAI embedding API:

| Category | Action |
|---|---|
| New chunk (ID not in Chroma) | Embed + upsert |
| Changed chunk (hash mismatch) | Re-embed + upsert |
| Deleted chunk (ID gone from new docs) | Delete from Chroma |
| Unchanged chunk (hash match) | Skip entirely — $0 cost |

On a typical commit that touches 2–3 files out of 100, ~95% of chunks are skipped. Re-ingestion cost drops from "embed everything" to "embed only what changed."

To pick up new commits automatically:

```bash
python -m ingestion.run_ingestion --pull   # git pull each repo, then diff + re-embed
python -m ingestion.run_ingestion          # diff only — assumes you already pulled
```

### Multi-repo ingestion

`run_ingestion.py` accepts a `REPOS` config list. Each entry specifies a GitHub slug and which subdirectories to walk (`src_dirs`), allowing fine-grained control over what gets ingested — for example, skipping `node_modules/`, `migrations/`, and test fixtures:

```python
REPOS = [
    {"slug": "hardikjoshi746/job_scrapper", "name": "job_scrapper", "src_dirs": ["backend", "frontend/src"]},
]
```

All repos share a single Chroma collection (`devdocs`) and a single BM25 index.

### Redis answer cache

Every successful query response is stored in Redis keyed by SHA-256 of the normalized question. Subsequent identical queries return instantly with zero LLM API cost.

```
Cache key = SHA-256(question.lower().strip())
Eviction  = LRU (allkeys-lru) — memory-pressure based, not time-based
Flush     = automatic on every re-ingestion run
```

**Why LRU over TTL:** answers go stale when code changes, not when time passes. A 1-hour TTL would evict correct answers for stable functions and keep wrong answers for recently-refactored ones. LRU keeps popular answers as long as they're valid; re-ingestion flushes everything when the codebase actually changes.

If Redis is unavailable at startup, the cache is silently skipped — the pipeline continues to work normally.

### Hybrid retrieval + RRF

Dense retrieval misses exact API names (`get_current_user`, `HTTPException`). BM25 catches those; dense catches paraphrases. RRF fuses both ranked lists using rank position only:

```
RRF score = Σ  1 / (k + rank_i)    k=60
```

### HyDE query rewriting

"How does auth work?" is semantically distant from `def get_current_user(credentials: HTTPAuthorizationCredentials)`. HyDE generates a fake "ideal answer" using Haiku 4.5 and embeds that instead. The fake answer is discarded after retrieval; the original query is used for reranking and generation.

### ABSTAIN over hallucination

`503 retrieval_quality_too_low` is a recoverable, honest failure. A confident wrong answer with citations is not. The evaluator is a pre-generation gate; the faithfulness check is a post-generation filter.

---

## Monitoring

Every `/query` request emits per-step spans and a structured log entry.

### Sample Trace (console)

```
[trace:35bf3ff1] START — How does authentication work in this app?
[35bf3ff1] rewrite          — 3554ms
[35bf3ff1] hybrid_search    — 2332ms
[35bf3ff1] rerank           — 4119ms
[35bf3ff1] evaluate         —  849ms
[35bf3ff1] generate         — 3571ms
[35bf3ff1] check_faithfulness — 2106ms
[trace:35bf3ff1] END
```

Cache hits log a single `cache_hit` event with no pipeline spans.

### Sample Log Entries (`logs/app.jsonl`)

```json
{"timestamp": "2026-09-09T10:12:01.000000+00:00", "event": "cache_hit", "question": "How does authentication work in this app?"}
{"timestamp": "2026-09-09T10:11:30.000000+00:00", "trace_id": "7235958f", "event": "query_complete", "quality": "EXPAND", "question": "How does authentication work in this app?"}
```

### Performance Optimizations

| Optimization | Impact |
|---|---|
| Redis answer cache | Repeat queries: 15–26s → <100ms, $0 LLM cost |
| HyDE response cache (in-memory) | rewrite: 3500ms → 0ms on repeat queries |
| Singleton API clients | Fixed connection exhaustion on long eval runs |
| Cross-encoder lazy load | Moved to first call — eliminates import-time thread deadlock |
| O(1) parent chunk index | EXPAND path lookup: O(n) scan → O(1) dict lookup |

---

## Data Storage & Flow

### Local Directory Structure

```
data/
├── raw/
│   └── repos/
│       └── job_scrapper/          # git clone of each repo
│           ├── backend/           # Python source
│           └── frontend/src/      # React/JS source
│
├── chunks/
│   └── chunks.jsonl               # normalized Document objects (inspectable, re-embeddable)
│
├── chroma/                        # Chroma vector store — collection: "devdocs"
│   ├── chroma.sqlite3
│   └── <collection-uuid>/
│
└── bm25.pkl                       # BM25Okapi index serialized with pickle
```

Issue files land at `data/raw/repos/{name}_issues.jsonl`.

### S3 Structure (target, not yet implemented)

```
s3://{S3_BUCKET}/
├── chunks/chunks.jsonl
├── chroma/chroma.tar.gz
└── bm25.pkl
```

---

## Project Structure

```
adRag/
├── pyproject.toml
├── pytest.ini
├── .env.example
├── .gitignore
│
├── ingestion/
│   ├── fetch_repo.py           # clone repo; fetch GitHub issues via REST API
│   ├── chunkers.py             # tree-sitter chunker, markdown chunker, issue chunker
│   ├── embed_and_store.py      # batch embed → Chroma "devdocs"; tokenize → BM25 pickle
│   └── run_ingestion.py        # REPOS config list → clone → chunk → embed → store
│
├── retrieval/
│   ├── dense.py                # Chroma similarity search
│   ├── sparse.py               # BM25 search (sync, CPU-only)
│   ├── hybrid.py               # RRF fusion of dense + sparse
│   └── reranker.py             # cross-encoder: top-20 → top-5
│
├── query/
│   ├── rewriter.py             # HyDE rewriter (Haiku 4.5)
│   └── pipeline.py             # HyDE → hybrid search → rerank
│
├── evaluator/
│   ├── retrieval_evaluator.py  # score chunks, route GOOD / EXPAND / ABSTAIN
│   └── faithfulness_check.py   # post-generation grounding filter
│
├── generation/
│   └── answer.py               # Claude Sonnet call; structured output + citations
│
├── api/
│   ├── main.py                 # FastAPI: POST /query, GET /health; Redis lifespan
│   └── cache.py                # Redis get/set with SHA-256 key and TTL
│
├── monitoring/
│   ├── tracer.py               # per-step spans; Phoenix (dev) or X-Ray (prod)
│   └── logger.py               # JSONL locally; CloudWatch Logs in production
│
├── eval/
│   ├── dataset.py
│   ├── metrics.py
│   ├── run_ablations.py
│   ├── ablation_results.json
│   └── results_full.json
│
├── tests/
│   ├── conftest.py
│   ├── test_chunkers.py
│   ├── test_retrieval.py
│   └── test_api.py
│
└── logs/
    └── app.jsonl
```

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Embeddings | `text-embedding-3-small` (OpenAI) | 1536-dim vectors; ~$0.01 total for ingestion |
| Vector store | Chroma — collection `devdocs` | `PersistentClient` auto-saves; idempotent on re-run |
| Sparse retrieval | `rank_bm25` | Tokenized by `.lower().split()`; serialized with pickle |
| Code parser | `tree-sitter` + language grammars | Python, JS, JSX, TS, TSX, Java; fixed-size fallback for others |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Free, local, ~80MB; applied to top-20 only |
| Answer cache | Redis | SHA-256 key, 1hr TTL; graceful degradation if unavailable |
| LLM — generation | `claude-sonnet-4-6` | Structured output with source citations |
| LLM — evaluation | `claude-haiku-4-5-20251001` | Per-chunk relevance scoring + faithfulness check |
| Traces (dev) | Arize Phoenix | Local UI at `localhost:6006` |
| Traces (prod) | AWS X-Ray | 100K traces/month free tier |
| Logs | JSONL locally / CloudWatch Logs | 5 GB ingestion/month free tier |
| API | FastAPI + uvicorn | |
| Python | 3.11+ | |

---

## Claude Code Integration (MCP)

`mcp__server.py` exposes a `query_codebase` tool via the Model Context Protocol. Once registered, Claude Code automatically calls it when you ask questions about the codebase — no manual queries, no leaving your session.

### Why this is cheaper than Claude Code reading files

Without the MCP tool, Claude Code answers codebase questions by reading files one by one — each file is thousands of tokens of context on every question.

| Approach | Tokens per question | Est. cost |
|---|---|---|
| Claude Code reading files | ~3,000–8,000 | ~$0.03–0.08 |
| MCP tool (cache miss) | ~1,000–2,000 | ~$0.008–0.015 |
| MCP tool (cache hit) | ~200 | ~$0.001 |

Cache hits compound — a team asking similar questions daily means most answers are free after day 1.

### Setup

```bash
# Register with Claude Code (one-time)
claude mcp add devdocs-copilot -- python /path/to/adRag/mcp__server.py
```

### Usage

Start the API, then open Claude Code in any project:

```bash
uvicorn api.main:app --reload   # must be running
claude                          # start Claude Code
```

Ask naturally — Claude Code calls the tool automatically:

```
> how does authentication work in this app?
> how does the frontend call the backend API?
> where is rate limiting implemented?
```

Claude Code will show "Using tool: query_codebase" and return a cited answer grounded in your actual source code.

### What the tool returns

```
## Authentication in This App

The app uses JWT Bearer token auth passed via the Authorization header...

**Sources (EXPAND):**
- job_scrapper::backend/dependencies.py::get_current_user (data/raw/repos/...)
- job_scrapper::backend/services/security.py::get_user_key (data/raw/repos/...)
```

---

## Getting Started

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Set: OPENAI_API_KEY, ANTHROPIC_API_KEY, GITHUB_TOKEN
# Optional: REDIS_URL (default: redis://localhost:6379), CACHE_TTL_SECONDS (default: 3600)

# 3. Start Redis (macOS)
brew install redis && brew services start redis

# 4. Add repos to REPOS list in ingestion/run_ingestion.py, then ingest
python -m ingestion.run_ingestion

# Re-ingest after new commits (only changed chunks are re-embedded)
python -m ingestion.run_ingestion --pull

# 5. Run the API
uvicorn api.main:app --reload

# 6. Query
curl -s -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question": "How does authentication work in this app?"}'

# 7. Run tests
pytest tests/
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Used for `text-embedding-3-small` |
| `ANTHROPIC_API_KEY` | Yes | Used for generation and evaluation |
| `GITHUB_TOKEN` | Yes | Used by `fetch_repo.py` to pull issues |
| `REPOS` | Yes | JSON array of repos: `[{"slug": "owner/repo", "name": "repo", "src_dirs": [...]}]` |
| `REDIS_URL` | No | Default: `redis://localhost:6379` |
| `TRACER_BACKEND` | No | `phoenix` (default) or `xray` |
| `AWS_REGION` | Prod only | For X-Ray and CloudWatch |
| `S3_BUCKET` | Prod only | For corpus + index storage |

---

## Cost

### Per-query breakdown

| Step | Model | Est. cost |
|---|---|---|
| HyDE rewrite | Haiku 4.5 | ~$0.0001 |
| Query embedding | `text-embedding-3-small` | ~$0.000001 |
| Retrieval eval (5 chunks) | Haiku 4.5 | ~$0.0005 |
| Generation | Sonnet 4.6 | ~$0.003–0.01 |
| Faithfulness check | Sonnet 4.6 | ~$0.002–0.005 |
| **Cache hit** | — | **$0.00** |

Generation + faithfulness = ~85% of per-query cost. Redis cache eliminates this entirely on repeat questions.

### Total project cost (dev/learning scale)

| Item | Cost |
|---|---|
| OpenAI embeddings (~500K tokens ingestion) | ~$0.01 one-time |
| Anthropic API (~1K queries during eval runs) | ~$1–3 total |
| Redis | $0 (local) |
| All infrastructure (EC2, S3, CloudWatch, X-Ray — free tier) | $0 |
| **Total** | **< $5** |

---

## Future Work

**Authorization** — `X-API-Key` header mapped to `allowed_repos` list; Chroma `where={"repo": {"$in": allowed_repos}}` filter enforced at query time. Different devs see only their repos.

**Rate limiting** — Redis-backed `fastapi-limiter`; 10 requests/minute per API key → 429 with `Retry-After`.

**UI** — Simple HTML/JS frontend served at `/`. Text input, response display, `retrieval_quality` badge.

**AWS deployment** — `infra/s3_sync.sh` and `infra/deploy_ec2.sh`. Designed but not yet written.

**Self-RAG** — LLM emits reflection tokens mid-generation to decide when to retrieve. Deferred.

**Fine-tuned embeddings** — Train on `(query, positive chunk, hard negative)` triples mined from real retrieval failures.