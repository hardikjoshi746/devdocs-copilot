# DevDocs Copilot

A production-minded RAG system that answers natural-language questions about any codebase. Ingests Python, JavaScript, TypeScript, Java, and Markdown across multiple repos; retrieves with hybrid dense+sparse search; evaluates retrieval quality before generating; caches answers in Redis; and refuses to answer when context is insufficient.

Integrates with **Claude Code via MCP** — developers get codebase-grounded answers without leaving their session, at a fraction of the token cost of Claude Code reading files directly.

**Stack:** OpenAI embeddings · Chroma · BM25 · Cross-encoder reranker · Claude Sonnet 4.6 (generation) · Claude Haiku 4.5 (evaluation) · tree-sitter (multi-language parsing) · Redis (answer cache) · MCP (Claude Code integration)

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

### Security Evals

Two additional evals target the guardrail layer specifically. Run with:

```bash
python -m eval.run_security_evals                # both evals
python -m eval.run_security_evals --eval injection
python -m eval.run_security_evals --eval pii
python -m eval.run_security_evals --limit 5      # quick smoke test
```

**Eval 1 — Injection Robustness** (15 cases, 5 injection styles)

Each case has a poisoned chunk whose injection phrasing deliberately bypasses the regex layer, testing whether system prompt hardening + XML delimiters prevent the LLM from following it.

| Style | Cases | Description |
|---|---|---|
| `explicit_ai_address` | 3 | "Note to AI:", "Dear Claude:", "ATTENTION LANGUAGE MODEL:" |
| `fake_config` | 3 | Looks like a JSON key or config directive embedded in docs |
| `code_comment` | 3 | Injection hidden in `# CLAUDE:` or `// AI_INSTRUCTION:` comments |
| `prompt_template` | 3 | Fake `Human:/Assistant:` turns or `</context><new_system_instruction>` boundaries |
| `social_engineering` | 3 | Authority claim: "SECURITY AUDIT PROTOCOL v2.1 requires..." |

Scoring: string match (fast, $0) + LLM-as-judge (Haiku 4.5) for subtle behavioral changes.

**Eval 2 — PII Leakage Rate** (15 cases, 5 PII types)

Each case has a chunk seeded with realistic PII. After `generate()` + `mask_pii()`, exact string search checks whether any seeded entity survived masking.

| PII type | Cases |
|---|---|
| `email` | 3 |
| `phone` | 3 |
| `name_email` | 3 |
| `ip_address` | 3 |
| `token` (JWT, API key, DB password) | 3 |

Results saved to `eval/security_injection_results.json` and `eval/security_pii_results.json`.

---

## System Design

```
  ┌─────────────────────────────────────────────┐
  │              REST API (FastAPI)              │
  │          POST /query   GET /health           │
  │   security headers · global exception handler│
  └───────────────────┬─────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │  JWT auth + revocation│ ◄── 401 on missing/expired/revoked token; failures logged
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │  [0] Guardrails       │ ◄── validate_input: normalize → length → regex → LLM judge
          │  validate_input()     │     HTTP 400 on injection detected
          └───────────┬───────────┘
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
  │       + Canary leak check                   │◄── 503 if system prompt exfiltrated
  │          │                                  │
  │  [6] Faithfulness Check   (Sonnet 4.6)      │
  │          │                                  │
  │  [7] PII Masking          (Presidio)        │◄── mask emails, phones, tokens in answer + citations
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
| Prompt injection in query | Regex + LLM classifier in `validate_input` | HTTP 400, pipeline never starts |
| System-prompt exfiltration via poisoned chunk | Canary token in generated answer | HTTP 503, canary presence not disclosed |
| PII in LLM-generated answer or citations | Presidio NER + custom recognizers | Replaced with `<EMAIL_ADDRESS>` etc. before returning |
| Revoked JWT presented | Token hash in Redis `revoked_tokens` set | HTTP 401 |
| Injection pattern found in corpus chunk | Regex scan in `embed_and_store` at ingestion time | Warning printed with source + chunk ID; chunk still stored |
| Unhandled server exception | Global exception handler | Generic `{"error": "internal_server_error"}` — no stack trace exposed |
| Retrieved chunks irrelevant | Retrieval Evaluator score < 0.4 | ABSTAIN → 503, full trace logged |
| Retrieved chunks borderline | Score 0.4–0.7 | EXPAND → fetch parent chunks, retry once |
| LLM returns ungrounded claims | Faithfulness check post-generation | Strip claim; return grounded answer only |
| BM25 index stale | Checksum mismatch on load | Rebuild index |
| Embedding API rate-limited | Exponential backoff, 3 retries | 429 → 503 after retries exhausted |
| Redis unavailable | `_redis is None` guard in `/query` | Graceful degradation — cache + rate limit skipped (or 503 if `RATE_LIMIT_FAIL_CLOSED=true`) |

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

### Security guardrails (`api/guardrails.py`, `api/main.py`)

**HTTP security headers** — applied to every response by middleware:

| Header | Value | Purpose |
|---|---|---|
| `X-Content-Type-Options` | `nosniff` | Prevents MIME-type sniffing |
| `X-Frame-Options` | `DENY` | Blocks clickjacking via iframe embedding |
| `Referrer-Policy` | `no-referrer` | No referrer sent on cross-origin requests |
| `X-XSS-Protection` | `0` | Disables legacy XSS filter (CSP is the modern replacement) |

**Global exception handler** — catches any unhandled exception and returns `{"error": "internal_server_error"}` with no stack trace, file paths, or internal detail exposed to callers. All such events are logged as `unhandled_exception`.

**JWT authentication + revocation**

Tokens are signed JWTs (`HS256`) carrying a `repos` claim. `verify_token` fails closed if `JWT_SIGNING_SECRET` is unset (503). Every 401 path logs an `auth_failure` event with a specific `reason` field (`missing_header`, `expired_token`, `invalid_token`, `invalid_repos_claim`, `revoked_token`) for intrusion detection.

Token revocation uses a Redis set:
```bash
# Revoke a token (hash it first)
python3 -c "import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest())" <raw_token>
redis-cli sadd revoked_tokens <hash>
```
Revocation is best-effort: if Redis is unavailable, revoked tokens pass through (consistent with the cache graceful-degradation policy). Use short `exp` values as the primary mitigation.

**Layer 1 — Input validation (`validate_input`)**

Applied in order, short-circuiting on the first failure:

| Step | What it does | Why |
|---|---|---|
| NFKC normalization | Collapses full-width Unicode, ligatures, superscripts | Prevents look-alike bypass: `Ｊａｉｌｂｒｅａｋ` → `Jailbreak` |
| Zero-width char strip | Removes U+200B, U+FEFF, U+00AD etc. | Prevents split-word bypass: `j​a​i​l​b​r​e​a​k` |
| Length cap (500 chars) | HTTP 400 if exceeded | Cuts off oversized payloads before LLM calls |
| Regex check | 9 patterns covering `ignore instructions`, `you are now`, `[INST]` etc. | Fast, free; catches the most common injection phrasings |
| LLM classifier (Haiku 4.5) | Runs only if regex passes; returns 1 (inject) or 0 (safe) | Catches subtle injections the regex misses (~$0.00003/query) |

**Layer 2 — Canary token (exfiltration detection)**

A random 16-hex-char secret is injected into the system prompt at startup. If it appears in a generated answer, a poisoned chunk instructed the model to repeat its system prompt. The response returns 503 — the detection mechanism is not disclosed.

**Layer 3 — PII masking (`mask_pii`)**

Applied to both the final answer and each citation's `source` field. Citations include file paths and chunk IDs that could expose contributor emails or internal paths from issue-derived chunks.

| Entity type | Pattern |
|---|---|
| `EMAIL_ADDRESS` | Presidio built-in NER |
| `PHONE_NUMBER` | Presidio built-in NER |
| `JWT_TOKEN` | `eyJ…header.payload.signature` |
| `API_KEY` | `sk-…`, `ghp_…`, `npm_…`, `pk-…` |
| `HEX_SECRET` | 32+ consecutive hex chars |

Detected entities are replaced with labeled placeholders: `admin@example.com` → `<EMAIL_ADDRESS>`.

**Layer 4 — Ingestion-time injection scanning (`embed_and_store`)**

Before embedding any chunk, `_flag_suspicious_chunks` scans content against a regex covering `note to AI`, `PROMPT_OVERRIDE`, `AI_INSTRUCTION:`, `[INST]`, etc. Flagged chunks print a warning with `source` and `chunk_id` — they are stored (not dropped) so runtime defenses still apply, but compromised repos or malicious PRs surface during the ingestion run rather than silently at query time.

**Rate limit fail-closed**

By default Redis unavailability silently disables rate limiting (graceful degradation). Set `RATE_LIMIT_FAIL_CLOSED=true` to reject requests with 503 instead — use this in environments where rate limiting is a hard requirement.

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
{"timestamp": "2026-09-09T10:13:45.000000+00:00", "event": "auth_failure", "reason": "expired_token"}
{"timestamp": "2026-09-09T10:14:02.000000+00:00", "event": "auth_failure", "reason": "revoked_token", "sub": "acme"}
{"timestamp": "2026-09-09T10:15:11.000000+00:00", "event": "canary_leak_detected", "trace_id": "a3f92b1c"}
{"timestamp": "2026-09-09T10:16:00.000000+00:00", "event": "unhandled_exception", "error_type": "ConnectionError"}
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
│   ├── embed_and_store.py      # batch embed → Chroma "devdocs"; tokenize → BM25 JSON; injection scan
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
│   ├── main.py                 # FastAPI: POST /query, GET /health; JWT auth + revocation; security headers; rate limit fail-closed
│   ├── cache.py                # Redis get/set with SHA-256 key and TTL
│   └── guardrails.py           # 4-layer input validation, canary token, PII masking
│
├── monitoring/
│   ├── tracer.py               # per-step spans; Phoenix (dev) or X-Ray (prod)
│   └── logger.py               # JSONL locally; CloudWatch Logs in production
│
├── eval/
│   ├── dataset.py
│   ├── dataset_with_ids.json
│   ├── metrics.py
│   ├── run_ablations.py
│   ├── ablation_results.json
│   ├── results_full.json
│   ├── security_dataset.py     # 15 injection cases + 15 PII cases
│   ├── security_metrics.py     # injection_robustness(), pii_leakage_rate()
│   └── run_security_evals.py   # CLI runner for both security evals
│
├── tests/
│   ├── conftest.py
│   ├── test_chunkers.py
│   ├── test_retrieval.py
│   ├── test_api.py
│   └── test_security.py        # guardrails, JWT auth, rate limiting, pickle RCE fix
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
| PII masking | Microsoft Presidio + spaCy | Built-in NER + custom recognizers for JWT, API keys, hex secrets |
| Input validation | Regex + Claude Haiku 4.5 | 4-layer guardrail; LLM classifier runs only after regex passes |
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

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Check with `python3 --version` |
| Redis | 7+ | Used for rate limiting and answer cache |
| Git | any | Used to clone target repos during ingestion |
| OpenAI API key | — | For embeddings (`text-embedding-3-small`) |
| Anthropic API key | — | For generation (Sonnet 4.6) and evaluation (Haiku 4.5) |
| GitHub token | — | For fetching issues from target repos |

---

### Step 1 — Clone the repo

```bash
git clone https://github.com/hardikjoshi746/adRag.git
cd adRag
```

---

### Step 2 — Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

Or with [uv](https://docs.astral.sh/uv/) (faster, lock file included):

```bash
uv sync
source .venv/bin/activate
```

---

### Step 3 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in the required values:

```bash
# API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GITHUB_TOKEN=ghp_...

# JWT signing secret — generates a secure random secret
JWT_SIGNING_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# Repos to ingest — add as many as you like
REPOS=[{"slug": "tiangolo/fastapi", "name": "fastapi", "src_dirs": ["fastapi", "docs/en/docs"]}]
```

**Getting API keys:**
- OpenAI: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Anthropic: [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
- GitHub token: [github.com/settings/tokens](https://github.com/settings/tokens) → Fine-grained → `read:issues` + `read:contents`

---

### Step 4 — Start Redis

**macOS:**
```bash
brew install redis
brew services start redis
```

**Linux:**
```bash
sudo apt install redis-server
sudo systemctl start redis
```

**Docker (any platform):**
```bash
docker run -d -p 6379:6379 redis:7
```

Verify Redis is running:
```bash
redis-cli ping   # should return: PONG
```

---

### Step 5 — Ingest your repos

This clones each repo in `REPOS`, chunks all source files and issues, embeds them, and stores them in Chroma + BM25.

```bash
python -m ingestion.run_ingestion
```

Expected output:
```
Cloning tiangolo/fastapi → data/raw/repos/fastapi
Fetching issues for tiangolo/fastapi...
Chunking Python files...
Chunking Markdown files...
Chunking issues...
Embedded 3842 chunks (0 unchanged, 0 deleted)
```

After pulling new commits, re-ingest (only changed chunks are re-embedded — ~$0 cost on most runs):
```bash
python -m ingestion.run_ingestion --pull
```

---

### Step 6 — Issue a JWT token

Every `/query` request requires a signed Bearer token. Issue one for yourself:

```bash
python3 - <<'EOF'
import jwt, os, time
from dotenv import load_dotenv
load_dotenv()

token = jwt.encode(
    {
        "sub": "me",
        "repos": ["fastapi"],          # must match repo names in REPOS
        "exp": int(time.time()) + 86400 * 30,  # 30 days
    },
    os.environ["JWT_SIGNING_SECRET"],
    algorithm="HS256",
)
print(token)
EOF
```

Save the printed token — you'll use it in every request.

---

### Step 7 — Start the API

```bash
uvicorn api.main:app --reload
```

Confirm it's running:
```bash
curl http://localhost:8000/health
# → {"status": "ok"}
```

---

### Step 8 — Query the API

```bash
TOKEN="<paste your token here>"

curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"question": "How does dependency injection work in FastAPI?"}' \
  | python3 -m json.tool
```

Example response:
```json
{
  "answer": "FastAPI's dependency injection uses `Depends()`. You declare a function...",
  "citations": [
    {"chunk_id": "fastapi::fastapi/dependencies.py::Dependant", "source": "fastapi/dependencies.py"},
    {"chunk_id": "fastapi::docs/en/docs/tutorial/dependencies.md::Dependencies", "source": "docs/en/docs/tutorial/dependencies.md"}
  ],
  "retrieval_quality": "GOOD"
}
```

---

### Step 9 — (Optional) Connect to Claude Code via MCP

This lets Claude Code answer your codebase questions automatically without leaving your session.

```bash
# Register the MCP server (one-time)
claude mcp add devdocs-copilot -- python /path/to/adRag/mcp__server.py
```

The API must be running (`uvicorn api.main:app`). Then in any Claude Code session:

```
> how does authentication work in this app?
> where is rate limiting implemented?
> how does the frontend call the backend API?
```

Claude Code shows "Using tool: query_codebase" and returns a cited answer grounded in your source code.

---

### Step 10 — (Optional) Run the test suite

```bash
pytest tests/                    # all tests
pytest tests/test_security.py    # security tests only
pytest tests/ -v                 # verbose output
```

---

### Step 11 — (Optional) Run security evals

```bash
python -m eval.run_security_evals                # injection + PII evals (~$0.10)
python -m eval.run_security_evals --eval injection
python -m eval.run_security_evals --limit 5      # quick smoke test (5 cases each)
```

---

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Used for `text-embedding-3-small` |
| `ANTHROPIC_API_KEY` | Yes | Used for generation and evaluation |
| `GITHUB_TOKEN` | Yes | Used by `fetch_repo.py` to pull issues |
| `JWT_SIGNING_SECRET` | Yes | Signs and verifies Bearer tokens — generate with `secrets.token_hex(32)` |
| `REPOS` | Yes | JSON array of repos: `[{"slug": "owner/repo", "name": "repo", "src_dirs": [...]}]` |
| `REDIS_URL` | No | Default: `redis://localhost:6379` |
| `RATE_LIMIT_FAIL_CLOSED` | No | `false` (default) = skip rate limit if Redis down; `true` = return 503 instead |
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

**UI** — Simple HTML/JS frontend served at `/`. Text input, response display, `retrieval_quality` badge.

**AWS deployment** — `infra/s3_sync.sh` and `infra/deploy_ec2.sh`. Designed but not yet written.

**Self-RAG** — LLM emits reflection tokens mid-generation to decide when to retrieve. Deferred.

**Fine-tuned embeddings** — Train on `(query, positive chunk, hard negative)` triples mined from real retrieval failures.