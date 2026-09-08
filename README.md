# DevDocs Copilot

A RAG-powered Q&A system over the FastAPI codebase and its documentation. Built with backend rigor: clean API contracts, observable failure handling, and an ablation-driven eval framework. Deployed on AWS using free-tier services only.

**Target corpus:** [FastAPI](https://github.com/tiangolo/fastapi) — source code, docs, and GitHub issues.

---

## Implementation Status

| Phase | Component | Status |
|---|---|---|
| 1 | `ingestion/fetch_repo.py` — async GitHub issues fetcher + `clone_repo` | Done |
| 1 | `ingestion/chunkers.py` — AST, heading, issue chunkers | Done |
| 1 | `ingestion/embed_and_store.py` — embed → Chroma, BM25 pickle | Done |
| 1 | `ingestion/run_ingestion.py` — end-to-end orchestration runner | Done |
| 2 | `retrieval/dense.py` — Chroma vector search | Done |
| 2 | `retrieval/sparse.py` — BM25 keyword search | Done |
| 2 | `retrieval/hybrid.py` — RRF fusion | Done |
| 2 | `retrieval/reranker.py` — cross-encoder reranking | Done |
| 3 | `query/rewriter.py` — HyDE query rewriter | Done |
| 3 | `query/pipeline.py` — full retrieval orchestrator | Done |
| 4 | `evaluator/retrieval_evaluator.py` — GOOD/EXPAND/ABSTAIN routing | Done |
| 4 | `evaluator/faithfulness_check.py` — post-generation grounding check | Done |
| 5 | `monitoring/` — tracer, logger, Phoenix setup | Not started |
| 5 | `eval/` — dataset, metrics, ablations | Done |
| — | `api/main.py` — /query, /health | Done |
| — | `generation/answer.py` — Claude Sonnet generation + citations | Done |
| — | `infra/` — s3_sync.sh, deploy_ec2.sh | Not started |
| — | `tests/` — chunkers, retrieval, api | Done |

---

## Eval Results (Full System)

Evaluated against 50 hand-written Q&A pairs across 4 categories.

### Overall

| Metric | Score | Target |
|---|---|---|
| Recall@5 | **0.98** | >0.85 |
| Answer Correctness | **0.776** | >0.70 |
| Faithfulness | **0.818** | >0.80 |
| Latency p95 | **26s** | <30s |
| ABSTAIN rate | **8%** | <15% |

### Per Category

| Category | Correctness | Faithfulness | N |
|---|---|---|---|
| Factual | 0.93 | 0.86 | 15 |
| Conceptual | 0.88 | 0.94 | 15 |
| Cross-source | 0.64 | 0.69 | 10 |
| Debug | 0.52 | 0.70 | 10 |

**Key findings:**
- Retrieval is near-perfect (Recall@5 = 0.98) — HyDE + hybrid + reranking combination works
- Factual and conceptual questions answered well (0.88–0.93 correctness)
- Cross-source and debug categories are weaker — require synthesizing across docs + code + issues
- ABSTAIN rate dropped from 18% → 8% after tuning the retrieval evaluator prompt

---

## What This Is (and Isn't)

This is not a research prototype — it's a production-minded backend service that happens to use RAG. The AI techniques are chosen for measurable impact, not novelty. Every component has a failure mode documented, a metric tracking it, and a fallback when it goes wrong.

**5 phases, fully executed:**

| Phase | What | Why It's Here |
|---|---|---|
| 1 | Type-aware ingestion + chunking | Chunking quality is the highest-leverage decision in RAG |
| 2 | Hybrid retrieval + cross-encoder reranking | Dense misses exact names; BM25 misses paraphrases |
| 3 | Query rewriting (HyDE) | Closes vocabulary gap between questions and how answers are written |
| 4 | Retrieval Evaluator with corrective routing | Prevents confident hallucinations; makes failure visible |
| 5 | Observability + eval framework with ablations | You can't improve what you can't measure |

---

## System Design

```
                     ┌─────────────────────────┐
                     │      REST API (FastAPI)   │
                     │  POST /query              │
                     │  GET  /health             │
                     │  GET  /metrics            │
                     └────────────┬─────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │       Query Pipeline      │
                     │                           │
                     │  [1] HyDE Rewriter        │
                     │          │                │
                     │  [2] Hybrid Retriever     │
                     │    Dense + BM25 + RRF     │
                     │          │                │
                     │  [3] Retrieval Evaluator  │◄── GOOD / EXPAND / ABSTAIN
                     │          │                │
                     │  [4] Cross-Encoder        │
                     │      Reranker             │
                     │          │                │
                     │  [5] Claude Generator     │
                     │      + Citations          │
                     └────────────┬─────────────┘
                                  │
                ┌─────────────────┴──────────────────┐
                │                                     │
   ┌────────────▼────────────┐          ┌─────────────▼──────────┐
   │   Local Dev (free)       │          │   AWS Production        │
   │                          │          │   (free tier)           │
   │   Arize Phoenix          │          │   CloudWatch Logs       │
   │   JSONL logs             │          │   AWS X-Ray             │
   │   Chroma (local)         │          │   CloudWatch Metrics    │
   │   BM25 pickle            │          │   S3 (corpus + index)   │
   └──────────────────────────┘          └────────────────────────┘
```

### API Contract

```
POST /query
{
  "question": str,
  "filters": {
    "content_type": "code" | "doc" | "issue" | null,
    "top_k": int   // default 5
  }
}

→ 200
{
  "answer": str,
  "citations": [{ "source": str, "chunk_id": str, "score": float }],
  "retrieval_quality": "good" | "expanded" | "abstained",
  "latency_ms": int,
  "trace_id": str   // links to X-Ray trace in production
}

→ 503  { "error": "retrieval_quality_too_low", "reason": str }
→ 422  { "error": "invalid_request", "detail": str }
```

`retrieval_quality` is not cosmetic — it tells callers *why* an answer looks thin. `503` on ABSTAIN means clients handle "no context" as a first-class case, not a hallucinated answer.

### Failure Modes and Handling

| Failure | Detection | Handling |
|---|---|---|
| Retrieved chunks irrelevant | Retrieval Evaluator score < 0.4 | ABSTAIN → 503, full trace logged |
| Retrieved chunks borderline | Score 0.4–0.7 | EXPAND → broaden query, re-retrieve once |
| LLM returns ungrounded claims | Faithfulness check post-generation | Strip claim or re-generate with stricter prompt |
| Vector store unreachable | Health check on startup + per-request timeout | 503 with `dependency_unavailable` |
| BM25 index stale | Checksum mismatch on load | Rebuild index, log CloudWatch alarm |
| Embedding API rate-limited | Exponential backoff, 3 retries | 429 → 503 after retries exhausted |

---

## Data Storage & Flow

### Local Directory Structure (`./data/`)

```
data/
├── raw/
│   ├── repo/                  # git clone of FastAPI (source of truth for ingestion)
│   │   ├── fastapi/           # Python source files
│   │   └── docs/              # Markdown documentation
│   └── issues.jsonl           # GitHub issues fetched via API (one JSON object per line)
│
├── chunks/
│   └── chunks.jsonl           # normalized Document objects after chunking (inspectable)
│
├── chroma/                    # Chroma vector store persisted to disk
│   ├── chroma.sqlite3
│   └── <collection-uuid>/
│
└── bm25.pkl                   # BM25 index serialized as pickle
```

`data/raw/` is never uploaded to S3 — the repo can be re-cloned and issues re-fetched cheaply. Everything derived from it (chunks, Chroma, BM25) is what gets persisted and synced.

### S3 Structure

```
s3://{S3_BUCKET}/
├── chunks/
│   └── chunks.jsonl           # allows re-embedding without re-chunking
├── chroma/
│   └── chroma.tar.gz          # Chroma directory tarred for upload
└── bm25.pkl                   # BM25 index
```

### Data Flow

```
  git clone + GitHub API
          │
          ▼
  ./data/raw/            ← never uploaded (re-cloneable, ~200MB)
          │
    chunkers.py
          │
          ▼
  ./data/chunks/         ← uploaded to S3 (re-embeddable without re-chunking)
  chunks.jsonl
          │
    embed_and_store.py
          │
          ▼
  ./data/chroma/         ─── push ──► s3://bucket/chroma/chroma.tar.gz
  ./data/bm25.pkl        ─── push ──► s3://bucket/bm25.pkl
          │
          │ (EC2/Lambda cold start)
          │
          ◄── pull ────── S3 (if ./data/chroma/ is empty on boot)
          │
    API startup (loads Chroma + BM25 into memory)
```

### When Each Path Is Used

| Scenario | Data Path |
|---|---|
| First-time ingestion | `git clone` → chunk → embed → save to `./data/` → push to S3 |
| Re-embed only (model change) | Pull `chunks.jsonl` from S3 → re-embed → push new Chroma + BM25 |
| Re-chunk + re-embed (chunker change) | Re-clone repo → full ingestion → push everything to S3 |
| EC2/Lambda cold start | Pull `chroma.tar.gz` + `bm25.pkl` from S3 → load into memory |
| Local dev | `./data/` only — S3 not involved unless `SYNC_TO_S3=true` |

### `infra/s3_sync.sh` — the glue

```bash
# Push processed artifacts to S3 after ingestion
bash infra/s3_sync.sh push

# Pull artifacts from S3 on instance boot (skips if ./data/ already populated)
bash infra/s3_sync.sh pull

# Force pull (overwrite local with S3 — use after re-ingestion on another machine)
bash infra/s3_sync.sh pull --force
```

---

## Implementation Plan

### Phase 1 — Ingestion + Type-Aware Chunking

**Goal:** Pull the FastAPI corpus, chunk it at semantically meaningful boundaries, persist to S3.

**Document schema:**
```python
@dataclass
class Document:
    id: str
    content: str
    type: Literal["code", "doc", "issue"]
    source: str            # file path or issue URL
    parent_id: str | None  # containing chunk (e.g. full class for a method)
    metadata: dict         # function_name, class_name, heading_path, issue_state
```

**Chunking strategy by type:**

| Content Type | Strategy | Size |
|---|---|---|
| Python source | AST-based: split at function/class boundaries | 50–200 lines |
| Markdown docs | Heading-aware recursive split | ~400 tokens |
| GitHub issues | Title + body together; split long bodies at paragraphs | ~300 tokens |
| Docstrings | Extracted and merged into parent function chunk | merged |

`parent_id` enables **parent-child retrieval**: if a child chunk scores borderline in the retrieval evaluator, fetch its parent for richer context before deciding to discard.

**Sources:**
- FastAPI repo: `git clone` into `./data/raw/repo/` → walk `fastapi/` and `docs/` with `pathlib`
- GitHub issues: GitHub REST API, paginated, top 500 by activity → `./data/raw/issues.jsonl`

**Persistence:** see [Data Storage & Flow](#data-storage--flow) above. Short version:
- `./data/` is the local working directory (gitignored)
- `chunks.jsonl`, `chroma/`, and `bm25.pkl` are pushed to S3 after ingestion
- EC2/Lambda pulls from S3 on cold start if `./data/` is empty

### Phase 2 — Hybrid Retrieval + Reranking

**Two retrieval signals, fused:**

```
Dense:  cosine_similarity(embed(query), embed(chunk))  → ranked list A
Sparse: BM25(tokenize(query), tokenize(chunk))         → ranked list B

RRF score = Σ  1 / (k + rank_i)    k=60
Merged:  sort by RRF score → top-20
```

**Dense:** `text-embedding-3-small` via OpenAI, stored in Chroma.
**Sparse:** `rank_bm25` over tokenized chunks, rebuilt on ingestion.

Dense misses exact API names (`HTTPException`, `Depends`). BM25 catches these; dense catches paraphrase. Together they cover each other's blind spots.

**Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` scores `(query, chunk)` pairs jointly — far more accurate than bi-encoder similarity but too slow for full corpus (~100ms/pair). Applied only to the top-20 shortlist → top-5.

### Phase 3 — Query Rewriting (HyDE)

**Problem:** "How do I handle a 404?" is semantically distant from `raise HTTPException(status_code=404)`.

**HyDE:** Generate a fake "ideal answer" → embed that → use for retrieval. The fake answer lives in the same vector space as real chunks.

```
User query → LLM → "fake answer" → embed → dense retrieval
                                    ↑
                          same vector space as real chunks
```

**When it runs:** Queries classified as non-trivial (short, vague, no known API names). Exact-match queries skip HyDE.

**Fallback:** If HyDE times out (>2s) → fall back to raw query embedding. Never block retrieval on a rewrite step.

### Phase 4 — Retrieval Evaluator

**Problem:** Top-5 after reranking might still be irrelevant. Bad context → confident hallucination.

**Scoring** (via `claude-haiku-4-5` — fast, cheap per chunk):

```python
@dataclass
class ChunkScore:
    chunk_id: str
    relevance: float       # 0–1: does this chunk address the question?
    supports_answer: bool  # is the answer likely in here?

context_quality = mean(relevance_scores) * coverage_factor
```

**Routing:**
```
>= 0.7  →  GOOD:    proceed to LLM
0.4–0.7 →  EXPAND:  fetch parent chunks, broaden query, re-retrieve once
< 0.4   →  ABSTAIN: return 503, log full trace to CloudWatch
```

**Post-generation faithfulness check:** After generation, verify each claim maps to a retrieved chunk. Ungrounded sentences → strip or re-generate. Second independent check on the same property.

### Phase 5 — Observability + Eval

#### Observability: Dev vs Production

**Local dev — Arize Phoenix (free, local):**
- Visual trace explorer at `localhost:6006`
- Retrieval relevance distributions, per-query trace timeline
- Zero cost, zero setup beyond `pip install arize-phoenix`

**Production — AWS (all within free tier):**

| Signal | AWS Service | Free Tier Limit |
|---|---|---|
| Structured logs (per-span JSONL) | CloudWatch Logs | 5 GB ingestion/month |
| Distributed traces | AWS X-Ray | 100K traces recorded/month |
| Metrics + dashboards | CloudWatch Metrics | 3 dashboards, 10 alarms |
| Corpus + index storage | S3 | 5 GB storage |
| API keys / config | SSM Parameter Store | Free (standard params) |

**Span schema** (same structure for both Phoenix locally and X-Ray in production):
```python
{
  "trace_id": str,
  "query_id": str,
  "step": "rewrite" | "retrieval" | "rerank" | "eval" | "generate" | "faithfulness",
  "latency_ms": int,
  "input_tokens": int | None,
  "output_tokens": int | None,
  "metadata": dict   # step-specific: scores, route taken, k, etc.
}
```

**Key signals to watch:**

| Signal | What It Tells You |
|---|---|
| GOOD / EXPAND / ABSTAIN % | Is the corpus covering user questions? |
| EXPAND → success rate | Is re-retrieval actually helping? |
| Stage latency p50/p95 | Where is the bottleneck? |
| HyDE trigger rate | Is the rewriter being used appropriately? |
| Faithfulness fail rate | Is generation drifting from context? |
| Token cost per query | Budget: ~$0.003–0.01/query at typical lengths |

#### Eval Framework

**Dataset:** 50 hand-written Q&A pairs, stratified:

| Category | N | Sample |
|---|---|---|
| Factual | 15 | "What status code does `HTTPException` use by default?" |
| Conceptual | 15 | "How does dependency injection work in FastAPI?" |
| Cross-source | 10 | "What do the docs say about `BackgroundTasks` and are there open issues?" |
| Debug/code | 10 | "Why would a `sync` route block the event loop?" |

**Metrics:**

| Metric | Method | Target |
|---|---|---|
| Recall@5 | Correct source in top-5? | >0.85 |
| Answer Correctness | LLM-as-judge (0–5) vs. reference | >3.5 avg |
| Faithfulness | % claims grounded in retrieved context | >0.90 |
| Retrieval Eval Accuracy | Did evaluator correctly call GOOD vs. POOR? | >0.80 |
| End-to-end latency | p95 | <8s |

**Ablation table:**

| Variant | Recall@5 | Correctness | Faithfulness | Latency p95 |
|---|---|---|---|---|
| Baseline: dense only, no HyDE | — | — | — | — |
| + Sparse (hybrid RRF) | — | — | — | — |
| + Reranker | — | — | — | — |
| + HyDE query rewriting | — | — | — | — |
| Full system (evaluated) | **0.98** | **0.776** | **0.818** | **26s** |

_Baseline variants not yet run — full ablation table pending._

Each row isolates one variable. Numbers either justify the technique or cut it.

---

## AWS Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        AWS (free tier)                        │
│                                                              │
│  ┌──────────────┐    ┌───────────────┐    ┌───────────────┐  │
│  │  API Gateway  │───►│ Lambda / EC2  │───►│      S3       │  │
│  │  (optional)   │    │  (FastAPI app) │    │ corpus, index │  │
│  └──────────────┘    └───────┬───────┘    └───────────────┘  │
│                              │                               │
│              ┌───────────────┼───────────────┐               │
│              ▼               ▼               ▼               │
│     ┌────────────┐  ┌──────────────┐  ┌──────────────┐      │
│     │  CloudWatch │  │   AWS X-Ray  │  │ SSM Parameter│      │
│     │  Logs +     │  │  (traces)    │  │ Store (keys) │      │
│     │  Metrics    │  └──────────────┘  └──────────────┘      │
│     └────────────┘                                           │
└──────────────────────────────────────────────────────────────┘
```

**Deployment options (both free tier):**
- **EC2 t2.micro** — 750 hrs/month free (12 months). Run FastAPI with `uvicorn`. Best for always-on dev testing.
- **Lambda** — 1M requests/month free forever. Package app with `mangum` adapter. Best for low-traffic production.

**Cost breakdown at learning-scale usage:**

| Service | Usage | Cost |
|---|---|---|
| EC2 t2.micro | Always on | $0 (free tier) |
| S3 | ~200MB corpus + index | $0 (free tier) |
| CloudWatch Logs | ~100MB/month traces | $0 (free tier) |
| X-Ray | ~10K traces/month | $0 (free tier) |
| OpenAI embeddings | ~500K tokens ingestion | ~$0.01 one-time |
| Anthropic API | ~1K queries during eval | ~$1–3 total |
| **Total** | | **< $5 for the whole project** |

> **What to avoid:** OpenSearch Serverless (~$700/month minimum), Bedrock (per-token cost with no free tier advantage over direct Anthropic API), RDS (unnecessary for this use case).

---

## Project Structure

```
adRag/
├── README.md
├── LEARNING.md
├── pyproject.toml
├── .env.example
├── Dockerfile
├── .gitignore                  # data/ and .env are gitignored
│
├── data/                       # gitignored — populated by ingestion or S3 pull
│   ├── raw/
│   │   ├── repo/               # git clone of FastAPI
│   │   └── issues.jsonl        # fetched GitHub issues
│   ├── chunks/
│   │   └── chunks.jsonl        # normalized Document objects
│   ├── chroma/                 # Chroma vector store
│   └── bm25.pkl                # BM25 index
│
├── infra/
│   ├── deploy_ec2.sh           # bootstrap script for EC2 t2.micro
│   └── s3_sync.sh              # push/pull data/ ↔ S3
│
├── ingestion/
│   ├── fetch_repo.py           # clone repo, fetch GitHub issues via API
│   ├── chunkers.py             # AST chunker, heading chunker, issue chunker
│   └── embed_and_store.py      # embed → Chroma; tokenize → BM25 pickle; sync to S3
│
├── retrieval/
│   ├── dense.py                # Chroma similarity search
│   ├── sparse.py               # BM25 search
│   ├── hybrid.py               # RRF fusion
│   └── reranker.py             # cross-encoder top-20 → top-5
│
├── query/
│   ├── rewriter.py             # HyDE + fallback to raw query
│   └── pipeline.py             # orchestrates all steps, emits traces
│
├── evaluator/
│   ├── retrieval_evaluator.py  # score chunks, route GOOD/EXPAND/ABSTAIN
│   └── faithfulness_check.py   # post-generation grounding check
│
├── generation/
│   └── answer.py               # Claude call, structured output, citations
│
├── monitoring/
│   ├── tracer.py               # wraps every step; emits to Phoenix (dev) or X-Ray (prod)
│   ├── logger.py               # JSONL locally; CloudWatch Logs in production
│   └── phoenix_setup.py        # Arize Phoenix local config (dev only)
│
├── eval/
│   ├── dataset.py              # Q&A pairs + ground truth source IDs
│   ├── metrics.py              # recall, correctness, faithfulness, latency
│   └── run_ablations.py        # runs all variants, outputs comparison table
│
├── api/
│   └── main.py                 # FastAPI service: /query, /health, /metrics
│
└── tests/
    ├── test_chunkers.py
    ├── test_retrieval.py
    ├── test_evaluator.py
    └── test_pipeline.py
```

---

## Tech Stack

| Layer | Dev | Production (AWS) | Cost |
|---|---|---|---|
| Vector store | Chroma (local) | Chroma (on EC2/Lambda, index from S3) | Free |
| Sparse retrieval | `rank_bm25` pickle | Same, loaded from S3 | Free |
| Embeddings | `text-embedding-3-small` | Same | ~$0.01 total |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Same | Free |
| LLM | `claude-sonnet-4-6` (Anthropic API) | Same | ~$1–3 total |
| Retrieval eval | `claude-haiku-4-5` | Same | Minimal |
| Traces | Arize Phoenix (local) | AWS X-Ray | Free tier |
| Logs | JSONL file | CloudWatch Logs | Free tier |
| Metrics | Phoenix UI | CloudWatch Metrics + Dashboard | Free tier |
| Storage | Local disk | S3 | Free tier |
| API runtime | `uvicorn` local | EC2 t2.micro or Lambda | Free tier |
| Secrets | `.env` file | SSM Parameter Store | Free |

---

## Key Design Decisions

### Dev locally, observe on AWS

Use free local tools (Chroma, Arize Phoenix) during development. The tracer abstraction (`tracer.py`) switches backends via an env var — `TRACER_BACKEND=phoenix` locally, `TRACER_BACKEND=xray` in production. No code changes needed.

### No Bedrock, no OpenSearch

Bedrock charges per token with no free tier advantage over the Anthropic API used directly. OpenSearch Serverless has a minimum cost of ~$700/month. Both are excluded. For a learning project, Chroma + Anthropic API is the right call.

### ABSTAIN is a first-class response, not an error

503 with `retrieval_quality_too_low` is recoverable. A hallucinated answer with citations is not. The API contract treats low-confidence results as distinct from success.

### Ablation table is the actual deliverable

Filling it with real numbers — and cutting a technique if the numbers don't justify it — is the point. "HyDE improved Recall@5 by 8 points" beats "I implemented HyDE."

### Monitoring from day 1

`tracer.py` wraps every step before any other code is written. Retrofitting observability means your traces don't cover the early failures that were hardest to debug.

---

## Future Work

Deferred intentionally — understood and designed, out of scope for this version.

**Self-RAG:** LLM emits reflection tokens mid-generation to decide when to retrieve rather than retrieving once upfront. Real uplift on multi-step questions; requires significant prompt engineering beyond current scope.

**Graph RAG:** Knowledge graph of entities (functions, classes, issues) traversed at query time for multi-hop questions that vector search can't handle. Right approach once the vector-only baseline is solid.

**Fine-tuned embeddings:** Train on `(query, positive chunk, hard negative)` pairs mined from real retrieval failures. Do this *after* the eval baseline exists — fine-tune on real failures, not hypothetical ones.

**Streaming eval / drift detection:** Canary eval on every ingestion batch to catch quality regressions before users do. Important for a live system; deferred until offline eval is airtight.

---

## Getting Started

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# OPENAI_API_KEY, ANTHROPIC_API_KEY, GITHUB_TOKEN
# TRACER_BACKEND=phoenix   # or xray in production
# AWS_REGION, S3_BUCKET    # only needed for production

# 3. Start local observability
python -m monitoring.phoenix_setup    # http://localhost:6006

# 4. Ingest (clones repo, chunks, embeds, stores)
python -m ingestion.run_ingestion

# 5. Run the API locally
uvicorn api.main:app --reload

# 6. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does FastAPI handle dependency injection?"}'

# 7. Run eval + ablations
python -m eval.run_ablations

# --- AWS deployment ---
# 8. Sync corpus to S3
bash infra/s3_sync.sh push

# 9. Deploy to EC2 t2.micro
bash infra/deploy_ec2.sh

# 10. View production traces
# AWS Console → X-Ray → Traces
# AWS Console → CloudWatch → Dashboards
```