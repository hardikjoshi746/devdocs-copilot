# DevDocs Copilot

A production-minded RAG system that answers natural-language questions about any codebase. Ingests Python, JavaScript, TypeScript, Java, and Markdown across multiple repos; retrieves with hybrid dense+sparse search; evaluates retrieval quality before generating; and refuses to answer when context is insufficient.

**Stack:** OpenAI embeddings · Chroma · BM25 · Cross-encoder reranker · Claude Sonnet 4.6 (generation) · Claude Haiku 4.5 (evaluation) · tree-sitter (multi-language parsing)

---

## What Was Built

| Module | File(s) | Description |
|---|---|---|
| Ingestion | `ingestion/fetch_repo.py` | Async GitHub issues fetcher + `clone_repo()` |
| Ingestion | `ingestion/chunkers.py` | tree-sitter chunker (Python, JS, JSX, TS, TSX, Java), heading-aware Markdown chunker, fixed-size fallback, issue chunker |
| Ingestion | `ingestion/embed_and_store.py` | Batched embedding → Chroma; BM25 index serialized to pickle |
| Ingestion | `ingestion/run_ingestion.py` | End-to-end orchestration runner |
| Retrieval | `retrieval/dense.py` | Chroma cosine similarity search |
| Retrieval | `retrieval/sparse.py` | BM25 keyword search |
| Retrieval | `retrieval/hybrid.py` | Reciprocal Rank Fusion (RRF, k=60) over dense + sparse |
| Retrieval | `retrieval/reranker.py` | Cross-encoder reranking: top-20 → top-5 |
| Query | `query/rewriter.py` | HyDE query rewriter (generates fake answer → embed that) |
| Query | `query/pipeline.py` | Full retrieval pipeline: HyDE → hybrid → rerank |
| Evaluator | `evaluator/retrieval_evaluator.py` | Scores chunks, routes GOOD / EXPAND / ABSTAIN |
| Evaluator | `evaluator/faithfulness_check.py` | Post-generation grounding check; strips ungrounded claims |
| Generation | `generation/answer.py` | Claude Sonnet call with structured output + source citations |
| Monitoring | `monitoring/tracer.py` | Per-step spans with latency; emits to Arize Phoenix (dev) or X-Ray (prod) |
| Monitoring | `monitoring/logger.py` | Structured JSONL logs locally; CloudWatch Logs in production |
| Eval | `eval/dataset.py` | 50 hand-written Q&A pairs with ground-truth source IDs |
| Eval | `eval/metrics.py` | Recall@5, answer correctness, faithfulness, latency |
| Eval | `eval/run_ablations.py` | Runs all system variants, outputs comparison table |
| API | `api/main.py` | FastAPI service: `POST /query`, `GET /health` |
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
                     ┌─────────────────────────┐
                     │      REST API (FastAPI)   │
                     │  POST /query              │
                     │  GET  /health             │
                     └────────────┬─────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │       Query Pipeline      │
                     │                           │
                     │  [1] HyDE Rewriter        │
                     │       (Haiku 4.5)         │
                     │          │                │
                     │  [2] Hybrid Retriever     │
                     │    Dense + BM25 + RRF     │
                     │          │                │
                     │  [3] Cross-Encoder        │
                     │      Reranker             │
                     │       top-20 → top-5      │
                     │          │                │
                     │  [4] Retrieval Evaluator  │◄── GOOD / EXPAND / ABSTAIN
                     │       (Haiku 4.5)         │
                     │          │                │
                     │  [5] Claude Generator     │
                     │      (Sonnet 4.6)         │
                     │      + Citations          │
                     │          │                │
                     │  [6] Faithfulness Check   │
                     └────────────┬─────────────┘
                                  │
                ┌─────────────────┴──────────────────┐
                │                                     │
   ┌────────────▼────────────┐          ┌─────────────▼──────────┐
   │   Local Dev              │          │   AWS Production        │
   │                          │          │   (free tier)           │
   │   Arize Phoenix          │          │   CloudWatch Logs       │
   │   JSONL logs             │          │   AWS X-Ray             │
   │   Chroma (local disk)    │          │   CloudWatch Metrics    │
   │   BM25 pickle            │          │   S3 (corpus + index)   │
   └──────────────────────────┘          └────────────────────────┘
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

---

## Design Decisions

### Type-aware chunking

Three distinct chunkers for three content types:

- **Python source** — AST-based (`ast.parse`), splits at function/class boundaries. Never cuts mid-function. Methods are child chunks of their parent class, enabling the EXPAND path to fetch the full class when a method chunk scores borderline.
- **Markdown docs** — heading-boundary split, skips `#` inside fenced code blocks. Preserves section context.
- **GitHub issues** — one Document per issue (title + body), enabling keyword search over real user bug reports.

```python
@dataclass
class Document:
    id: str            # filepath::ClassName::method — unique across corpus
    content: str
    type: Literal["code", "doc", "issue"]
    source: str        # file path or issue URL
    parent_id: str | None  # class chunk for method chunks; None otherwise
    metadata: dict
```

### Hybrid retrieval + RRF

Dense retrieval misses exact API names (`HTTPException`, `Depends`). BM25 catches those; dense catches paraphrases. RRF fuses both ranked lists using rank position only (ignoring incompatible raw scores):

```
RRF score = Σ  1 / (k + rank_i)    k=60
```

### HyDE query rewriting

"How do I handle a 404?" is semantically distant from `raise HTTPException(status_code=404)`. HyDE generates a fake "ideal answer" using Haiku 4.5, embeds that instead of the question. The fake answer lives in the same vector space as real chunks — retrieval improves. The fake answer is discarded after retrieval; the original query is used for reranking and generation.

### ABSTAIN over hallucination

`503 retrieval_quality_too_low` is a recoverable, honest failure. A confident wrong answer with citations is not. The evaluator is a pre-generation gate; the faithfulness check is a post-generation filter. Both independent checks reduce ungrounded output.

### Monitoring from the start

`tracer.py` wraps every pipeline step. The `TRACER_BACKEND` env var switches between Arize Phoenix (local) and AWS X-Ray (production) with no code changes. Latency, token counts, and routing decisions are captured per span.

---

## Monitoring

Every `/query` request emits per-step spans and a structured log entry.

### Sample Trace (console)

```
[trace:35bf3ff1] START — What parameters does HTTPException accept?
[35bf3ff1] rewrite          — 3554ms
[35bf3ff1] hybrid_search    — 2332ms
[35bf3ff1] rerank           — 4119ms
[35bf3ff1] evaluate         —  849ms
[35bf3ff1] generate         — 3571ms
[35bf3ff1] check_faithfulness — 2106ms
[trace:35bf3ff1] END
```

### Sample Log Entry (`logs/app.jsonl`)

```json
{"timestamp": "2026-09-08T20:42:47.518793+00:00", "trace_id": "7235958f", "event": "query_complete", "quality": "EXPAND", "question": "What parameters does HTTPException accept?"}
```

### Performance Optimizations Applied

| Optimization | Impact |
|---|---|
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
│   ├── repo/                  # git clone of FastAPI
│   │   ├── fastapi/           # Python source files
│   │   └── docs/en/docs/      # English Markdown docs
│   └── issues.jsonl           # fetched GitHub issues (one JSON object per line)
│
├── chunks/
│   └── chunks.jsonl           # normalized Document objects (inspectable, re-embeddable)
│
├── chroma/                    # Chroma vector store persisted to disk
│   ├── chroma.sqlite3
│   └── <collection-uuid>/
│
└── bm25.pkl                   # BM25Okapi index serialized with pickle
```

`data/` is gitignored. `data/raw/` is never uploaded — the repo can be re-cloned and issues re-fetched. `chunks.jsonl`, `chroma/`, and `bm25.pkl` are the artifacts that need to be stored or synced.

### S3 Structure (target, not yet implemented)

```
s3://{S3_BUCKET}/
├── chunks/chunks.jsonl        # allows re-embedding without re-chunking
├── chroma/chroma.tar.gz       # Chroma directory tarred for upload
└── bm25.pkl
```

---

## Project Structure

```
adRag/
├── pyproject.toml
├── pytest.ini
├── .env.example
├── .gitignore                  # data/ and .env gitignored
│
├── ingestion/
│   ├── fetch_repo.py           # clone repo; fetch GitHub issues via REST API
│   ├── chunkers.py             # AST chunker, heading chunker, issue chunker
│   ├── embed_and_store.py      # batch embed → Chroma; tokenize → BM25 pickle
│   └── run_ingestion.py        # orchestration: clone → chunk → embed → store
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
├── monitoring/
│   ├── tracer.py               # per-step spans; Phoenix (dev) or X-Ray (prod)
│   └── logger.py               # JSONL locally; CloudWatch Logs in production
│
├── eval/
│   ├── dataset.py              # 50 Q&A pairs with ground-truth source IDs
│   ├── dataset_with_ids.json   # dataset serialized with chunk IDs
│   ├── metrics.py              # recall@5, correctness, faithfulness, latency
│   ├── run_ablations.py        # runs all variants, writes comparison JSON
│   ├── ablation_results.json   # full ablation output
│   └── results_full.json       # per-question detailed results
│
├── api/
│   └── main.py                 # FastAPI: POST /query, GET /health
│
├── tests/
│   ├── conftest.py
│   ├── test_chunkers.py
│   ├── test_retrieval.py
│   └── test_api.py
│
└── logs/
    └── app.jsonl               # runtime structured logs
```

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Embeddings | `text-embedding-3-small` (OpenAI) | 1536-dim vectors; ~$0.01 total for ingestion |
| Vector store | Chroma (persistent, local) | `PersistentClient` auto-saves; idempotent on re-run |
| Sparse retrieval | `rank_bm25` | Tokenized by `.lower().split()`; serialized with pickle |
| Code parser | `tree-sitter` + language grammars | Python, JS, JSX, TS, TSX, Java; fixed-size fallback for others |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Free, local, ~80MB; applied to top-20 only |
| LLM — generation | `claude-sonnet-4-6` | Structured output with source citations |
| LLM — evaluation | `claude-haiku-4-5-20251001` | Per-chunk relevance scoring + faithfulness check |
| Traces (dev) | Arize Phoenix | Local UI at `localhost:6006` |
| Traces (prod) | AWS X-Ray | 100K traces/month free tier |
| Logs | JSONL locally / CloudWatch Logs | 5 GB ingestion/month free tier |
| API | FastAPI + uvicorn | |
| Python | 3.11+ | |

---

## Getting Started

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Set: OPENAI_API_KEY, ANTHROPIC_API_KEY, GITHUB_TOKEN
# Optional: TRACER_BACKEND=phoenix  (default; use xray in production)

# 3. Ingest (clones FastAPI repo, chunks, embeds, stores — ~10 min first run)
python -m ingestion.run_ingestion

# 4. Run the API
uvicorn api.main:app --reload

# 5. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does FastAPI handle dependency injection?"}'

# 6. Run eval suite
python -m eval.run_ablations

# 7. Run tests
pytest tests/
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Used for `text-embedding-3-small` |
| `ANTHROPIC_API_KEY` | Yes | Used for generation and evaluation |
| `GITHUB_TOKEN` | Yes | Used by `fetch_repo.py` to pull issues |
| `TRACER_BACKEND` | No | `phoenix` (default) or `xray` |
| `AWS_REGION` | Prod only | For X-Ray and CloudWatch |
| `S3_BUCKET` | Prod only | For corpus + index storage |

---

## Cost

Total cost to build and evaluate this project at dev/learning scale:

| Item | Cost |
|---|---|
| OpenAI embeddings (~500K tokens ingestion) | ~$0.01 one-time |
| Anthropic API (~1K queries during eval runs) | ~$1–3 total |
| All infrastructure (EC2, S3, CloudWatch, X-Ray — free tier) | $0 |
| **Total** | **< $5** |

---

## Future Work

**Multi-repo ingestion** — `REPOS` config list with per-repo `src_dirs` scoping; single Chroma collection (`devdocs`) with `repo` + `language` metadata fields for filtering. Auth and query-time filtering deferred.

**UI** — Simple HTML/JS frontend served by FastAPI at `/`. Text input, response display, retrieval quality badge.

**AWS deployment** — `infra/s3_sync.sh` (push/pull data artifacts) and `infra/deploy_ec2.sh` (bootstrap EC2 t2.micro). Both are designed but not yet written.

**Self-RAG** — LLM emits reflection tokens mid-generation to decide when to retrieve rather than retrieving once upfront. Real uplift on multi-step questions; deferred.

**Fine-tuned embeddings** — Train on `(query, positive chunk, hard negative)` triples mined from real retrieval failures. Do this after the eval baseline is solid, not before.