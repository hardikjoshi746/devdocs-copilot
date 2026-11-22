"""
monitoring/tracer.py

Real OpenTelemetry tracing — replaces the old print() stub.

Concepts:
  TracerProvider  — global engine, configured once at import time
  Tracer          — factory that creates spans, scoped to "adrag"
  Span            — one unit of work: has a name, start time, end time, attributes
  Trace           — a tree of spans representing one full request

Why OTel over print():
  - Spans are nested automatically via context — no manual trace_id threading needed
  - Every span records start/end time, so latency per step is always captured
  - Attributes (query, quality, num_results) are queryable in Phoenix UI
  - BatchSpanProcessor sends spans async in background — zero latency impact on requests

Exporter:
  Dev  → Arize Phoenix running locally at localhost:6006 (OTLP HTTP)
  Prod → set PHOENIX_COLLECTOR_ENDPOINT to your OTLP collector URL (e.g. AWS X-Ray, Honeycomb)
"""

import os
import uuid
from contextlib import contextmanager
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


# ---------------------------------------------------------------------------
# Bootstrap — runs once at import time, before any request is handled
# ---------------------------------------------------------------------------

# PHOENIX_COLLECTOR_ENDPOINT defaults to local Phoenix dev server.
# Override in .env for staging/prod to point at a real OTLP collector.
_PHOENIX_ENDPOINT = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")

_provider = TracerProvider()

# BatchSpanProcessor buffers spans and sends them in batches asynchronously.
# This means tracing adds ~0ms to request latency — spans are flushed in background.
_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=_PHOENIX_ENDPOINT))
)

# Register as the global provider — otel_trace.get_tracer() calls anywhere in
# the codebase will use this provider without needing to pass it around.
otel_trace.set_tracer_provider(_provider)

# "adrag" is the instrumentation scope — appears as the service name in Phoenix UI
_tracer = otel_trace.get_tracer("adrag")


# ---------------------------------------------------------------------------
# Public API — same signatures as the old print() version, calling code unchanged
# ---------------------------------------------------------------------------

@contextmanager
def trace(query: str):
    """
    Root span for one full request. Wraps the entire pipeline in api/main.py.

    Yields a short trace_id string for log correlation.
    OTel context propagation handles parent-child nesting automatically —
    any span() opened inside this block becomes a child span.
    """
    trace_id = str(uuid.uuid4())[:8]
    with _tracer.start_as_current_span("request") as span:
        span.set_attribute("query", query)
        # Store trace_id as attribute so log lines can be correlated with traces
        span.set_attribute("trace_id", trace_id)
        yield trace_id


@contextmanager
def span(trace_id: str, step: str, metadata: dict = None):
    """
    Child span for one pipeline step (rewrite, hybrid_search, rerank, etc.).

    trace_id is accepted for backward compatibility but ignored internally —
    OTel reads the active context to determine the parent span automatically.

    metadata dict is stored as span attributes, queryable in Phoenix UI.
    Example: span(trace_id, "evaluate", {"quality": "GOOD"})
    """
    with _tracer.start_as_current_span(step) as s:
        span_data = {"step": step, "trace_id": trace_id, "metadata": metadata or {}}
        yield span_data
        # Set attributes after yield — caller may have added keys to span_data["metadata"]
        # during the with block. Setting here ensures all attributes are captured.
        for k, v in span_data["metadata"].items():
            s.set_attribute(k, str(v))