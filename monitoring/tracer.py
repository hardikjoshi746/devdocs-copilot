import os
import time
import uuid
from contextlib import contextmanager

BACKEND = os.environ.get("TRACER_BACKEND", "phoenix")

@contextmanager
def trace(query: str):
    trace_id = str(uuid.uuid4())[:8]
    print(f"[trace:{trace_id}] START — {query[:60]}")
    yield trace_id
    print(f"[trace:{trace_id}] END")

@contextmanager
def span(trace_id: str, step: str, metadata: dict = None):
    start = time.perf_counter()
    span_data = {"step": step, "trace_id": trace_id, "metadata": metadata or {}}
    yield span_data
    span_data["latency_ms"] = round((time.perf_counter() - start) * 1000)
    if trace_id:
        print(f"[{trace_id}] {step} — {span_data['latency_ms']}ms")