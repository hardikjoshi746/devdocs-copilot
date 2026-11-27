FROM python:3.12-slim

WORKDIR /app

# Install uv — faster than pip for installing dependencies
RUN pip install uv

# Copy dependency spec first — Docker caches this layer separately.
# If only source code changes, this layer (and the install below) are reused.
COPY pyproject.toml .

# Install all dependencies into the system Python (no venv needed inside a container —
# the container itself is the isolation boundary)
RUN uv pip install --system -e .

# Copy source code — this layer changes most often, so it goes last
COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]