# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies in a virtualenv or specific path
COPY server/requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Stage 2: Final
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Copy installed python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only necessary server and client files
# Tránh copy thư mục flutter_nongsan, .git, .venv
COPY server/ ./server/
COPY client/ ./client/
COPY README.md .

# Remove unnecessary files to shrink image
RUN find . -type d -name "__pycache__" -exec rm -rf {} + \
    && find . -type f -name "*.pyc" -delete

WORKDIR /app/server

# Setup HuggingFace Cache and Ingest Knowledge Base
ENV HF_HOME=/app/server/ml/hf_cache
# Create data directory and set permissions
RUN mkdir -p /app/server/data && chmod 777 /app/server/data

RUN python ingest_kb.py

# Expose port
EXPOSE 7860

# Metadata
LABEL maintainer="Agricultural AI Team"
LABEL version="2.0"

# Command to run
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
