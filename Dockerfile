# High-performance Spam/Scam API - Docker image with Ollama and Sarah model
FROM python:3.11-slim

# Prevent Python from writing pyc and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_HOST=0.0.0.0:11434

WORKDIR /app

# Install system deps (zstd required by Ollama install script)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (using official install script)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model (used by extract_intel for NER)
RUN pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Application (root-level layout)
COPY config.py ./
COPY api/ ./api/
COPY core/ ./core/
COPY extraction/ ./extraction/
COPY bots/ ./bots/
COPY training/ ./training/
COPY models/ ./models/
COPY Modelfile ./Modelfile
COPY setup_ollama_model.py ./setup_ollama_model.py

# Create startup script (Sarah setup non-fatal so API and ML models always start)
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "=== Starting Spam Detection API with Ollama ==="\n\
\n\
# Start Ollama in background\n\
echo "[1/3] Starting Ollama server..."\n\
ollama serve &\n\
OLLAMA_PID=$!\n\
\n\
# Setup sarah model in background so API can start immediately\n\
echo "[2/3] Setting up sarah model in background..."\n\
python3 /app/setup_ollama_model.py > /tmp/sarah_setup.log 2>&1 &\n\
\n\
# Start the API (ML models load here)\n\
echo "[3/3] Starting FastAPI server..."\n\
exec uvicorn api.main:app --host ${HOST} --port ${PORT}\n\
' > /app/start.sh && chmod +x /app/start.sh

# Cloud Run expects PORT env; default 8000 for local/VM
ENV PORT=8000 HOST=0.0.0.0
EXPOSE 8000 11434

# Use startup script
CMD ["/app/start.sh"]
