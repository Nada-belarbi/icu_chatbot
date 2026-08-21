# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# System packages needed by torch/transformers/chromadb wheels and lxml/python-docx parsing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY config/ ./config/
COPY src/ ./src/
COPY scripts/ ./scripts/

# Not copied: .env (secrets), data/, models/, logs/ — these are mounted as volumes
# (see docker-compose.yml) so the Chroma index, CamemBERT weights and conversation
# data survive container rebuilds and are never baked into the image.

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py"]
