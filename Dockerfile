FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY encoding/__init__.py ./encoding/__init__.py
COPY encoding/config.py ./encoding/config.py
COPY frontend/ ./frontend/
# COPY data/embeddings/ ./data/embeddings/

ENV EMBEDDINGS_DIR=/app/data/embeddings/minilm_1500chars
ENV DEFAULT_TOP_K=5

EXPOSE 7860

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
