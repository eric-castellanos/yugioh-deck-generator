# --- Stage 1: Build ---
FROM python:3.11-slim as build

COPY sh/install_poetry.sh /tmp/
RUN bash /tmp/install_poetry.sh
ENV PATH="/root/.local/bin:$PATH"
    
WORKDIR /app
    
COPY pyproject.toml poetry.lock ./
RUN poetry export --only pull_api --without-hashes -f requirements.txt > requirements.txt