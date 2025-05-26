# --- Stage 1: Build ---
FROM python:3.11 as build

COPY sh/install_poetry.sh /tmp/
RUN bash /tmp/install_poetry.sh
ENV PATH="/root/.local/bin:$PATH"
    
WORKDIR /app
    
COPY pyproject.toml poetry.lock ./
RUN poetry export --only pull_api --without-hashes -f requirements.txt > requirements.txt
    
# ✅ Clean copy to /app, not /app/app
COPY src/data_pull_api/pull_data.py ./pull_data.py
    
# --- Stage 2: Run ---
FROM python:3.11-slim as run
    
WORKDIR /app
    
# Install dependencies
COPY --from=build /app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
    
# Copy the script
COPY --from=build /app/pull_data.py ./pull_data.py
    
CMD ["python", "pull_data.py"]