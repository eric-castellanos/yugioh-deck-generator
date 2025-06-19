# --- Stage 1: Build ---
FROM python:3.11-slim as build

# Install wget and clean up after
RUN apt-get update && apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/*

COPY sh/install_poetry.sh /tmp/
RUN bash /tmp/install_poetry.sh && \
    export PATH="/root/.local/bin:$PATH" && \
    poetry --version
ENV PATH="/root/.local/bin:$PATH"
    
WORKDIR /app
    
COPY pyproject.toml poetry.lock ./
RUN poetry export --only clustering --without-hashes -f requirements.txt > requirements.txt

# Copy all necessary files for clustering
COPY src/model/clustering/hdbscan-clustering.py ./hdbscan-clustering.py
COPY src/utils/ ./src/utils/
COPY src/__init__.py ./src/__init__.py
    
# --- Stage 2: Run ---
FROM python:3.11-slim as run
    
WORKDIR /app
    
# Install dependencies
COPY --from=build /app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
    
# Copy the clustering module and utilities
COPY --from=build /app/hdbscan-clustering.py ./hdbscan-clustering.py
COPY --from=build /app/src/ ./src/
    
# Set PYTHONPATH for module imports
ENV PYTHONPATH=/app

# MLflow configuration - connects to mlflow-local service by default
ENV MLFLOW_TRACKING_URI=http://mlflow-local:5000
ENV MLFLOW_EXPERIMENT_NAME=/yugioh_card_clustering

# Default command - can be overridden with different arguments
CMD ["python", "hdbscan-clustering.py", "--feature-type", "combined", "--experiment-config", "combined_grid_search"]
