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
RUN poetry export --only preprocessing --without-hashes -f requirements.txt > requirements.txt

# Clean copy to /app
COPY src/preprocessing/EDA/eda.py ./eda.py
    
# --- Stage 2: Run ---
FROM python:3.11-slim as run
    
WORKDIR /app
    
# Install dependencies
COPY --from=build /app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
    
# Copy the script
COPY --from=build /app/eda.py ./eda.py
    
CMD ["python", "eda.py"]