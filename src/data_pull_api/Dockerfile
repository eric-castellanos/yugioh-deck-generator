# --- Stage 1: Build ---
FROM python:3.11 as build 

COPY sh/install_poetry.sh /tmp/
RUN bash /tmp/install_poetry.sh

# Copy pyproject.toml and poetry.lock from two levels up
COPY pyproject.toml poetry.lock ./

ENV PATH="/root/.local/bin:$PATH"

# Export dependencies from the specific group
RUN poetry export --only pull_api --without-hashes -f requirements.txt > requirements.txt

COPY src/data_pull_api/pull_data.py ./app/pull_data.py

# --- Stage 2: Run ---
FROM python:3.11-slim as run


# Install dependencies
COPY --from=build /app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script only
COPY --from=build /app/pull_data.py ./pull_data.py

# Ensure poetry-installed packages are in path
ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONPATH="/root/.local/lib/python3.11/site-packages"

# Run script
CMD ["python", "pull_data.py"]