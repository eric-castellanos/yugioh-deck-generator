# Yu-Gi-Oh! Deck Generator

Deterministic deck generation, legality validation, evaluation, and LLM-based explanation.

## Environment Setup (pyenv + Poetry)

```bash
pyenv install 3.11.9 -s
pyenv local 3.11.9
poetry env use $(pyenv which python)
poetry install
```

## Common Commands

```bash
make lint
make format
make test
make fetch-cards
make normalize-cards
make services-up
make load-cards
make append-cards TARGET_FILE=data/processed/cards.parquet
make phase1-pipeline
make feature-pipeline
make build-features
make services-down
```

## ZenML Dashboard Login

- URL: `http://localhost:8080`
- Default username: `admin`
- Default password: `admin123`

You can override these via environment variables before starting services:

```bash
export ZENML_DEFAULT_USER_NAME=your_user
export ZENML_DEFAULT_USER_PASSWORD=your_password
make services-up
```

If ZenML was already initialized with an existing Docker volume, credential changes may not apply. In that case:

```bash
docker compose -f infra/docker/docker-compose.yml down -v
make services-up
```
