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
make postgres-up
make load-cards
make append-cards TARGET_FILE=data/processed/cards.parquet
make phase1-pipeline
make postgres-down
```
