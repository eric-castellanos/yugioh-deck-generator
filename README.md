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
```
