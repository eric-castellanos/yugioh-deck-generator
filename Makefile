.PHONY: install lint format test check

install:
	poetry install

lint:
	poetry run ruff check src tests
	poetry run black --check src tests

format:
	poetry run ruff check --fix src tests
	poetry run black src tests

test:
	poetry run pytest -q

check: lint test
