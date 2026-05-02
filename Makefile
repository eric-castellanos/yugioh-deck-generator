.PHONY: install lint format test check fetch-cards normalize-cards

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

fetch-cards:
	poetry run python -m yugioh_deck_generator.data_collection.fetch_ygoprodeck_cards

normalize-cards:
	poetry run python -m yugioh_deck_generator.data.normalize_cards

check: lint test
