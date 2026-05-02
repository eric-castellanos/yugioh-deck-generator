.PHONY: install lint format test typecheck check fetch-cards normalize-cards postgres-up postgres-down load-cards append-cards phase1-pipeline

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

typecheck:
	poetry run mypy src

fetch-cards:
	poetry run python -m yugioh_deck_generator.data_collection.fetch_ygoprodeck_cards

normalize-cards:
	poetry run python -m yugioh_deck_generator.cleaning.normalize_cards

postgres-up:
	docker compose -f infra/docker/docker-compose.yml up -d

postgres-down:
	docker compose -f infra/docker/docker-compose.yml down

load-cards:
	poetry run python -m yugioh_deck_generator.cleaning.load_to_postgres

append-cards:
	@if [ -z "$(TARGET_FILE)" ]; then echo "Usage: make append-cards TARGET_FILE=data/processed/cards.parquet"; exit 1; fi
	poetry run python -m yugioh_deck_generator.cleaning.load_to_postgres --mode append --target-file "$(TARGET_FILE)"

phase1-pipeline: normalize-cards load-cards

check: lint typecheck test
