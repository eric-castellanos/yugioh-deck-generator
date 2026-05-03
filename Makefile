.PHONY: install lint format test typecheck check fetch-cards normalize-cards services-up services-down services-mlflow-up load-cards append-cards phase1-pipeline build-features feature-pipeline cluster-embeddings cluster-embeddings-sweep mlflow-init-experiments

ZENML_CONFIG_PATH := $(CURDIR)/.zenml
ZENML_ENV := ZENML_CONFIG_PATH=$(ZENML_CONFIG_PATH)

install:
	$(ZENML_ENV) poetry install

lint:
	$(ZENML_ENV) poetry run ruff check src tests
	$(ZENML_ENV) poetry run black --check src tests

format:
	$(ZENML_ENV) poetry run black src tests
	$(ZENML_ENV) poetry run ruff check --fix src tests

test:
	$(ZENML_ENV) poetry run pytest -q

typecheck:
	$(ZENML_ENV) poetry run mypy src

fetch-cards:
	$(ZENML_ENV) poetry run python -m yugioh_deck_generator.data_collection.fetch_ygoprodeck_cards

normalize-cards:
	$(ZENML_ENV) poetry run python -m yugioh_deck_generator.cleaning.normalize_cards

services-up:
	docker compose -f infra/docker/docker-compose.yml up -d

services-mlflow-up:
	docker compose -f infra/docker/docker-compose.yml up -d postgres minio minio-init mlflow

services-down:
	docker compose -f infra/docker/docker-compose.yml down

load-cards:
	$(ZENML_ENV) poetry run python -m yugioh_deck_generator.cleaning.load_to_postgres

append-cards:
	@if [ -z "$(TARGET_FILE)" ]; then echo "Usage: make append-cards TARGET_FILE=data/processed/cards.parquet"; exit 1; fi
	$(ZENML_ENV) poetry run python -m yugioh_deck_generator.cleaning.load_to_postgres --mode append --target-file "$(TARGET_FILE)"

phase1-pipeline: normalize-cards load-cards

build-features:
	$(ZENML_ENV) poetry run python -m yugioh_deck_generator.features.build_features --publish-postgres --publish-feature-store --publish-vector-store

feature-pipeline: build-features

cluster-embeddings:
	$(ZENML_ENV) poetry run python -m yugioh_deck_generator.clustering.cluster_embeddings --publish-postgres --mlflow-tracking-uri http://localhost:5000

cluster-embeddings-sweep:
	$(ZENML_ENV) poetry run python -m yugioh_deck_generator.clustering.sweep_hdbscan --mlflow-tracking-uri http://localhost:5000

mlflow-init-experiments:
	$(ZENML_ENV) poetry run python -m yugioh_deck_generator.training.setup_mlflow_experiments --tracking-uri http://localhost:5000

check: lint typecheck test
