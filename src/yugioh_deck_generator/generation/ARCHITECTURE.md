# Generation Architecture

## Purpose
This module implements deterministic, format-aware Yu-Gi-Oh! deck generation with blueprint-first orchestration, constrained sampling, validation/repair, and `.ydk` export.

## High-Level Flow
1. CLI parses command + config paths.
2. Config/data loaders read format rules, profiles, staple pools, cards, and cluster assignments.
3. Generator orchestrates per-format deck creation.
4. Blueprint defines one deck spec (sizes, ratios, novelty/staple controls, cluster anchors).
5. Candidate pool filters legal sampling pools for main/extra deck candidate sets.
6. Sampler builds main/extra card ID lists using staples, novelty, ratio targets, copy limits.
7. Validator/constraints enforce legality and rule-based checks.
8. Repair attempts size corrections if validation fails.
9. Accepted decks are written as artifacts (`jsonl`, `parquet`, `.ydk`) and summarized.

## Module Responsibilities

### `schemas.py`
- Typed domain models.
- Pydantic config/spec models:
  - `RatioConfig`
  - `FormatConfig`
  - `DeckSpec`
- Runtime result dataclass:
  - `GeneratedDeck`

### `config_loader.py`
- Loads JSON/YAML config files.
- Parses/validates config structure for:
  - formats
  - staple pools
  - sampling profiles
- Produces typed `FormatConfig` + nested `RatioConfig`.

### `io.py`
- Reads tabular inputs (`parquet/csv/json/jsonl`).
- Reads banlist snapshots (`json/yaml`) into `forbidden/limited/semi_limited` structure.
- Writes output artifacts:
  - `deck_blueprints.jsonl`
  - `generated_decks.jsonl`
  - `rejection_log.jsonl`
  - `summary.json`
  - `.ydk` files (`#main`, `#extra`, `!side`)

### `blueprint.py`
- Creates one `DeckSpec` per deck attempt.
- Selects:
  - run/deck IDs
  - target sizes/ratios
  - staple + novelty controls
  - primary/secondary clusters (from cluster distribution)

### `candidate_pool.py`
- Splits card universe into:
  - `main_df` (non-token candidates)
  - `extra_df` (Fusion/Synchro/Xyz/Link/Pendulum types)
- Optional allowlist filtering.

### `staples.py`
- Bernoulli-style staple inclusion (`p_staple * weight` per staple entry).
- Returns staple candidate IDs for sampler injection.

### `novelty.py`
- Computes novelty candidate IDs from cluster metadata.
- Mixed novelty strategy:
  - near-cluster share (`near_share`)
  - far-cluster share (`far_share`)
- Supports exploratory picks beyond nearest clusters.

### `sampler.py`
- Core constrained constructor for one deck.
- Main deck stages:
  1. Staple injection
  2. Novelty injection
  3. Ratio-bucket fill (monster/spell/trap)
  4. Tail fill to exact size
- Prefers 2-3 copies per selected main card.
- Respects per-card copy limits (banlist-derived when provided).
- Samples extra deck candidates up to target size.

### `constraints.py`
- Hard validation checks:
  - main/extra size bounds
  - card existence
  - max-copy limits (global + banlist)
  - extra-deck type legality
  - ratio tolerance checks
- Summon-path static checks:
  - Fusion enabler presence
  - Synchro tuner/non-tuner support
  - Xyz same-level support
  - Link effect-monster support
  - Pendulum scale viability

### `validator.py`
- Thin validation adapter around `constraints.validate_deck`.
- Keeps generator call-sites clean.

### `repair.py`
- Lightweight size repair pass when initial validation fails:
  - fill/truncate main
  - fill/truncate extra
- Returns repaired IDs + repair count.

### `generator.py`
- End-to-end orchestration.
- Per format:
  - loads banlist and derives copy limits
  - loops deck attempts
  - blueprint -> sample -> validate -> optional repair -> re-validate
  - collects accepted/rejected results + diagnostics
- Writes run artifacts and `.ydk` files.

### `cli.py`
- Entry points:
  - `generate`
  - `validate`
- Wires external files/configs to pipeline functions.

## Data Contracts

### Primary Inputs
- Cards table (`id`, `name`, `type`, plus optional `level/scale/...`)
- Cluster assignments (`card_id`, `cluster_id`, `nearest_cluster_id`, optional confidence)
- Format config (`configs/generation/formats.json`)
- Sampling profile config (`configs/generation/sampling_profiles.json`)
- Staple pools (`configs/generation/staple_pools.json`)
- Optional banlist snapshots via `legality_source`

### Primary Outputs
- Per run directory: `data/generated/decks/<run_id>/`
  - `deck_blueprints.jsonl`
  - `generated_decks.jsonl`
  - `rejection_log.jsonl`
  - `generation_results.parquet` (if accepted decks exist)
  - `summary.json`
  - `<deck_id>.ydk` for each accepted deck

## Constraint Enforcement Points
- **During sampling** (`sampler.py`):
  - copy-limit-aware card insertion
  - target composition shaping
- **After sampling** (`constraints.py` via `validator.py`):
  - legality gates and summon-path checks
- **Fallback** (`repair.py`):
  - size-only corrections before final validation

## Determinism
- Randomness is driven by seeded `Random` instances.
- Reproducibility depends on:
  - stable inputs/configs
  - fixed seed
  - deterministic ordering of source artifacts

## Current Limitations
- Repair is size-focused, not full legality-aware replacement.
- Novelty uses cluster heuristics; no learned synergy model yet.
- Some staple pools are placeholders pending complete card-id mapping.
- Full test execution currently depends on environment alignment (Python/Pydantic versions).
