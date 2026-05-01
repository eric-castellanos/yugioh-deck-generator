# Overview

The system generates, evaluates, and explains Yu-Gi-Oh! decks from user constraints such as format, archetype, strategy, required cards, and deck size.

## Core Principle

- The generator builds legal decks deterministically.
- The LLM explains and enriches the result; it does **not** generate decks.

# High-Level Architecture

```text
User Request
   ↓
API Layer
   ↓
Constraint Parser
   ↓
Retrieval Layer (RAG)
   ↓
Candidate Card Pool
   ↓
Deck Generator
   ↓
Legality Validator
   ↓
Deck Evaluator
   ↓
LLM Explanation Layer
   ↓
Final Response
```

# Core Components

## API Layer

Handles incoming requests.

### Endpoints

- `POST /generate-deck`
- `POST /evaluate-deck`
- `POST /explain-deck`
- `GET /cards/search`
- `GET /cards/{id}`

### Tech

- FastAPI
- Pydantic

## Data Layer

Stores all structured data.

### Sources

- YGOPRODeck API
- Banlists
- Decklists (future)
- Simulation results (future)

### Tables

- `cards`
- `card_archetypes`
- `card_sets`
- `card_prices`
- `banlist_info`
- `decks`
- `deck_cards`
- `deck_scores`

### File Storage

```text
data/
  raw/
  processed/
  embeddings/
```

## Ingestion Layer

### Pipeline

`API -> Raw JSON -> Validation -> Normalization -> Storage`

### Key Principles

- Always store raw data.
- Version datasets.
- Normalize nested structures.
- Preserve card text exactly.

## Feature Engineering Layer

### Card Features

- type, attribute, race
- level / rank / link
- ATK / DEF
- archetype
- effect text

### Derived Features

- starter
- extender
- searcher
- interruption
- normal summon dependency

### Deck Features

- monster/spell/trap ratios
- starter count
- consistency metrics
- archetype density

## Retrieval Layer (RAG)

Used for context, not generation.

### Sources

- card text
- rulings
- archetype summaries
- decklists
- combo guides

### Tech Options

- Chroma
- FAISS
- pgvector

### Functions

- `search_cards()`
- `find_similar_cards()`
- `get_archetype_context()`
- `get_format_context()`

## Deck Generation Layer

### Inputs

- format
- archetype
- strategy
- required_cards
- deck_size

### Flow

```text
constraints
-> candidate pool
-> synergy scoring
-> deck construction
-> validation
-> output
```

### Approach (MVP)

- Rule-based
- Heuristic scoring
- Embedding similarity

### Future

- Genetic algorithms
- Reinforcement learning
- Simulation optimization

## Legality Layer

Ensures decks are valid.

### Checks

- deck size
- copy limits
- banlist restrictions
- format legality

This must always be deterministic; never rely on LLM output.

## Deck Evaluation Layer

Scores deck quality.

### Metrics

- consistency
- synergy
- power
- brick risk
- balance

### Example

```yaml
overall_score: 84
consistency: 78
synergy: 88
brick_risk: 22
```

## LLM Layer

Used only for explanation.

### Responsibilities

- explain strategy
- describe combos
- justify card choices
- summarize deck

### Inputs

- final decklist
- card metadata
- evaluation scores
- retrieved context

### Constraints

- cannot modify deck
- cannot override legality

## Simulation Layer (Optional)

### MVP

- draw 5 cards simulation
- starter probability
- brick probability

### Long-Term

- duel simulation (EDOPro / ocgcore)
- matchup modeling

# Repo Structure

```text
yugioh-ai-deck-generator/
  apps/
    api/
    web/

  src/
    data/
    features/
    retrieval/
    generation/
    validation/
    evaluation/
    llm/
    training/

  infra/
  data/
  notebooks/
  tests/
  docs/
```

# MVP Flow

1. User submits request.
2. Parse constraints.
3. Retrieve candidate cards.
4. Generate deck.
5. Validate legality.
6. Evaluate deck.
7. Generate explanation.
8. Return response.

# Design Principles

- Deterministic generation > LLM generation
- Structured data is the source of truth
- RAG provides context only
- Every deck should be reproducible
- Evaluation should be structured (JSON)
- Keep components modular

# Long-Term Evolution

```text
Deck data
-> Model training
-> Deck ranker
-> Optimization loop
-> Simulation feedback
```

## Future Features

- Meta prediction
- Side deck generator
- Budget decks
- Matchup-aware generation
- Full duel simulation
