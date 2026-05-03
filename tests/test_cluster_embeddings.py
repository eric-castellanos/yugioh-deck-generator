from __future__ import annotations

import numpy as np
import pandas as pd

from yugioh_deck_generator.clustering.cluster_embeddings import (
    _heuristic_role_from_text,
    build_cluster_metadata,
)


def test_heuristic_role_mapping() -> None:
    assert _heuristic_role_from_text("add from deck to hand")[0] == "searcher"
    assert _heuristic_role_from_text("draw 2 cards")[0] == "draw"
    assert _heuristic_role_from_text("destroy target card")[0] == "removal"
    assert _heuristic_role_from_text("negate this effect")[0] == "negation"
    assert _heuristic_role_from_text("send card to GY")[0] == "graveyard_setup"
    assert _heuristic_role_from_text("cannot activate cards")[0] == "floodgate"


def test_build_cluster_metadata_includes_required_fields() -> None:
    cards_df = pd.DataFrame(
        [
            {
                "id": 1,
                "name": "Card A",
                "type": "Effect Monster",
                "archetype": "Test",
                "desc": "add from deck then draw",
            },
            {
                "id": 2,
                "name": "Card B",
                "type": "Spell Card",
                "archetype": "Test",
                "desc": "destroy target card",
            },
        ]
    )
    cluster_df = pd.DataFrame(
        [
            {
                "card_id": 1,
                "cluster_id": 0,
                "cluster_probability": 0.9,
                "outlier_score": 0.1,
                "embedding_model": "m",
                "embedding_version": "v1",
                "cluster_version": "c1",
                "generated_at": "2026-01-01T00:00:00Z",
            },
            {
                "card_id": 2,
                "cluster_id": 0,
                "cluster_probability": 0.8,
                "outlier_score": 0.2,
                "embedding_model": "m",
                "embedding_version": "v1",
                "cluster_version": "c1",
                "generated_at": "2026-01-01T00:00:00Z",
            },
        ]
    )
    vectors = np.asarray([[0.1, 0.2], [0.2, 0.3]], dtype=float)
    metadata = build_cluster_metadata(
        cards_df=cards_df,
        cluster_df=cluster_df,
        vectors=vectors,
        cluster_version="c1",
        generated_at="2026-01-01T00:00:00Z",
        use_llm_labeling=False,
        ollama_url="http://localhost:11434",
        ollama_model="mistral",
    )

    assert not metadata.empty
    row = metadata.iloc[0]
    assert row["functional_role"] in {
        "starter",
        "extender",
        "searcher",
        "draw",
        "removal",
        "negation",
        "board_breaker",
        "floodgate",
        "graveyard_setup",
        "graveyard_hate",
        "recursion",
        "protection",
        "burn",
        "battle_trick",
        "archetype_core",
        "misc",
    }
    assert row["review_status"] == "unreviewed"
    assert isinstance(row["metadata"], dict)
    assert "top_terms" in row["metadata"]
