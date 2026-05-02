from __future__ import annotations

import pandas as pd

from yugioh_deck_generator.features.build_features import (
    EMBEDDING_DIM,
    build_card_embeddings,
    build_card_features,
    build_card_role_tags,
)


def test_build_features_outputs() -> None:
    cards_df = pd.DataFrame(
        [
            {
                "id": 1,
                "name": "Test Starter",
                "type": "Effect Monster",
                "race": "Warrior",
                "attribute": "LIGHT",
                "desc": "Add 1 card from your Deck to your hand.",
                "archetype": "Test",
                "atk": 500,
                "def": 500,
                "level": 4,
                "scale": None,
                "linkval": None,
            },
            {
                "id": 2,
                "name": "Test Spell",
                "type": "Spell Card",
                "race": "Normal",
                "attribute": None,
                "desc": "Destroy 1 card on the field.",
                "archetype": None,
                "atk": None,
                "def": None,
                "level": None,
                "scale": None,
                "linkval": None,
            },
        ]
    )

    features = build_card_features(
        cards_df, feature_version="v1", generated_at="2026-01-01T00:00:00Z"
    )
    embeddings = build_card_embeddings(
        cards_df, embedding_version="v1", generated_at="2026-01-01T00:00:00Z"
    )
    role_tags = build_card_role_tags(
        cards_df, role_version="v1", generated_at="2026-01-01T00:00:00Z"
    )

    assert len(features) == 2
    assert len(embeddings) == 2
    assert embeddings.iloc[0]["embedding_dim"] == EMBEDDING_DIM
    assert len(embeddings.iloc[0]["embedding_vector"]) == EMBEDDING_DIM

    first = features[features["card_id"] == 1].iloc[0]
    assert bool(first["is_effect_monster"]) is True
    assert bool(first["is_spell"]) is False

    second = features[features["card_id"] == 2].iloc[0]
    assert bool(second["is_spell"]) is True

    assert not role_tags.empty
    assert set(role_tags["role_tag"].tolist()).intersection({"starter", "searcher", "interruption"})
