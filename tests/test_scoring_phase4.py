from __future__ import annotations

import pandas as pd

from yugioh_deck_generator.generation.scoring import score_decks


def test_score_decks_emits_v2_metrics(tmp_path) -> None:
    decks_df = pd.DataFrame(
        [
            {
                "deck_id": "d1",
                "deck_spec_id": "s1",
                "run_id": "r1",
                "format": "Edison",
                "status": "accepted",
                "main": [1, 1, 2, 2, 3, 4, 5, 6, 7, 8],
                "extra": [],
                "diagnostics": {
                    "num_staples_injected": 2,
                    "num_novelty_cards": 1,
                    "num_repairs": 0,
                    "num_resamples": 0,
                },
                "constraint_flags": {"main_size_ok": True},
                "errors": [],
            }
        ]
    )

    cards_df = pd.DataFrame(
        [
            {
                "id": 1,
                "type": "Effect Monster",
                "desc": "Add 1 card from your Deck to your hand.",
                "level": 4,
            },
            {
                "id": 2,
                "type": "Spell Card",
                "desc": "Draw 2 cards.",
                "level": None,
            },
            {
                "id": 3,
                "type": "Trap Card",
                "desc": "Negate the activation.",
                "level": None,
            },
            {
                "id": 4,
                "type": "Effect Monster",
                "desc": "Special Summon this card.",
                "level": 4,
            },
            {
                "id": 5,
                "type": "Effect Monster",
                "desc": "Vanilla.",
                "level": 8,
            },
            {
                "id": 6,
                "type": "Effect Monster",
                "desc": "Vanilla.",
                "level": 8,
            },
            {
                "id": 7,
                "type": "Spell Card",
                "desc": "Search your deck.",
                "level": None,
            },
            {
                "id": 8,
                "type": "Trap Card",
                "desc": "Set 1 card.",
                "level": None,
            },
        ]
    )

    embeddings_df = pd.DataFrame(
        [
            {"card_id": 1, "embedding_vector": [1.0, 0.0]},
            {"card_id": 2, "embedding_vector": [0.9, 0.1]},
            {"card_id": 3, "embedding_vector": [0.7, 0.3]},
            {"card_id": 4, "embedding_vector": [0.95, 0.05]},
            {"card_id": 5, "embedding_vector": [0.2, 0.8]},
            {"card_id": 6, "embedding_vector": [0.1, 0.9]},
            {"card_id": 7, "embedding_vector": [0.8, 0.2]},
            {"card_id": 8, "embedding_vector": [0.6, 0.4]},
        ]
    )

    role_tags_df = pd.DataFrame(
        [
            {"card_id": 1, "role_tag": "starter", "confidence": 0.9},
            {"card_id": 5, "role_tag": "brick", "confidence": 0.8},
        ]
    )

    scored_df, summary, artifacts = score_decks(
        decks_df=decks_df,
        cards_df=cards_df,
        output_dir=tmp_path,
        embeddings_df=embeddings_df,
        role_tags_df=role_tags_df,
        sim_trials=200,
        sim_seed=42,
        score_version="both",
    )

    assert len(scored_df) == 1
    row = scored_df.iloc[0]
    assert "evaluation_v2" in row
    assert 0.0 <= float(row["heuristic_score_v2"]) <= 100.0
    assert 0.0 <= float(row["consistency_score"]) <= 100.0
    assert 0.0 <= float(row["synergy_score"]) <= 100.0
    assert 0.0 <= float(row["brick_risk_score"]) <= 100.0
    assert 0.0 <= float(row["balance_score"]) <= 100.0
    assert float(row["main_embedding_coverage"]) > 0.0
    assert float(row["starter_hit_rate"]) > 0.0

    assert summary["scored_deck_count"] == 1
    assert "heuristic_score_v2_mean" in summary
    assert len(artifacts) >= 3


def test_score_decks_v2_works_without_optional_inputs(tmp_path) -> None:
    decks_df = pd.DataFrame(
        [
            {
                "deck_id": "d2",
                "deck_spec_id": "s2",
                "run_id": "r2",
                "format": "GOAT",
                "status": "accepted",
                "main": [10, 11, 12, 13, 14],
                "extra": [],
                "diagnostics": {},
                "constraint_flags": {},
                "errors": [],
            }
        ]
    )
    cards_df = pd.DataFrame(
        [
            {"id": 10, "type": "Effect Monster", "desc": "", "level": 4},
            {"id": 11, "type": "Spell Card", "desc": "", "level": None},
            {"id": 12, "type": "Trap Card", "desc": "", "level": None},
            {"id": 13, "type": "Effect Monster", "desc": "", "level": 8},
            {"id": 14, "type": "Effect Monster", "desc": "", "level": 4},
        ]
    )

    scored_df, _, _ = score_decks(
        decks_df=decks_df,
        cards_df=cards_df,
        output_dir=tmp_path,
        sim_trials=50,
        sim_seed=7,
        score_version="v2",
    )

    row = scored_df.iloc[0]
    assert "evaluation_v2" in row
    assert 0.0 <= float(row["heuristic_score_v2"]) <= 100.0
