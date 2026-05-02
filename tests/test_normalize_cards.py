from __future__ import annotations

import json
from pathlib import Path

from yugioh_deck_generator.cleaning.normalize_cards import normalize_payload


def test_normalize_payload_shapes_and_counts() -> None:
    fixture_path = Path("tests/fixtures/sample_ygoprodeck_cards.json")
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    tables = normalize_payload(payload)

    assert set(tables.keys()) == {
        "cards",
        "card_images",
        "card_sets",
        "card_prices",
        "card_archetypes",
        "banlist_info",
    }

    assert len(tables["cards"]) == 2
    assert len(tables["card_images"]) == 1
    assert len(tables["card_sets"]) == 1
    assert len(tables["card_prices"]) == 1
    assert len(tables["card_archetypes"]) == 1
    assert len(tables["banlist_info"]) == 1

    first_card = tables["cards"][0]
    assert first_card["id"] == 123
    assert first_card["name"] == "Test Dragon"
    assert first_card["def"] == 2000
    assert "def_" not in first_card
    assert first_card["formats"] == "TCG|OCG|Master Duel"
    assert first_card["views"] == 1000
