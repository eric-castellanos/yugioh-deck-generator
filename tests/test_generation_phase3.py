from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from yugioh_deck_generator.generation.cli import _load_allowed_ids_from_cdb
from yugioh_deck_generator.generation.constraints import validate_deck
from yugioh_deck_generator.generation.generator import run_generation
from yugioh_deck_generator.generation.io import write_ydk


def _cards_fixture() -> pd.DataFrame:
    rows = []
    cid = 1000

    def add(name: str, type_text: str, n: int) -> None:
        nonlocal cid
        for _ in range(n):
            rows.append({"id": cid, "name": name, "type": type_text})
            cid += 1

    add("Alpha Monster", "Effect Monster", 30)
    add("Beta Spell", "Spell Card", 18)
    add("Gamma Trap", "Trap Card", 18)
    add("Polymerization", "Spell Card", 1)
    add("Fusion Test", "Fusion Monster", 6)
    add("Synchro Test", "Synchro Monster", 6)
    return pd.DataFrame(rows).drop_duplicates(subset=["id"]).reset_index(drop=True)


def _cluster_fixture(cards_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "card_id": cards_df["id"],
            "cluster_id": [1 for _ in range(len(cards_df))],
            "cluster_probability": [1.0 for _ in range(len(cards_df))],
            "outlier_score": [0.0 for _ in range(len(cards_df))],
            "nearest_cluster_id": [2 for _ in range(len(cards_df))],
        }
    )


def test_ydk_writer_format(tmp_path: Path) -> None:
    out = tmp_path / "deck.ydk"
    write_ydk(out, [1, 1, 2], [3], [])
    text = out.read_text(encoding="utf-8")
    assert text.startswith("#main\n1\n1\n2\n#extra\n3\n!side\n")


def test_fusion_requires_enabler() -> None:
    cards = pd.DataFrame(
        [
            {"id": 1, "name": "Fusion Test", "type": "Fusion Monster"},
            {"id": 2, "name": "Normal Monster", "type": "Effect Monster"},
            {"id": 3, "name": "Polymerization", "type": "Spell Card"},
        ]
    )

    ok_without, _, _ = validate_deck(
        main_ids=[2],
        extra_ids=[1],
        side_ids=[],
        cards_df=cards,
        main_size=1,
        extra_min=1,
        extra_max=15,
        ratio_targets={"monster": 1.0, "spell": 0.0, "trap": 0.0},
    )
    assert ok_without is False

    ok_with, _, _ = validate_deck(
        main_ids=[2, 3],
        extra_ids=[1],
        side_ids=[],
        cards_df=cards,
        main_size=2,
        extra_min=1,
        extra_max=15,
        ratio_targets={"monster": 0.5, "spell": 0.5, "trap": 0.0},
    )
    assert ok_with is True


def test_extra_summon_paths_require_support() -> None:
    cards = pd.DataFrame(
        [
            {"id": 1, "name": "Synchro Boss", "type": "Synchro Monster", "level": 8, "scale": None},
            {"id": 2, "name": "Xyz Boss", "type": "Xyz Monster", "level": None, "scale": None},
            {"id": 3, "name": "Link Boss", "type": "Link Monster", "level": None, "scale": None},
            {
                "id": 4,
                "name": "Pendulum Boss",
                "type": "Pendulum Effect Monster",
                "level": 7,
                "scale": None,
            },
            {
                "id": 10,
                "name": "Tuner A",
                "type": "Tuner Effect Monster",
                "level": 3,
                "scale": None,
            },
            {"id": 11, "name": "NonTuner A", "type": "Effect Monster", "level": 5, "scale": None},
            {"id": 12, "name": "Level4 A", "type": "Effect Monster", "level": 4, "scale": None},
            {"id": 13, "name": "Level4 B", "type": "Effect Monster", "level": 4, "scale": None},
            {"id": 14, "name": "Pend A", "type": "Pendulum Effect Monster", "level": 4, "scale": 1},
            {"id": 15, "name": "Pend B", "type": "Pendulum Effect Monster", "level": 5, "scale": 8},
        ]
    )

    ok_fail, flags_fail, _ = validate_deck(
        main_ids=[12],
        extra_ids=[1, 2, 3, 4],
        side_ids=[],
        cards_df=cards,
        main_size=1,
        extra_min=1,
        extra_max=15,
        ratio_targets={"monster": 1.0, "spell": 0.0, "trap": 0.0},
    )
    assert ok_fail is False
    assert flags_fail["synchro_path_ok"] is False
    assert flags_fail["xyz_path_ok"] is False
    assert flags_fail["link_path_ok"] is False
    assert flags_fail["pendulum_path_ok"] is False

    ok_pass, flags_pass, _ = validate_deck(
        main_ids=[10, 11, 12, 13, 14, 15],
        extra_ids=[1, 2, 3, 4],
        side_ids=[],
        cards_df=cards,
        main_size=6,
        extra_min=1,
        extra_max=15,
        ratio_targets={"monster": 1.0, "spell": 0.0, "trap": 0.0},
        tolerance_count=2,
    )
    assert ok_pass is True
    assert flags_pass["synchro_path_ok"] is True
    assert flags_pass["xyz_path_ok"] is True
    assert flags_pass["link_path_ok"] is True
    assert flags_pass["pendulum_path_ok"] is True


def test_ratio_tolerance_applies_to_all_three_types() -> None:
    cards = pd.DataFrame(
        [
            {"id": 1, "name": "M1", "type": "Effect Monster"},
            {"id": 2, "name": "M2", "type": "Effect Monster"},
            {"id": 3, "name": "M3", "type": "Effect Monster"},
            {"id": 4, "name": "S1", "type": "Spell Card"},
            {"id": 5, "name": "S2", "type": "Spell Card"},
            {"id": 6, "name": "T1", "type": "Trap Card"},
        ]
    )

    # Deck = 3 monsters, 2 spells, 1 trap (size 6)
    # Target = 2/2/2 with tolerance 0 should fail on monster and trap.
    ok_strict, _, errors_strict = validate_deck(
        main_ids=[1, 2, 3, 4, 5, 6],
        extra_ids=[],
        side_ids=[],
        cards_df=cards,
        main_size=6,
        extra_min=0,
        extra_max=15,
        ratio_targets={"monster": 2 / 6, "spell": 2 / 6, "trap": 2 / 6},
        tolerance_count=0,
    )
    assert ok_strict is False
    assert any("monster ratio mismatch" in e for e in errors_strict)
    assert any("trap ratio mismatch" in e for e in errors_strict)

    # With tolerance 1, the same deck should pass (all three checks use the tolerance).
    ok_tolerant, _, _ = validate_deck(
        main_ids=[1, 2, 3, 4, 5, 6],
        extra_ids=[],
        side_ids=[],
        cards_df=cards,
        main_size=6,
        extra_min=0,
        extra_max=15,
        ratio_targets={"monster": 2 / 6, "spell": 2 / 6, "trap": 2 / 6},
        tolerance_count=1,
    )
    assert ok_tolerant is True


def test_material_requirements_archetype_tag_validation() -> None:
    cards = pd.DataFrame(
        [
            {
                "id": 1,
                "name": "X-Saber Boss",
                "type": "Synchro Monster",
                "desc": '1 Tuner + 1 or more non-Tuner "X-Saber" monsters',
            },
            {"id": 2, "name": "Generic Tuner", "type": "Tuner Effect Monster", "desc": ""},
            {"id": 3, "name": "Generic NonTuner", "type": "Effect Monster", "desc": ""},
            {"id": 4, "name": "X-Saber Airbellum", "type": "Effect Monster", "desc": ""},
        ]
    )

    ok_fail, flags_fail, errors_fail = validate_deck(
        main_ids=[2, 3],
        extra_ids=[1],
        side_ids=[],
        cards_df=cards,
        main_size=2,
        extra_min=1,
        extra_max=15,
        ratio_targets={"monster": 1.0, "spell": 0.0, "trap": 0.0},
    )
    assert ok_fail is False
    assert flags_fail["material_requirements_ok"] is False
    assert any("missing required material monster tag" in e for e in errors_fail)

    ok_pass, flags_pass, _ = validate_deck(
        main_ids=[2, 4],
        extra_ids=[1],
        side_ids=[],
        cards_df=cards,
        main_size=2,
        extra_min=1,
        extra_max=15,
        ratio_targets={"monster": 1.0, "spell": 0.0, "trap": 0.0},
    )
    assert ok_pass is True
    assert flags_pass["material_requirements_ok"] is True


def test_material_requirements_count_validation() -> None:
    cards = pd.DataFrame(
        [
            {
                "id": 1,
                "name": "Mokey Mokey King",
                "type": "Fusion Monster",
                "desc": '3 "Mokey Mokey"',
            },
            {"id": 2, "name": "Mokey Mokey", "type": "Normal Monster", "desc": ""},
            {"id": 3, "name": "Polymerization", "type": "Spell Card", "desc": ""},
        ]
    )

    ok_fail, flags_fail, errors_fail = validate_deck(
        main_ids=[2, 3],
        extra_ids=[1],
        side_ids=[],
        cards_df=cards,
        main_size=2,
        extra_min=1,
        extra_max=15,
        ratio_targets={"monster": 0.5, "spell": 0.5, "trap": 0.0},
    )
    assert ok_fail is False
    assert flags_fail["material_requirements_ok"] is False
    assert any("required=3 found=1" in e for e in errors_fail)

    ok_pass, flags_pass, _ = validate_deck(
        main_ids=[2, 2, 2, 3],
        extra_ids=[1],
        side_ids=[],
        cards_df=cards,
        main_size=4,
        extra_min=1,
        extra_max=15,
        ratio_targets={"monster": 0.75, "spell": 0.25, "trap": 0.0},
    )
    assert ok_pass is True
    assert flags_pass["material_requirements_ok"] is True


def test_material_requirements_typed_tuner_validation() -> None:
    cards = pd.DataFrame(
        [
            {
                "id": 1,
                "name": "Fish Synchro Boss",
                "type": "Synchro Monster",
                "desc": "1 Fish-Type Tuner + 1+ non-Tuner monsters",
            },
            {
                "id": 2,
                "name": "Generic Tuner",
                "type": "Tuner Effect Monster",
                "race": "Spellcaster",
            },
            {"id": 3, "name": "Fish Tuner", "type": "Tuner Effect Monster", "race": "Fish"},
            {"id": 4, "name": "NonTuner", "type": "Effect Monster", "race": "Warrior"},
        ]
    )

    ok_fail, flags_fail, errors_fail = validate_deck(
        main_ids=[2, 4],
        extra_ids=[1],
        side_ids=[],
        cards_df=cards,
        main_size=2,
        extra_min=1,
        extra_max=15,
        ratio_targets={"monster": 1.0, "spell": 0.0, "trap": 0.0},
    )
    assert ok_fail is False
    assert flags_fail["material_requirements_ok"] is False
    assert any("fish tuner" in e for e in errors_fail)

    ok_pass, flags_pass, _ = validate_deck(
        main_ids=[3, 4],
        extra_ids=[1],
        side_ids=[],
        cards_df=cards,
        main_size=2,
        extra_min=1,
        extra_max=15,
        ratio_targets={"monster": 1.0, "spell": 0.0, "trap": 0.0},
    )
    assert ok_pass is True
    assert flags_pass["material_requirements_ok"] is True


def test_material_requirements_attribute_monster_validation() -> None:
    cards = pd.DataFrame(
        [
            {
                "id": 1,
                "name": "Wind Synchro Boss",
                "type": "Synchro Monster",
                "desc": "1 WIND monster + 1+ non-Tuner monsters",
            },
            {"id": 2, "name": "Dark Monster", "type": "Effect Monster", "attribute": "DARK"},
            {"id": 3, "name": "Wind Monster", "type": "Effect Monster", "attribute": "WIND"},
            {"id": 4, "name": "Generic NonTuner", "type": "Effect Monster", "attribute": "EARTH"},
        ]
    )

    ok_fail, flags_fail, errors_fail = validate_deck(
        main_ids=[2, 4],
        extra_ids=[1],
        side_ids=[],
        cards_df=cards,
        main_size=2,
        extra_min=1,
        extra_max=15,
        ratio_targets={"monster": 1.0, "spell": 0.0, "trap": 0.0},
    )
    assert ok_fail is False
    assert flags_fail["material_requirements_ok"] is False
    assert any("wind monster" in e for e in errors_fail)

    ok_pass, flags_pass, _ = validate_deck(
        main_ids=[3, 4],
        extra_ids=[1],
        side_ids=[],
        cards_df=cards,
        main_size=2,
        extra_min=1,
        extra_max=15,
        ratio_targets={"monster": 1.0, "spell": 0.0, "trap": 0.0},
    )
    assert ok_pass is True
    assert flags_pass["material_requirements_ok"] is True


def test_material_requirements_attribute_tuner_validation() -> None:
    cards = pd.DataFrame(
        [
            {
                "id": 1,
                "name": "Wind Tuner Synchro Boss",
                "type": "Synchro Monster",
                "desc": "1 WIND Tuner + 1+ non-Tuner monsters",
            },
            {"id": 2, "name": "Dark Tuner", "type": "Tuner Effect Monster", "attribute": "DARK"},
            {"id": 3, "name": "Wind Tuner", "type": "Tuner Effect Monster", "attribute": "WIND"},
            {"id": 4, "name": "Generic NonTuner", "type": "Effect Monster", "attribute": "EARTH"},
        ]
    )

    ok_fail, flags_fail, errors_fail = validate_deck(
        main_ids=[2, 4],
        extra_ids=[1],
        side_ids=[],
        cards_df=cards,
        main_size=2,
        extra_min=1,
        extra_max=15,
        ratio_targets={"monster": 1.0, "spell": 0.0, "trap": 0.0},
    )
    assert ok_fail is False
    assert flags_fail["material_requirements_ok"] is False
    assert any("wind tuner" in e for e in errors_fail)

    ok_pass, flags_pass, _ = validate_deck(
        main_ids=[3, 4],
        extra_ids=[1],
        side_ids=[],
        cards_df=cards,
        main_size=2,
        extra_min=1,
        extra_max=15,
        ratio_targets={"monster": 1.0, "spell": 0.0, "trap": 0.0},
    )
    assert ok_pass is True
    assert flags_pass["material_requirements_ok"] is True


def test_run_generation_outputs_files(tmp_path: Path) -> None:
    cards = _cards_fixture()
    clusters = _cluster_fixture(cards)

    format_cfg = tmp_path / "formats.json"
    format_cfg.write_text(
        json.dumps(
            {
                "formats": {
                    "Edison": {
                        "main_deck_size": 40,
                        "extra_deck_size": 15,
                        "min_main": 40,
                        "max_main": 40,
                        "min_extra": 15,
                        "max_extra": 15,
                        "card_type_ratios": {
                            "monster": 0.5,
                            "spell": 0.25,
                            "trap": 0.25,
                            "tolerance_count": 4,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    summary = run_generation(
        cards_df=cards,
        clusters_df=clusters,
        formats={
            "Edison": type(
                "F",
                (),
                {
                    "name": "Edison",
                    "main_deck_size": 40,
                    "extra_deck_size": 15,
                    "min_extra": 15,
                    "max_extra": 15,
                    "legality_source": None,
                    "card_type_ratios": type(
                        "R", (), {"monster": 0.5, "spell": 0.25, "trap": 0.25, "tolerance_count": 4}
                    )(),
                },
            )()
        },
        selected_formats=["Edison"],
        decks_per_format=2,
        mode="archetype_reconstruction",
        staple_pools={"Edison": []},
        p_staple=0.0,
        novelty_ratio=0.0,
        output_dir=str(tmp_path / "generated"),
        seed=42,
    )

    assert summary["accepted_count"] >= 1
    run_dir = Path(tmp_path / "generated" / summary["run_id"])
    assert (run_dir / "deck_blueprints.jsonl").exists()
    assert (run_dir / "generated_decks.jsonl").exists()
    ydk_files = list(run_dir.glob("*.ydk"))
    assert len(ydk_files) >= 1


def test_load_allowed_ids_from_cdb_reads_datas_ids(tmp_path: Path) -> None:
    cdb_path = tmp_path / "cards.cdb"
    conn = sqlite3.connect(str(cdb_path))
    try:
        conn.execute("CREATE TABLE datas (id INTEGER)")
        conn.executemany("INSERT INTO datas (id) VALUES (?)", [(1001,), (1002,), (0,), (-1,)])
        conn.commit()
    finally:
        conn.close()

    allowed = _load_allowed_ids_from_cdb(cdb_path)
    assert allowed == {1001, 1002}


def test_run_generation_with_allowlist_filters_output_ids(tmp_path: Path) -> None:
    cards = _cards_fixture()
    clusters = _cluster_fixture(cards)

    allowed_main = (
        cards[~cards["type"].str.contains("Fusion|Synchro", case=False, regex=True)]["id"]
        .astype(int)
        .tolist()
    )
    allowed_extra = (
        cards[cards["type"].str.contains("Fusion|Synchro", case=False, regex=True)]["id"]
        .astype(int)
        .head(15)
        .tolist()
    )
    allow_card_ids = set(allowed_main + allowed_extra)

    summary = run_generation(
        cards_df=cards,
        clusters_df=clusters,
        formats={
            "Edison": type(
                "F",
                (),
                {
                    "name": "Edison",
                    "main_deck_size": 40,
                    "extra_deck_size": 15,
                    "min_extra": 15,
                    "max_extra": 15,
                    "legality_source": None,
                    "card_type_ratios": type(
                        "R", (), {"monster": 0.5, "spell": 0.25, "trap": 0.25, "tolerance_count": 4}
                    )(),
                },
            )()
        },
        selected_formats=["Edison"],
        decks_per_format=1,
        mode="archetype_reconstruction",
        staple_pools={"Edison": []},
        p_staple=0.0,
        novelty_ratio=0.0,
        output_dir=str(tmp_path / "generated_allowlist"),
        seed=42,
        allow_card_ids=allow_card_ids,
    )

    assert summary["accepted_count"] == 1
    decks_path = Path(
        tmp_path / "generated_allowlist" / summary["run_id"] / "generated_decks.jsonl"
    )
    rows = [
        json.loads(line)
        for line in decks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    generated_ids = set(rows[0]["main"] + rows[0]["extra"])
    assert generated_ids.issubset(allow_card_ids)


def test_run_generation_allowlist_without_main_candidates_raises(tmp_path: Path) -> None:
    cards = _cards_fixture()
    clusters = _cluster_fixture(cards)
    # Keep only extra-deck card IDs so main candidate pool becomes empty.
    allow_card_ids = set(
        cards[cards["type"].str.contains("Fusion|Synchro", case=False, regex=True)]["id"]
        .astype(int)
        .tolist()
    )

    with pytest.raises(RuntimeError, match="No main-deck candidates remain"):
        run_generation(
            cards_df=cards,
            clusters_df=clusters,
            formats={
                "Edison": type(
                    "F",
                    (),
                    {
                        "name": "Edison",
                        "main_deck_size": 40,
                        "extra_deck_size": 15,
                        "min_extra": 15,
                        "max_extra": 15,
                        "legality_source": None,
                        "card_type_ratios": type(
                            "R",
                            (),
                            {"monster": 0.5, "spell": 0.25, "trap": 0.25, "tolerance_count": 4},
                        )(),
                    },
                )()
            },
            selected_formats=["Edison"],
            decks_per_format=1,
            mode="archetype_reconstruction",
            staple_pools={"Edison": []},
            p_staple=0.0,
            novelty_ratio=0.0,
            output_dir=str(tmp_path / "generated_allowlist_fail"),
            seed=42,
            allow_card_ids=allow_card_ids,
        )
