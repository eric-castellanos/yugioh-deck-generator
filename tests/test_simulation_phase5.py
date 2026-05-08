from __future__ import annotations

import pandas as pd

from yugioh_deck_generator.simulation.aggregation import aggregate_deck_win_rates
from yugioh_deck_generator.simulation.request_builder import build_simulation_requests
from yugioh_deck_generator.simulation.schemas import DuelResult, SimulationRequest


def test_build_simulation_requests_generates_cartesian_pairs(tmp_path) -> None:
    opponent_dir = tmp_path / "opponents"
    opponent_dir.mkdir(parents=True, exist_ok=True)
    (opponent_dir / "a.ydk").write_text("#main\n1\n#extra\n!side\n", encoding="utf-8")
    (opponent_dir / "b.ydk").write_text("#main\n2\n#extra\n!side\n", encoding="utf-8")

    decks_df = pd.DataFrame(
        [
            {
                "deck_id": "d1",
                "format": "Edison",
                "main": [1, 2, 3],
                "extra": [4],
                "side": [],
            },
            {
                "deck_id": "d2",
                "format": "GOAT",
                "main": [5, 6, 7],
                "extra": [],
                "side": [],
            },
        ]
    )

    requests = build_simulation_requests(
        decks_df=decks_df,
        generation_run_id="r1",
        opponent_dir=opponent_dir,
        output_dir=tmp_path / "simout",
        scenario="phase5_round_robin_v1",
        base_seed=100,
    )

    assert len(requests) == 4
    by_deck = {}
    for req in requests:
        by_deck.setdefault(req.deck_id, set()).add(req.opponent_deck_id)
    assert by_deck["d1"] == {"a", "b"}
    assert by_deck["d2"] == {"a", "b"}


def test_aggregate_deck_win_rates() -> None:
    requests = [
        SimulationRequest(
            simulation_id="s1",
            generation_run_id="r1",
            deck_id="d1",
            format="Edison",
            deck_path="/tmp/d1.ydk",
            opponent_deck_id="o1",
            opponent_deck_path="/tmp/o1.ydk",
            scenario="phase5_round_robin_v1",
            seed=1,
        ),
        SimulationRequest(
            simulation_id="s2",
            generation_run_id="r1",
            deck_id="d1",
            format="Edison",
            deck_path="/tmp/d1.ydk",
            opponent_deck_id="o2",
            opponent_deck_path="/tmp/o2.ydk",
            scenario="phase5_round_robin_v1",
            seed=2,
        ),
    ]
    results = [
        DuelResult(
            simulation_id="s1",
            deck_id="d1",
            opponent_deck_id="o1",
            winner="self",
            status="completed",
            seed=1,
            engine_version="stub-v1",
            error="",
        ),
        DuelResult(
            simulation_id="s2",
            deck_id="d1",
            opponent_deck_id="o2",
            winner="opponent",
            status="completed",
            seed=2,
            engine_version="stub-v1",
            error="",
        ),
    ]

    agg = aggregate_deck_win_rates(
        requests=requests,
        results=results,
        scenario="phase5_round_robin_v1",
    )
    assert len(agg) == 1
    row = agg[0]
    assert row.deck_id == "d1"
    assert row.wins == 1
    assert row.losses == 1
    assert row.duel_count == 2
    assert row.win_rate == 0.5
