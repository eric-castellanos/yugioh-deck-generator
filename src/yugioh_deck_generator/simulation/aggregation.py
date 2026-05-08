from __future__ import annotations

from collections import defaultdict

from yugioh_deck_generator.simulation.schemas import DeckAggregate, DuelResult, SimulationRequest


def aggregate_deck_win_rates(
    *,
    requests: list[SimulationRequest],
    results: list[DuelResult],
    scenario: str,
) -> list[DeckAggregate]:
    format_by_deck: dict[str, str] = {}
    for req in requests:
        format_by_deck[req.deck_id] = req.format

    wins_losses: dict[str, dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "count": 0}
    )
    for row in results:
        acc = wins_losses[row.deck_id]
        acc["count"] += 1
        if row.winner == "self":
            acc["wins"] += 1
        else:
            acc["losses"] += 1

    out: list[DeckAggregate] = []
    for deck_id, acc in wins_losses.items():
        duel_count = int(acc["count"])
        wins = int(acc["wins"])
        losses = int(acc["losses"])
        win_rate = (wins / float(duel_count)) if duel_count else 0.0
        out.append(
            DeckAggregate(
                deck_id=deck_id,
                format=format_by_deck.get(deck_id, "unknown"),
                duel_count=duel_count,
                wins=wins,
                losses=losses,
                win_rate=win_rate,
                scenario=scenario,
            )
        )

    return sorted(out, key=lambda x: (x.format, x.deck_id))
