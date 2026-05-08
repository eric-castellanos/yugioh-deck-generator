from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from yugioh_deck_generator.generation.io import write_summary
from yugioh_deck_generator.simulation.schemas import DeckAggregate, DuelResult, SimulationRequest


def write_requests(path: str | Path, rows: list[SimulationRequest]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=True) + "\n")


def write_duel_results(path: str | Path, rows: list[DuelResult]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=True) + "\n")


def write_deck_aggregates(path: str | Path, rows: list[DeckAggregate]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=True) + "\n")


def write_duel_results_parquet(path: str | Path, rows: list[DuelResult]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row.model_dump(mode="json") for row in rows])
    df.to_parquet(p, index=False)


def build_summary(
    *,
    requests: list[SimulationRequest],
    results: list[DuelResult],
    aggregates: list[DeckAggregate],
    scenario: str,
) -> dict[str, Any]:
    completed = sum(1 for r in results if r.status == "completed")
    failed = sum(1 for r in results if r.status != "completed")
    avg_win_rate = (
        sum(float(row.win_rate) for row in aggregates) / float(len(aggregates))
        if aggregates
        else 0.0
    )
    return {
        "scenario": scenario,
        "simulation_request_count": int(len(requests)),
        "duel_result_count": int(len(results)),
        "completed_count": int(completed),
        "failed_count": int(failed),
        "deck_aggregate_count": int(len(aggregates)),
        "avg_deck_win_rate": float(avg_win_rate),
    }


def write_summary_json(path: str | Path, summary: dict[str, Any]) -> None:
    write_summary(path, summary)
