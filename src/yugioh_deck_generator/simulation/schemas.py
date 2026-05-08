from __future__ import annotations

from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    simulation_id: str
    generation_run_id: str
    deck_id: str
    format: str
    deck_path: str
    opponent_deck_id: str
    opponent_deck_path: str
    scenario: str = "phase5_round_robin_v1"
    seed: int = Field(default=0)


class DuelResult(BaseModel):
    simulation_id: str
    deck_id: str
    opponent_deck_id: str
    winner: str = Field(pattern=r"^(self|opponent|timeout|invalid)$")
    status: str = Field(pattern=r"^(completed|failed)$")
    seed: int
    engine_version: str
    error: str = ""


class DeckAggregate(BaseModel):
    deck_id: str
    format: str
    duel_count: int
    wins: int
    losses: int
    win_rate: float
    scenario: str = "phase5_round_robin_v1"
