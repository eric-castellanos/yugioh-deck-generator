from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, model_validator


class RatioConfig(BaseModel):
    """Monster/Spell/Trap composition targets used during constrained main-deck sampling."""

    monster: float = Field(...)
    spell: float = Field(...)
    trap: float = Field(...)
    tolerance_count: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def _validate_ratios(self) -> RatioConfig:
        for label, value in (
            ("monster", self.monster),
            ("spell", self.spell),
            ("trap", self.trap),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} ratio must be between 0 and 1")
        if abs((self.monster + self.spell + self.trap) - 1.0) > 1e-9:
            raise ValueError("monster + spell + trap must equal 1.0")
        return self


class FormatConfig(BaseModel):
    """Per-format legality and size/range constraints loaded from config files."""

    name: str
    main_deck_size: int = Field(default=40, ge=1)
    extra_deck_size: int = Field(default=15, ge=0)
    min_main: int = Field(default=40, ge=1)
    max_main: int = Field(default=40, ge=1)
    min_extra: int = Field(default=0, ge=0)
    max_extra: int = Field(default=15, ge=0)
    legality_source: str | None = None
    tcg_release_cutoff: str | None = None
    card_type_ratios: RatioConfig = Field(
        default_factory=lambda: RatioConfig(monster=0.5, spell=0.3, trap=0.2, tolerance_count=2)
    )
    generation: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_main_extra_bounds(self) -> FormatConfig:
        if self.min_main > self.max_main:
            raise ValueError("min_main must be <= max_main")
        if self.min_extra > self.max_extra:
            raise ValueError("min_extra must be <= max_extra")
        if not self.min_main <= self.main_deck_size <= self.max_main:
            raise ValueError("main_deck_size must be between min_main and max_main")
        if not self.min_extra <= self.extra_deck_size <= self.max_extra:
            raise ValueError("extra_deck_size must be between min_extra and max_extra")
        return self


class DeckSpec(BaseModel):
    """Resolved per-deck blueprint consumed by the deterministic sampler/validator flow."""

    deck_spec_id: str
    run_id: str
    format: str
    strategy_type: str
    seed: int
    main_deck_size: int = Field(..., ge=1)
    extra_deck_size: int = Field(..., ge=0)
    p_staple: float = Field(...)
    novelty_ratio: float = Field(...)
    card_type_ratios: RatioConfig
    primary_cluster: int | None = None
    secondary_clusters: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_probabilities(self) -> DeckSpec:
        if not 0.0 <= self.p_staple <= 1.0:
            raise ValueError("p_staple must be between 0 and 1")
        if not 0.0 <= self.novelty_ratio <= 1.0:
            raise ValueError("novelty_ratio must be between 0 and 1")
        return self


@dataclass
class GeneratedDeck:
    """Lightweight runtime result container for generated deck outputs and diagnostics."""

    deck_id: str
    deck_spec_id: str
    run_id: str
    format: str
    main: list[int]
    extra: list[int]
    side: list[int]
    diagnostics: dict[str, Any]
    status: str
