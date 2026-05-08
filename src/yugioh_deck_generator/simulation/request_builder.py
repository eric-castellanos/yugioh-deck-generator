from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from yugioh_deck_generator.generation.io import write_ydk
from yugioh_deck_generator.simulation.schemas import SimulationRequest


def _extract_int_list(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
    return []


def _load_opponent_decks(opponent_dir: str | Path) -> list[tuple[str, Path]]:
    root = Path(opponent_dir)
    files = sorted(root.glob("*.ydk"))
    return [(f.stem, f) for f in files]


def build_simulation_requests(
    *,
    decks_df: pd.DataFrame,
    generation_run_id: str,
    opponent_dir: str | Path,
    output_dir: str | Path,
    scenario: str,
    base_seed: int,
) -> list[SimulationRequest]:
    out = Path(output_dir)
    generated_ydk_dir = out / "generated_ydk"
    generated_ydk_dir.mkdir(parents=True, exist_ok=True)

    opponents = _load_opponent_decks(opponent_dir)
    if not opponents:
        raise ValueError(f"No opponent .ydk files found in {opponent_dir}")

    requests: list[SimulationRequest] = []
    deck_rows = decks_df.to_dict(orient="records")
    for row_idx, row in enumerate(deck_rows):
        deck_id = str(row.get("deck_id") or uuid4())
        fmt = str(row.get("format") or "unknown")
        main = _extract_int_list(row.get("main"))
        extra = _extract_int_list(row.get("extra"))
        side = _extract_int_list(row.get("side"))

        generated_path = generated_ydk_dir / f"{deck_id}.ydk"
        write_ydk(generated_path, main=main, extra=extra, side=side)

        for opp_idx, (opp_id, opp_path) in enumerate(opponents):
            seed = int(base_seed) + (row_idx * len(opponents)) + opp_idx
            requests.append(
                SimulationRequest(
                    simulation_id=str(uuid4()),
                    generation_run_id=generation_run_id,
                    deck_id=deck_id,
                    format=fmt,
                    deck_path=str(generated_path),
                    opponent_deck_id=opp_id,
                    opponent_deck_path=str(opp_path),
                    scenario=scenario,
                    seed=seed,
                )
            )

    return requests
