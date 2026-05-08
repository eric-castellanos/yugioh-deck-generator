from __future__ import annotations

import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any

from yugioh_deck_generator.simulation.engine_adapter import EngineAdapter
from yugioh_deck_generator.simulation.schemas import DuelResult, SimulationRequest

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

_WORKER_STATE: dict[str, Any] = {}


def _build_adapter(
    engine_mode: str,
    engine_version: str,
    engine_options: dict[str, Any] | None = None,
) -> EngineAdapter:
    options = dict(engine_options or {})
    return EngineAdapter(engine_mode=engine_mode, engine_version=engine_version, **options)


def _init_worker(engine_mode: str, engine_version: str, engine_options: dict[str, Any]) -> None:
    adapter = _build_adapter(
        engine_mode=engine_mode,
        engine_version=engine_version,
        engine_options=engine_options,
    )
    adapter.validate_environment()
    _WORKER_STATE["adapter"] = adapter


def _run_one_worker(payload: dict[str, Any]) -> dict[str, Any]:
    adapter: EngineAdapter = _WORKER_STATE["adapter"]
    req = SimulationRequest.model_validate(payload)
    result = adapter.run_duel(req)
    return result.model_dump(mode="json")


def run_simulations(
    *,
    requests: list[SimulationRequest],
    output_dir: str | Path,
    engine_mode: str,
    engine_version: str,
    engine_options: dict[str, Any] | None,
    num_workers: int,
    checkpoint_every: int,
) -> list[DuelResult]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "duel_results.checkpoint.jsonl"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    logger.info(
        "Running simulations total=%d workers=%d engine=%s",
        len(requests),
        int(num_workers),
        engine_mode,
    )

    rows: list[dict[str, Any]] = []
    append_since_flush = 0
    log_every = max(500, int(checkpoint_every))
    processed_since_log = 0
    with checkpoint_path.open("a", encoding="utf-8") as ckpt:
        if int(num_workers) <= 1:
            adapter = _build_adapter(
                engine_mode=engine_mode,
                engine_version=engine_version,
                engine_options=engine_options,
            )
            adapter.validate_environment()
            for req in requests:
                res = adapter.run_duel(req).model_dump(mode="json")
                rows.append(res)
                ckpt.write(json.dumps(res, ensure_ascii=True) + "\n")
                append_since_flush += 1
                processed_since_log += 1
                if append_since_flush >= max(1, int(checkpoint_every)):
                    ckpt.flush()
                    append_since_flush = 0
                if processed_since_log >= log_every:
                    logger.info("Simulation progress processed=%d/%d", len(rows), len(requests))
                    processed_since_log = 0
        else:
            worker_count = max(1, int(num_workers))
            payloads = [req.model_dump(mode="json") for req in requests]
            with mp.Pool(
                processes=worker_count,
                initializer=_init_worker,
                initargs=(engine_mode, engine_version, dict(engine_options or {})),
            ) as pool:
                chunk_size = max(1, min(200, len(payloads) // (worker_count * 4) or 1))
                for res in pool.imap_unordered(_run_one_worker, payloads, chunksize=chunk_size):
                    rows.append(res)
                    ckpt.write(json.dumps(res, ensure_ascii=True) + "\n")
                    append_since_flush += 1
                    processed_since_log += 1
                    if append_since_flush >= max(1, int(checkpoint_every)):
                        ckpt.flush()
                        append_since_flush = 0
                    if processed_since_log >= log_every:
                        logger.info(
                            "Simulation progress processed=%d/%d",
                            len(rows),
                            len(requests),
                        )
                        processed_since_log = 0
        ckpt.flush()

    results = [DuelResult.model_validate(row) for row in rows]
    logger.info("Simulation complete results=%d", len(results))
    return results
