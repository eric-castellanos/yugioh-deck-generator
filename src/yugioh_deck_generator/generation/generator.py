from __future__ import annotations

import json
import logging
import multiprocessing as mp
import re
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty
from random import Random
from typing import Any
from uuid import uuid4

import pandas as pd
from tqdm import tqdm

from yugioh_deck_generator.generation.blueprint import build_blueprint
from yugioh_deck_generator.generation.candidate_pool import build_candidate_pool
from yugioh_deck_generator.generation.io import (
    append_jsonl,
    read_banlist,
    write_jsonl,
    write_summary,
    write_ydk,
)
from yugioh_deck_generator.generation.repair import repair_deck
from yugioh_deck_generator.generation.sampler import prepare_sampling_inputs, sample_deck
from yugioh_deck_generator.generation.schemas import GeneratedDeck
from yugioh_deck_generator.generation.validator import validate_generated_deck

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)
ENABLE_PROGRESS_TRACKING = True


def _format_slug(name: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name or "").strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "format"


def _apply_tcg_release_cutoff(
    cards_df: pd.DataFrame, cutoff_date: str | None
) -> tuple[pd.DataFrame, int]:
    if not cutoff_date:
        return cards_df, 0
    if "tcg_date" not in cards_df.columns:
        logger.warning(
            "tcg_release_cutoff=%s configured but cards_df has no tcg_date column",
            cutoff_date,
        )
        return cards_df.iloc[0:0].copy(), len(cards_df)
    parsed_dates = pd.to_datetime(cards_df["tcg_date"], errors="coerce")
    cutoff_ts = pd.to_datetime(cutoff_date, errors="coerce")
    if pd.isna(cutoff_ts):
        logger.warning("Invalid tcg_release_cutoff=%s; skipping cutoff filter", cutoff_date)
        return cards_df, 0
    mask = parsed_dates.notna() & (parsed_dates <= cutoff_ts)
    filtered = cards_df[mask].copy()
    dropped = int((~mask).sum())
    return filtered, dropped


def _copy_limits_from_banlist(banlist: dict[str, list[int]] | None) -> dict[int, int]:
    limits: dict[int, int] = {}
    if not banlist:
        return limits
    for cid in banlist.get("forbidden", []):
        limits[int(cid)] = 0
    for cid in banlist.get("limited", []):
        limits[int(cid)] = 1
    for cid in banlist.get("semi_limited", []):
        limits[int(cid)] = 2
    return limits


def generate_for_format(
    *,
    cards_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    format_name: str,
    format_cfg: Any,
    decks_per_format: int,
    mode: str,
    seed: int,
    staple_pool: list[dict[str, Any]],
    p_staple: float,
    novelty_ratio: float,
    progress_blueprints_path: str | None = None,
    progress_accepted_path: str | None = None,
    progress_rejected_path: str | None = None,
    output_root: str | None = None,
    progress_queue: Any | None = None,
    allow_card_ids: set[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    logger.info(
        "Generating for format=%s decks_per_format=%d mode=%s seed=%d",
        format_name,
        decks_per_format,
        mode,
        seed,
    )
    if decks_per_format <= 0:
        return [], [], []

    banlist = read_banlist(format_cfg.legality_source)
    card_copy_limits = _copy_limits_from_banlist(banlist)
    generation_cards_df = cards_df
    cutoff_date = getattr(format_cfg, "tcg_release_cutoff", None)
    generation_cards_df, dropped_by_cutoff = _apply_tcg_release_cutoff(
        generation_cards_df, cutoff_date
    )
    if cutoff_date:
        logger.info(
            "Format=%s applied tcg_release_cutoff=%s kept=%d dropped=%d",
            format_name,
            cutoff_date,
            len(generation_cards_df),
            dropped_by_cutoff,
        )
    if allow_card_ids is not None:
        generation_cards_df = cards_df[cards_df["id"].astype(int).isin(allow_card_ids)].copy()
        generation_cards_df, dropped_by_cutoff_after_allow = _apply_tcg_release_cutoff(
            generation_cards_df, cutoff_date
        )
        logger.info(
            (
                "Format=%s restricting generation cards via allowlist: "
                "original=%d filtered=%d dropped_by_cutoff=%d"
            ),
            format_name,
            len(cards_df),
            len(generation_cards_df),
            dropped_by_cutoff_after_allow,
        )
    main_pool, extra_pool = build_candidate_pool(generation_cards_df, allow_card_ids=allow_card_ids)
    if main_pool.empty:
        raise RuntimeError(
            f"No main-deck candidates remain for format={format_name} after cards.cdb filtering"
        )
    logger.info(
        "Format=%s pools ready main_candidates=%d extra_candidates=%d copy_limits=%d",
        format_name,
        len(main_pool),
        len(extra_pool),
        len(card_copy_limits),
    )

    merged = generation_cards_df.merge(clusters_df, how="left", left_on="id", right_on="card_id")
    sampling_inputs = prepare_sampling_inputs(main_df=main_pool, extra_df=extra_pool)
    rng = Random(seed)
    generation_cfg = getattr(format_cfg, "generation", {}) or {}
    if not isinstance(generation_cfg, dict):
        generation_cfg = {}
    max_resample_attempts = int(generation_cfg.get("max_resample_attempts", 50))
    card_lookup = generation_cards_df.set_index("id", drop=False)
    known_ids = set(int(x) for x in card_lookup.index.tolist())
    total_attempts_cap = max(max_resample_attempts * decks_per_format * 5, decks_per_format)

    blueprints: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    idx = 0
    total_attempts = 0
    sample_time_s = 0.0
    validate_time_s = 0.0
    repair_time_s = 0.0
    sample_calls = 0
    validate_calls = 0
    repair_calls = 0
    failed_flag_counts: Counter[str] = Counter()
    failed_combo_counts: Counter[str] = Counter()
    while len(accepted) < decks_per_format and total_attempts < total_attempts_cap:
        spec = build_blueprint(
            fmt=format_cfg,
            strategy_type=mode,
            seed=seed + idx,
            p_staple=p_staple,
            novelty_ratio=novelty_ratio,
            clusters_df=clusters_df,
        )
        spec_row = spec.model_dump()
        blueprints.append(spec_row)
        if progress_blueprints_path:
            append_jsonl(Path(progress_blueprints_path), [spec_row])

        ok = False
        flags: dict[str, bool] = {}
        errors: list[str] = []
        main_ids: list[int] = []
        extra_ids: list[int] = []
        diagnostics: dict[str, Any] = {}
        repairs_total = 0
        resamples = 0
        attempts_for_slot = 0

        for _attempt in range(max_resample_attempts):
            attempts_for_slot += 1
            total_attempts += 1
            t0 = time.perf_counter()
            main_ids, extra_ids, diagnostics = sample_deck(
                rng=rng,
                main_df=main_pool,
                extra_df=extra_pool,
                joined_df=merged,
                main_size=spec.main_deck_size,
                extra_size=spec.extra_deck_size,
                ratios={
                    "monster": spec.card_type_ratios.monster,
                    "spell": spec.card_type_ratios.spell,
                    "trap": spec.card_type_ratios.trap,
                },
                staple_pool=staple_pool,
                p_staple=spec.p_staple,
                novelty_ratio=spec.novelty_ratio,
                allowed_novelty_clusters=set(spec.secondary_clusters),
                card_copy_limits=card_copy_limits,
                sampling_inputs=sampling_inputs,
            )
            sample_time_s += time.perf_counter() - t0
            sample_calls += 1

            t1 = time.perf_counter()
            ok, flags, errors = validate_generated_deck(
                cards_df=generation_cards_df,
                main_ids=main_ids,
                extra_ids=extra_ids,
                side_ids=[],
                main_size=format_cfg.main_deck_size,
                extra_min=format_cfg.min_extra,
                extra_max=format_cfg.max_extra,
                banlist=banlist,
                ratios={
                    "monster": spec.card_type_ratios.monster,
                    "spell": spec.card_type_ratios.spell,
                    "trap": spec.card_type_ratios.trap,
                },
                tolerance_count=spec.card_type_ratios.tolerance_count,
                card_lookup=card_lookup,
                known_ids=known_ids,
                tcg_release_cutoff=cutoff_date,
            )
            validate_time_s += time.perf_counter() - t1
            validate_calls += 1
            if not ok:
                failed_flags = sorted([k for k, passed in flags.items() if not passed])
                failed_flag_counts.update(failed_flags)
                failed_combo_counts["|".join(failed_flags) if failed_flags else "unknown"] += 1

            repairs = 0
            if not ok:
                failed_set = {k for k, passed in flags.items() if not passed}
                skip_repair_for_flags = {"type_ratio_ok", "material_requirements_ok"}
                should_skip_repair = bool(failed_set) and failed_set.issubset(skip_repair_for_flags)
                if should_skip_repair:
                    repairs_total += 0
                    resamples += 1
                    continue
                t2 = time.perf_counter()
                main_ids, extra_ids, repairs = repair_deck(
                    rng=rng,
                    main_ids=main_ids,
                    extra_ids=extra_ids,
                    main_size=format_cfg.main_deck_size,
                    extra_size=min(format_cfg.extra_deck_size, format_cfg.max_extra),
                    main_pool=main_pool,
                    extra_pool=extra_pool,
                )
                repair_time_s += time.perf_counter() - t2
                repair_calls += 1
                t3 = time.perf_counter()
                ok, flags, errors = validate_generated_deck(
                    cards_df=generation_cards_df,
                    main_ids=main_ids,
                    extra_ids=extra_ids,
                    side_ids=[],
                    main_size=format_cfg.main_deck_size,
                    extra_min=format_cfg.min_extra,
                    extra_max=format_cfg.max_extra,
                    banlist=banlist,
                    ratios={
                        "monster": spec.card_type_ratios.monster,
                        "spell": spec.card_type_ratios.spell,
                        "trap": spec.card_type_ratios.trap,
                    },
                    tolerance_count=spec.card_type_ratios.tolerance_count,
                    card_lookup=card_lookup,
                    known_ids=known_ids,
                    tcg_release_cutoff=cutoff_date,
                )
                validate_time_s += time.perf_counter() - t3
                validate_calls += 1
                if not ok:
                    failed_flags = sorted([k for k, passed in flags.items() if not passed])
                    failed_flag_counts.update(failed_flags)
                    failed_combo_counts["|".join(failed_flags) if failed_flags else "unknown"] += 1
            repairs_total += repairs

            if ok:
                break
            resamples += 1

        diagnostics["num_repairs"] = repairs_total
        diagnostics["num_resamples"] = resamples

        deck = GeneratedDeck(
            deck_id=str(uuid4()),
            deck_spec_id=spec.deck_spec_id,
            run_id=spec.run_id,
            format=format_name,
            main=main_ids,
            extra=extra_ids,
            side=[],
            diagnostics=diagnostics,
            status="accepted" if ok else "rejected",
        )
        row = asdict(deck)
        row["constraint_flags"] = flags
        row["errors"] = errors

        if ok:
            accepted.append(row)
            if progress_accepted_path:
                append_jsonl(Path(progress_accepted_path), [row])
            if output_root:
                deck_num = len(accepted)
                file_name = f"deck_{_format_slug(format_name)}_{deck_num}.ydk"
                write_ydk(
                    Path(output_root) / file_name,
                    row["main"],
                    row["extra"],
                    row["side"],
                )
            if progress_queue is not None:
                progress_queue.put(1)
        else:
            rejected.append(row)
            if progress_rejected_path:
                append_jsonl(Path(progress_rejected_path), [row])
        idx += 1

    logger.info(
        "Format generation complete format=%s accepted=%d/%d rejected=%d total_attempts=%d cap=%d",
        format_name,
        len(accepted),
        decks_per_format,
        len(rejected),
        total_attempts,
        total_attempts_cap,
    )
    total_timed_s = sample_time_s + validate_time_s + repair_time_s
    logger.info(
        "Format timing format=%s sample_s=%.3f validate_s=%.3f repair_s=%.3f total_timed_s=%.3f "
        "sample_calls=%d validate_calls=%d repair_calls=%d "
        "avg_sample_ms=%.2f avg_validate_ms=%.2f avg_repair_ms=%.2f "
        "failed_flags=%s failed_combos=%s",
        format_name,
        sample_time_s,
        validate_time_s,
        repair_time_s,
        total_timed_s,
        sample_calls,
        validate_calls,
        repair_calls,
        (sample_time_s / sample_calls * 1000.0) if sample_calls else 0.0,
        (validate_time_s / validate_calls * 1000.0) if validate_calls else 0.0,
        (repair_time_s / repair_calls * 1000.0) if repair_calls else 0.0,
        dict(failed_flag_counts),
        dict(failed_combo_counts),
    )
    if len(accepted) < decks_per_format:
        raise RuntimeError(
            f"Unable to generate requested decks for format={format_name}: "
            f"accepted={len(accepted)} requested={decks_per_format} after {total_attempts} attempts"
        )
    return blueprints, accepted, rejected


def _drain_progress_queue(progress_queue: Any, pbar: tqdm) -> int:
    advanced = 0
    while True:
        try:
            delta = int(progress_queue.get_nowait())
        except Empty:
            break
        advanced += delta
    if advanced > 0:
        pbar.update(advanced)
    return advanced


def _generate_for_format_task(
    task: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return generate_for_format(**task)


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    write_summary(path, payload)


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def run_generation(
    *,
    cards_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    formats: dict[str, Any],
    selected_formats: list[str],
    decks_per_format: int,
    mode: str,
    staple_pools: dict[str, list[dict[str, Any]]],
    p_staple: float,
    novelty_ratio: float,
    output_dir: str,
    resume_run_id: str | None = None,
    seed: int,
    parallel_formats: int | None = None,
    allow_card_ids: set[int] | None = None,
) -> dict[str, Any]:
    logger.info(
        "Run generation start formats=%s decks_per_format=%d mode=%s seed=%d output_dir=%s",
        selected_formats,
        decks_per_format,
        mode,
        seed,
        output_dir,
    )
    if allow_card_ids is not None:
        logger.info("cards.cdb ID filtering enabled allowlist_size=%d", len(allow_card_ids))
    run_id = resume_run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(output_dir) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("Run output directory: %s", out_root)

    blueprints_path = out_root / "deck_blueprints.jsonl"
    accepted_path = out_root / "generated_decks.jsonl"
    rejected_path = out_root / "rejection_log.jsonl"
    checkpoint_path = out_root / "checkpoint.json"
    progress_dir = out_root / "_progress"
    progress_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = _load_checkpoint(checkpoint_path)
    completed_formats = set(checkpoint.get("completed_formats", []))
    if resume_run_id:
        logger.info("Resuming run_id=%s completed_formats=%s", run_id, sorted(completed_formats))

    all_blueprints: list[dict[str, Any]] = []
    all_accepted: list[dict[str, Any]] = []
    all_rejected: list[dict[str, Any]] = []

    progress_meta: dict[str, dict[str, Path]] = {}
    tasks: list[dict[str, Any]] = []
    for fmt_name in selected_formats:
        if fmt_name not in formats:
            raise ValueError(f"Unknown format `{fmt_name}`")
        fmt_blueprints_path = progress_dir / f"{fmt_name}.deck_blueprints.jsonl"
        fmt_accepted_path = progress_dir / f"{fmt_name}.generated_decks.jsonl"
        fmt_rejected_path = progress_dir / f"{fmt_name}.rejection_log.jsonl"
        progress_meta[fmt_name] = {
            "blueprints": fmt_blueprints_path,
            "accepted": fmt_accepted_path,
            "rejected": fmt_rejected_path,
        }
        fmt_accepted_count = _count_jsonl_rows(fmt_accepted_path)
        if fmt_accepted_count >= decks_per_format:
            completed_formats.add(fmt_name)
        if fmt_name in completed_formats:
            logger.info("Skipping already completed format=%s", fmt_name)
            continue
        remaining = decks_per_format - fmt_accepted_count
        if remaining <= 0:
            completed_formats.add(fmt_name)
            continue
        tasks.append(
            {
                "cards_df": cards_df,
                "clusters_df": clusters_df,
                "format_name": fmt_name,
                "format_cfg": formats[fmt_name],
                "decks_per_format": remaining,
                "mode": mode,
                "seed": seed,
                "staple_pool": staple_pools.get(fmt_name, []),
                "p_staple": p_staple,
                "novelty_ratio": novelty_ratio,
                "progress_blueprints_path": str(fmt_blueprints_path),
                "progress_accepted_path": str(fmt_accepted_path),
                "progress_rejected_path": str(fmt_rejected_path),
                "output_root": str(out_root),
                "allow_card_ids": allow_card_ids,
            }
        )

    accepted_existing = sum(_count_jsonl_rows(m["accepted"]) for m in progress_meta.values())
    rejected_existing = sum(_count_jsonl_rows(m["rejected"]) for m in progress_meta.values())
    total_requested = decks_per_format * len(selected_formats)
    auto_parallel_formats = max(1, mp.cpu_count() - 1)
    requested_parallel_formats = (
        max(1, int(parallel_formats)) if parallel_formats is not None else auto_parallel_formats
    )
    logger.info(
        "Parallel formats resolved requested=%s auto_default=%d effective=%d",
        parallel_formats,
        auto_parallel_formats,
        requested_parallel_formats,
    )
    if ENABLE_PROGRESS_TRACKING:
        pbar = tqdm(total=total_requested, desc="Generating decks", unit="deck", dynamic_ncols=True)
        if accepted_existing:
            pbar.update(accepted_existing)
            pbar.set_postfix(formats_done=len(completed_formats), rejected=rejected_existing)
    else:
        pbar = None

    try:
        if requested_parallel_formats > 1 and len(tasks) > 1:
            workers = min(requested_parallel_formats, len(tasks))
            logger.info("Running format generation in parallel workers=%d", workers)
            ctx = mp.get_context("fork")
            progress_queue = None
            if pbar is not None:
                with ctx.Manager() as manager:
                    progress_queue = manager.Queue()
                    tasks_with_progress = [
                        dict(task, progress_queue=progress_queue) for task in tasks
                    ]
                    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                        future_to_format = {
                            ex.submit(_generate_for_format_task, task): task["format_name"]
                            for task in tasks_with_progress
                        }
                        pending = set(future_to_format)
                        while pending:
                            done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                            _drain_progress_queue(progress_queue, pbar)
                            pbar.set_postfix(
                                formats_done=len(completed_formats),
                                rejected=rejected_existing + len(all_rejected),
                            )
                            for future in done:
                                fmt_name = future_to_format[future]
                                bp, ok, bad = future.result()
                                all_blueprints.extend(bp)
                                all_accepted.extend(ok)
                                all_rejected.extend(bad)
                                completed_formats.add(fmt_name)
                                checkpoint = {
                                    "run_id": run_id,
                                    "requested_formats": selected_formats,
                                    "requested_decks_per_format": decks_per_format,
                                    "mode": mode,
                                    "seed": seed,
                                    "completed_formats": sorted(completed_formats),
                                }
                                _write_checkpoint(checkpoint_path, checkpoint)
                                logger.info(
                                    "Accumulated results after format=%s accepted=%d rejected=%d",
                                    fmt_name,
                                    accepted_existing + len(all_accepted),
                                    rejected_existing + len(all_rejected),
                                )
                        _drain_progress_queue(progress_queue, pbar)
            else:
                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                    future_to_format = {
                        ex.submit(_generate_for_format_task, task): task["format_name"]
                        for task in tasks
                    }
                    pending = set(future_to_format)
                    while pending:
                        done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                        for future in done:
                            fmt_name = future_to_format[future]
                            bp, ok, bad = future.result()
                            all_blueprints.extend(bp)
                            all_accepted.extend(ok)
                            all_rejected.extend(bad)
                            completed_formats.add(fmt_name)
                            checkpoint = {
                                "run_id": run_id,
                                "requested_formats": selected_formats,
                                "requested_decks_per_format": decks_per_format,
                                "mode": mode,
                                "seed": seed,
                                "completed_formats": sorted(completed_formats),
                            }
                            _write_checkpoint(checkpoint_path, checkpoint)
                            logger.info(
                                "Accumulated results after format=%s accepted=%d rejected=%d",
                                fmt_name,
                                accepted_existing + len(all_accepted),
                                rejected_existing + len(all_rejected),
                            )
        else:
            for task in tasks:
                fmt_name = task["format_name"]
                bp, ok, bad = _generate_for_format_task(task)
                all_blueprints.extend(bp)
                all_accepted.extend(ok)
                all_rejected.extend(bad)
                completed_formats.add(fmt_name)
                checkpoint = {
                    "run_id": run_id,
                    "requested_formats": selected_formats,
                    "requested_decks_per_format": decks_per_format,
                    "mode": mode,
                    "seed": seed,
                    "completed_formats": sorted(completed_formats),
                }
                _write_checkpoint(checkpoint_path, checkpoint)
                if pbar is not None:
                    pbar.update(len(ok))
                    pbar.set_postfix(
                        formats_done=len(completed_formats),
                        rejected=rejected_existing + len(all_rejected),
                    )
                logger.info(
                    "Accumulated results after format=%s accepted=%d rejected=%d",
                    fmt_name,
                    accepted_existing + len(all_accepted),
                    rejected_existing + len(all_rejected),
                )
    finally:
        if pbar is not None:
            pbar.close()

    final_blueprints: list[dict[str, Any]] = []
    final_accepted: list[dict[str, Any]] = []
    final_rejected: list[dict[str, Any]] = []
    for fmt_name in selected_formats:
        meta = progress_meta.get(fmt_name)
        if not meta:
            continue
        final_blueprints.extend(_read_jsonl_rows(meta["blueprints"]))
        final_accepted.extend(_read_jsonl_rows(meta["accepted"]))
        final_rejected.extend(_read_jsonl_rows(meta["rejected"]))

    write_jsonl(blueprints_path, final_blueprints)
    write_jsonl(accepted_path, final_accepted)
    write_jsonl(rejected_path, final_rejected)

    summary = {
        "run_id": run_id,
        "requested_formats": selected_formats,
        "requested_decks_per_format": decks_per_format,
        "accepted_count": len(final_accepted),
        "rejected_count": len(final_rejected),
        "mode": mode,
        "seed": seed,
        "completed_formats": sorted(completed_formats),
    }
    write_summary(out_root / "summary.json", summary)

    if summary["accepted_count"]:
        accepted_df = pd.read_json(accepted_path, orient="records", lines=True)
        accepted_df.to_parquet(out_root / "generation_results.parquet", index=False)
        logger.info("Wrote parquet results for %d accepted decks", len(accepted_df))
    else:
        logger.info("No accepted decks to write parquet results")

    logger.info(
        "Run generation complete run_id=%s accepted=%d rejected=%d",
        run_id,
        summary["accepted_count"],
        summary["rejected_count"],
    )
    return summary
