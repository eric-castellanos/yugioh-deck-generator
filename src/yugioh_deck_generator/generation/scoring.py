from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from yugioh_deck_generator.generation.io import read_table, write_jsonl, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score generated decks with a simple heuristic.")
    parser.add_argument("--decks-file", required=True)
    parser.add_argument("--cards-file", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument(
        "--mlflow-experiment",
        default="yugioh-deck-generator/generation-heuristic-scoring",
    )
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def _is_monster(card_type: str) -> bool:
    return "monster" in card_type.lower()


def _is_spell(card_type: str) -> bool:
    return "spell" in card_type.lower()


def _is_trap(card_type: str) -> bool:
    return "trap" in card_type.lower()


def _main_type_counts(main_ids: list[int], card_type_map: dict[int, str]) -> tuple[int, int, int]:
    monster = 0
    spell = 0
    trap = 0
    for card_id in main_ids:
        card_type = card_type_map.get(int(card_id), "")
        if _is_monster(card_type):
            monster += 1
        elif _is_spell(card_type):
            spell += 1
        elif _is_trap(card_type):
            trap += 1
    return monster, spell, trap


def _extract_int_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [int(x) for x in parsed]
        except json.JSONDecodeError:
            return []
    return []


def _score_one_deck(deck: dict[str, Any], card_type_map: dict[int, str]) -> dict[str, Any]:
    main_ids = _extract_int_list(deck.get("main"))
    extra_ids = _extract_int_list(deck.get("extra"))
    diagnostics = deck.get("diagnostics") or {}
    constraint_flags = deck.get("constraint_flags") or {}
    errors = deck.get("errors") or []

    main_size = len(main_ids)
    extra_size = len(extra_ids)
    unique_main = len(set(main_ids))
    unique_ratio = (unique_main / float(main_size)) if main_size else 0.0

    monster_count, spell_count, trap_count = _main_type_counts(main_ids, card_type_map)
    known_type_count = monster_count + spell_count + trap_count
    monster_ratio = (monster_count / float(known_type_count)) if known_type_count else 0.0
    spell_ratio = (spell_count / float(known_type_count)) if known_type_count else 0.0
    trap_ratio = (trap_count / float(known_type_count)) if known_type_count else 0.0

    # Very light structural priors for a first-pass ranking.
    target_monster = 0.50
    target_spell = 0.30
    target_trap = 0.20
    ratio_penalty = (
        abs(monster_ratio - target_monster)
        + abs(spell_ratio - target_spell)
        + abs(trap_ratio - target_trap)
    )

    staples = int(diagnostics.get("num_staples_injected", 0) or 0)
    novelty = int(diagnostics.get("num_novelty_cards", 0) or 0)
    repairs = int(diagnostics.get("num_repairs", 0) or 0)
    resamples = int(diagnostics.get("num_resamples", 0) or 0)
    accepted_flags = sum(1 for _, passed in dict(constraint_flags).items() if bool(passed))
    total_flags = len(dict(constraint_flags))

    heuristic_score = (
        100.0 + (15.0 * novelty) + (1.5 * staples) + (12.0 * unique_ratio) - (30.0 * ratio_penalty)
    )

    return {
        "deck_id": deck.get("deck_id"),
        "deck_spec_id": deck.get("deck_spec_id"),
        "run_id": deck.get("run_id"),
        "format": deck.get("format"),
        "status": deck.get("status"),
        "heuristic_score_v1": float(heuristic_score),
        "main_size": main_size,
        "extra_size": extra_size,
        "unique_main_count": unique_main,
        "unique_main_ratio": float(unique_ratio),
        "monster_ratio": float(monster_ratio),
        "spell_ratio": float(spell_ratio),
        "trap_ratio": float(trap_ratio),
        "num_staples_injected": staples,
        "num_novelty_cards": novelty,
        "num_repairs": repairs,
        "num_resamples": resamples,
        "constraint_flags_passed": int(accepted_flags),
        "constraint_flags_total": int(total_flags),
        "error_count": int(len(errors) if isinstance(errors, list) else 0),
    }


def score_decks(
    *,
    decks_df: pd.DataFrame,
    cards_df: pd.DataFrame,
    output_dir: str | Path,
    source_summary_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    card_type_map = {
        int(row["id"]): str(row.get("type", ""))
        for _, row in cards_df[["id", "type"]].drop_duplicates(subset=["id"]).iterrows()
    }

    deck_records = decks_df.to_dict(orient="records")
    scored_rows = [_score_one_deck(row, card_type_map) for row in deck_records]
    scored_df = pd.DataFrame(scored_rows)

    scored_jsonl = out_root / "deck_heuristic_scores.jsonl"
    scored_parquet = out_root / "deck_heuristic_scores.parquet"
    summary_json = out_root / "scoring_summary.json"
    per_format_json = out_root / "deck_scores_by_format.json"

    write_jsonl(scored_jsonl, scored_rows)
    scored_df.to_parquet(scored_parquet, index=False)

    top_row = (
        scored_df.sort_values("heuristic_score_v1", ascending=False).iloc[0].to_dict()
        if not scored_df.empty
        else {}
    )
    summary = {
        "scored_deck_count": int(len(scored_df)),
        "heuristic_score_mean": (
            float(scored_df["heuristic_score_v1"].mean()) if len(scored_df) else 0.0
        ),
        "heuristic_score_std": (
            float(scored_df["heuristic_score_v1"].std(ddof=0)) if len(scored_df) else 0.0
        ),
        "heuristic_score_min": (
            float(scored_df["heuristic_score_v1"].min()) if len(scored_df) else 0.0
        ),
        "heuristic_score_max": (
            float(scored_df["heuristic_score_v1"].max()) if len(scored_df) else 0.0
        ),
        "top_deck": top_row,
    }
    write_summary(summary_json, summary)

    by_format: dict[str, Any] = {}
    if len(scored_df):
        grouped = scored_df.groupby("format", dropna=False)["heuristic_score_v1"]
        for fmt, series in grouped:
            by_format[str(fmt)] = {
                "count": int(series.count()),
                "mean": float(series.mean()),
                "min": float(series.min()),
                "max": float(series.max()),
            }
    write_summary(per_format_json, by_format)

    artifacts = [scored_jsonl, scored_parquet, summary_json, per_format_json]
    if source_summary_path and source_summary_path.exists():
        artifacts.append(source_summary_path)
    return scored_df, summary, artifacts


def log_scoring_to_mlflow(
    *,
    tracking_uri: str | None,
    experiment: str,
    run_name: str,
    run_tags: dict[str, str],
    summary_metrics: dict[str, Any],
    artifacts: list[Path],
    scored_df: pd.DataFrame,
) -> None:
    if not tracking_uri:
        return

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.set_tag("scoring_method", "heuristic_score_v1")
        for key, value in run_tags.items():
            mlflow.set_tag(key, value)
        mlflow.log_metrics(
            {
                "scored_deck_count": float(summary_metrics["scored_deck_count"]),
                "heuristic_score_mean": float(summary_metrics["heuristic_score_mean"]),
                "heuristic_score_std": float(summary_metrics["heuristic_score_std"]),
                "heuristic_score_min": float(summary_metrics["heuristic_score_min"]),
                "heuristic_score_max": float(summary_metrics["heuristic_score_max"]),
            }
        )
        # Batch per-deck metrics to avoid one network request per metric call.
        from mlflow.entities import Metric
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        run_id = active_run.info.run_id
        metric_specs: list[tuple[str, str]] = [
            ("heuristic_score_v1", "deck_heuristic_score_v1"),
            ("unique_main_ratio", "deck_unique_main_ratio"),
            ("monster_ratio", "deck_monster_ratio"),
            ("spell_ratio", "deck_spell_ratio"),
            ("trap_ratio", "deck_trap_ratio"),
            ("num_staples_injected", "deck_num_staples_injected"),
            ("num_novelty_cards", "deck_num_novelty_cards"),
            ("num_repairs", "deck_num_repairs"),
            ("num_resamples", "deck_num_resamples"),
            ("constraint_flags_passed", "deck_constraint_flags_passed"),
            ("constraint_flags_total", "deck_constraint_flags_total"),
            ("error_count", "deck_error_count"),
        ]

        batch_metrics: list[Metric] = []
        # Keep chunk comfortably below typical backend payload limits.
        chunk_size = 1000
        for step, row in enumerate(scored_df.to_dict(orient="records")):
            ts_ms = int(time.time() * 1000)
            for row_key, metric_name in metric_specs:
                batch_metrics.append(
                    Metric(key=metric_name, value=float(row[row_key]), step=step, timestamp=ts_ms)
                )
            if len(batch_metrics) >= chunk_size:
                client.log_batch(run_id=run_id, metrics=batch_metrics)
                batch_metrics = []
        if batch_metrics:
            client.log_batch(run_id=run_id, metrics=batch_metrics)

        # Also log a deck-level table artifact for full per-deck details.
        mlflow.log_table(scored_df, artifact_file="per_deck_scoring_table.json")
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))


def run(args: argparse.Namespace) -> int:
    decks_df = read_table(args.decks_file)
    cards_df = read_table(args.cards_file)

    decks_path = Path(args.decks_file)
    source_summary_path = decks_path.parent / "summary.json"
    out_root = Path(args.output_dir) if args.output_dir else decks_path.parent / "scoring"

    scored_df, summary, artifacts = score_decks(
        decks_df=decks_df,
        cards_df=cards_df,
        output_dir=out_root,
        source_summary_path=source_summary_path,
    )

    run_name = args.mlflow_run_name or f"deck-scoring-{args.run_id or 'adhoc'}"
    tags = {
        "source_decks_file": str(decks_path),
        "source_cards_file": str(Path(args.cards_file)),
        "scoring_output_dir": str(out_root),
    }
    if args.run_id:
        tags["generation_run_id"] = args.run_id
    if not scored_df.empty and "run_id" in scored_df.columns:
        first_run_id = (
            str(scored_df["run_id"].dropna().iloc[0]) if scored_df["run_id"].notna().any() else ""
        )
        if first_run_id:
            tags.setdefault("generation_run_id", first_run_id)
    log_scoring_to_mlflow(
        tracking_uri=args.mlflow_tracking_uri,
        experiment=args.mlflow_experiment,
        run_name=run_name,
        run_tags=tags,
        summary_metrics=summary,
        artifacts=artifacts,
        scored_df=scored_df,
    )
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
