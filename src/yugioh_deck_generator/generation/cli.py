from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

from yugioh_deck_generator.generation.config_loader import (
    load_formats,
    load_sampling_profiles,
    load_staple_pools,
)
from yugioh_deck_generator.generation.generator import run_generation
from yugioh_deck_generator.generation.io import read_table
from yugioh_deck_generator.generation.scoring import run as run_scoring
from yugioh_deck_generator.generation.validator import validate_generated_deck

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 deck generation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--cluster-assignments", required=True)
    gen.add_argument("--cards-file", required=True)
    gen.add_argument("--formats", nargs="+", required=True)
    gen.add_argument("--decks-per-format", type=int, required=True)
    gen.add_argument(
        "--mode", choices=["archetype_reconstruction", "archetype_plus_novelty"], required=True
    )
    gen.add_argument("--config", required=True)
    gen.add_argument("--format-config", required=True)
    gen.add_argument("--staples-config", required=True)
    gen.add_argument("--output-dir", default="data/generated/decks")
    gen.add_argument("--resume-run-id", default=None)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--parallel-formats", type=int, default=None)
    gen.add_argument("--p-staple", type=float, default=None)
    gen.add_argument("--novelty-ratio", type=float, default=None)
    gen.add_argument(
        "--cards-cdb-path",
        default=None,
        help=(
            "Optional path to cards.cdb; when provided, generation is restricted "
            "to IDs present in datas.id."
        ),
    )

    val = sub.add_parser("validate")
    val.add_argument("--decks-file", required=True)
    val.add_argument("--cards-file", required=True)
    val.add_argument("--format", required=True)
    val.add_argument("--format-config", required=True)

    score = sub.add_parser("score")
    score.add_argument("--decks-file", required=True)
    score.add_argument("--cards-file", required=True)
    score.add_argument("--output-dir", default=None)
    score.add_argument("--mlflow-tracking-uri", default=None)
    score.add_argument(
        "--mlflow-experiment",
        default="yugioh-deck-generator/generation-heuristic-scoring",
    )
    score.add_argument("--mlflow-run-name", default=None)
    score.add_argument("--run-id", default=None)

    return parser.parse_args()


def _load_allowed_ids_from_cdb(path: str | Path) -> set[int]:
    db_path = Path(path)
    if not db_path.exists():
        raise FileNotFoundError(f"cards.cdb not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT id FROM datas WHERE id > 0").fetchall()
    finally:
        conn.close()
    return {int(r[0]) for r in rows if r and int(r[0]) > 0}


def _cmd_generate(args: argparse.Namespace) -> int:
    cards_df = read_table(args.cards_file)
    clusters_df = read_table(args.cluster_assignments)
    formats = load_formats(args.format_config)
    staple_pools = load_staple_pools(args.staples_config)
    profiles = load_sampling_profiles(args.config)

    profile = next(iter(profiles.values()), {})
    p_staple = args.p_staple if args.p_staple is not None else float(profile.get("p_staple", 0.3))
    novelty_ratio = (
        args.novelty_ratio
        if args.novelty_ratio is not None
        else float(profile.get("novelty_ratio", 0.1))
    )
    allow_card_ids = None
    if args.cards_cdb_path:
        allow_card_ids = _load_allowed_ids_from_cdb(args.cards_cdb_path)

    run_generation(
        cards_df=cards_df,
        clusters_df=clusters_df,
        formats=formats,
        selected_formats=args.formats,
        decks_per_format=args.decks_per_format,
        mode=args.mode,
        staple_pools=staple_pools,
        p_staple=p_staple,
        novelty_ratio=novelty_ratio,
        output_dir=args.output_dir,
        resume_run_id=args.resume_run_id,
        seed=args.seed,
        parallel_formats=args.parallel_formats,
        allow_card_ids=allow_card_ids,
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    cards_df = read_table(args.cards_file)
    decks_df = read_table(args.decks_file)
    formats = load_formats(args.format_config)
    fmt = formats[args.format]

    failures = 0
    for _, row in decks_df.iterrows():
        ok, _, _ = validate_generated_deck(
            cards_df=cards_df,
            main_ids=[int(x) for x in row["main"]],
            extra_ids=[int(x) for x in row["extra"]],
            side_ids=[int(x) for x in row.get("side", [])],
            main_size=fmt.main_deck_size,
            extra_min=fmt.min_extra,
            extra_max=fmt.max_extra,
            banlist={},
            ratios={
                "monster": fmt.card_type_ratios.monster,
                "spell": fmt.card_type_ratios.spell,
                "trap": fmt.card_type_ratios.trap,
            },
            tolerance_count=fmt.card_type_ratios.tolerance_count,
        )
        if not ok:
            failures += 1

    return 1 if failures else 0


def _cmd_score(args: argparse.Namespace) -> int:
    return run_scoring(args)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
    args = parse_args()
    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "score":
        return _cmd_score(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
