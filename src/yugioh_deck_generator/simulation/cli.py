from __future__ import annotations

import argparse
import logging
from pathlib import Path

from yugioh_deck_generator.generation.io import read_table
from yugioh_deck_generator.simulation.aggregation import aggregate_deck_win_rates
from yugioh_deck_generator.simulation.io import (
    build_summary,
    write_deck_aggregates,
    write_duel_results,
    write_duel_results_parquet,
    write_requests,
    write_summary_json,
)
from yugioh_deck_generator.simulation.request_builder import build_simulation_requests
from yugioh_deck_generator.simulation.runner import run_simulations

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 duel simulation CLI")
    parser.add_argument("--decks-file", required=True)
    parser.add_argument("--generation-run-id", default=None)
    parser.add_argument("--opponent-dir", default="data/opponent_decks")
    parser.add_argument("--scenario", default="phase5_round_robin_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument(
        "--engine-mode",
        default="stub",
        choices=["stub", "windbot", "ygoenv"],
    )
    parser.add_argument("--engine-version", default="stub-v1")
    parser.add_argument("--windbot-exe-path", default=None)
    parser.add_argument("--windbot-cards-cdb-path", default=None)
    parser.add_argument("--windbot-mono-path", default="mono")
    parser.add_argument("--windbot-host", default="127.0.0.1")
    parser.add_argument("--windbot-port", type=int, default=7911)
    parser.add_argument("--windbot-host-info", default="")
    parser.add_argument("--windbot-dialog", default="default")
    parser.add_argument("--windbot-chat", action="store_true")
    parser.add_argument("--windbot-debug", action="store_true")
    parser.add_argument("--windbot-timeout-sec", type=int, default=30)
    parser.add_argument("--ygoenv-task-id", default="YGOPro-v1")
    parser.add_argument("--ygoenv-db-path", default=None)
    parser.add_argument("--ygoenv-code-list-file", default=None)
    parser.add_argument("--ygoenv-max-steps", type=int, default=2000)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    decks_df = read_table(args.decks_file)
    run_id = args.generation_run_id
    if not run_id:
        deck_path = Path(args.decks_file)
        run_id = deck_path.parent.name

    out_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path("data/generated/simulations") / str(run_id)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    requests = build_simulation_requests(
        decks_df=decks_df,
        generation_run_id=str(run_id),
        opponent_dir=args.opponent_dir,
        output_dir=out_root,
        scenario=args.scenario,
        base_seed=args.seed,
    )

    results = run_simulations(
        requests=requests,
        output_dir=out_root,
        engine_mode=args.engine_mode,
        engine_version=args.engine_version,
        engine_options={
            "windbot_exe_path": args.windbot_exe_path,
            "windbot_cards_cdb_path": args.windbot_cards_cdb_path,
            "windbot_mono_path": args.windbot_mono_path,
            "windbot_host": args.windbot_host,
            "windbot_port": args.windbot_port,
            "windbot_host_info": args.windbot_host_info,
            "windbot_dialog": args.windbot_dialog,
            "windbot_chat": bool(args.windbot_chat),
            "windbot_debug": bool(args.windbot_debug),
            "windbot_timeout_sec": args.windbot_timeout_sec,
            "ygoenv_task_id": args.ygoenv_task_id,
            "ygoenv_db_path": args.ygoenv_db_path,
            "ygoenv_code_list_file": args.ygoenv_code_list_file,
            "ygoenv_max_steps": args.ygoenv_max_steps,
        },
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
    )

    aggregates = aggregate_deck_win_rates(
        requests=requests,
        results=results,
        scenario=args.scenario,
    )

    requests_path = out_root / "simulation_requests.jsonl"
    results_jsonl_path = out_root / "duel_results.jsonl"
    results_parquet_path = out_root / "duel_results.parquet"
    aggregates_path = out_root / "deck_aggregate_metrics.jsonl"
    summary_path = out_root / "summary.json"

    write_requests(requests_path, requests)
    write_duel_results(results_jsonl_path, results)
    write_duel_results_parquet(results_parquet_path, results)
    write_deck_aggregates(aggregates_path, aggregates)

    summary = build_summary(
        requests=requests,
        results=results,
        aggregates=aggregates,
        scenario=args.scenario,
    )
    write_summary_json(summary_path, summary)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
