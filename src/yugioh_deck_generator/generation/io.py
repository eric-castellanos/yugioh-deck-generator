from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    logger.info("Reading table from %s", p)
    if suffix == ".parquet":
        df = pd.read_parquet(p)
        logger.info("Loaded parquet rows=%d cols=%d", len(df), len(df.columns))
        return df
    if suffix in {".csv"}:
        df = pd.read_csv(p)
        logger.info("Loaded csv rows=%d cols=%d", len(df), len(df.columns))
        return df
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            df = pd.read_json(p, orient="records", lines=True)
            logger.info("Loaded jsonl rows=%d cols=%d", len(df), len(df.columns))
            return df
        df = pd.read_json(p)
        logger.info("Loaded json rows=%d cols=%d", len(df), len(df.columns))
        return df
    logger.error("Unsupported table extension `%s` for %s", suffix, p)
    raise ValueError(f"Unsupported table extension: {suffix}")


def read_banlist(path: str | Path | None) -> dict[str, list[int]]:
    if path is None:
        logger.info("No banlist path provided; using empty banlist")
        return {}
    if isinstance(path, str) and not path.strip():
        logger.info("Empty banlist path provided; using empty banlist")
        return {}
    p = Path(path)
    if p.is_dir():
        logger.warning("Banlist path is a directory (%s); using empty banlist", p)
        return {}
    if not p.exists():
        logger.warning("Banlist path does not exist: %s; using empty banlist", p)
        return {}
    logger.info("Reading banlist from %s", p)

    if p.suffix.lower() in {".json"}:
        raw = json.loads(p.read_text(encoding="utf-8"))
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required to parse banlist YAML: "
                f"{path}. Install with `poetry add pyyaml`."
            ) from exc
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        logger.warning("Banlist payload is not a mapping: %s; using empty banlist", p)
        return {}

    def _ids(key: str) -> list[int]:
        val = raw.get(key, [])
        if not isinstance(val, list):
            return []
        return [int(x) for x in val]

    parsed = {
        "forbidden": _ids("forbidden"),
        "limited": _ids("limited"),
        "semi_limited": _ids("semi_limited"),
    }
    logger.info(
        "Loaded banlist forbidden=%d limited=%d semi_limited=%d",
        len(parsed["forbidden"]),
        len(parsed["limited"]),
        len(parsed["semi_limited"]),
    )
    return parsed


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing jsonl rows=%d to %s", len(rows), p)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def append_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Appending jsonl rows=%d to %s", len(rows), p)
    with p.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_summary(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing summary with keys=%d to %s", len(payload.keys()), p)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_ydk(
    path: str | Path, main: list[int], extra: list[int], side: list[int] | None = None
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    side_ids = side or []
    logger.info(
        "Writing ydk to %s main=%d extra=%d side=%d",
        p,
        len(main),
        len(extra),
        len(side_ids),
    )

    lines = ["#main"]
    lines.extend(str(cid) for cid in main)
    lines.append("#extra")
    lines.extend(str(cid) for cid in extra)
    lines.append("!side")
    lines.extend(str(cid) for cid in side_ids)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
