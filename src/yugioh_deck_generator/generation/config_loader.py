from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from yugioh_deck_generator.generation.schemas import FormatConfig, RatioConfig

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def _to_int(value: Any, *, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"PyYAML is required to parse YAML config: {path}. Install with `poetry add pyyaml`."
        ) from exc
    logger.debug("Loading YAML config from %s", path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be an object: {path}")
    logger.debug("Loaded YAML config from %s with top-level keys=%s", path, list(data.keys()))
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    logger.info("Loading config file: %s", p)
    if not p.exists():
        logger.error("Config file does not exist: %s", p)
        raise FileNotFoundError(p)
    if p.suffix.lower() in {".json"}:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Config root must be an object: {p}")
        logger.debug("Loaded JSON config from %s with top-level keys=%s", p, list(raw.keys()))
        return raw
    if p.suffix.lower() in {".yml", ".yaml"}:
        return _load_yaml(p)
    logger.error("Unsupported config extension `%s` for %s", p.suffix, p)
    raise ValueError(f"Unsupported config extension: {p.suffix}")


def load_formats(path: str | Path) -> dict[str, FormatConfig]:
    data = load_config(path)
    raw_formats = data.get("formats", data)
    if not isinstance(raw_formats, dict):
        raise ValueError("`formats` config must be a mapping")

    result: dict[str, FormatConfig] = {}
    for name, raw in raw_formats.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Format `{name}` must be an object")
        ratios_raw = raw.get("card_type_ratios", {})
        ratios = RatioConfig(
            monster=float(ratios_raw.get("monster", 0.5)),
            spell=float(ratios_raw.get("spell", 0.3)),
            trap=float(ratios_raw.get("trap", 0.2)),
            tolerance_count=_to_int(ratios_raw.get("tolerance_count"), fallback=2),
        )
        main_deck_size = _to_int(raw.get("main_deck_size", raw.get("deck_size")), fallback=40)
        extra_deck_size = _to_int(raw.get("extra_deck_size"), fallback=15)
        result[name] = FormatConfig(
            name=name,
            main_deck_size=main_deck_size,
            extra_deck_size=extra_deck_size,
            min_main=_to_int(
                raw.get("min_main", raw.get("main_deck_size")), fallback=main_deck_size
            ),
            max_main=_to_int(
                raw.get("max_main", raw.get("main_deck_size")), fallback=main_deck_size
            ),
            min_extra=_to_int(raw.get("min_extra"), fallback=0),
            max_extra=_to_int(
                raw.get("max_extra", raw.get("extra_deck_size")), fallback=extra_deck_size
            ),
            legality_source=raw.get("legality_source"),
            card_type_ratios=ratios,
            generation=raw.get("generation", {}),
        )

    logger.info("Loaded %d format configs from %s", len(result), path)
    return result


def load_staple_pools(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    data = load_config(path)
    pools = data.get("staple_pools", data)
    if not isinstance(pools, dict):
        raise ValueError("`staple_pools` config must be a mapping")
    out: dict[str, list[dict[str, Any]]] = {}
    for fmt, entries in pools.items():
        if not isinstance(entries, list):
            raise ValueError(f"Staple pool `{fmt}` must be a list")
        out[fmt] = entries
    logger.info("Loaded staple pools for %d formats from %s", len(out), path)
    return out


def load_sampling_profiles(path: str | Path) -> dict[str, dict[str, Any]]:
    data = load_config(path)
    profiles = data.get("sampling_profiles", data)
    if not isinstance(profiles, dict):
        raise ValueError("`sampling_profiles` config must be a mapping")
    parsed = {k: v for k, v in profiles.items() if isinstance(v, dict)}
    logger.info("Loaded %d sampling profiles from %s", len(parsed), path)
    return parsed
