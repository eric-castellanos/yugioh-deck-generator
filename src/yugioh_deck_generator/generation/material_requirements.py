from __future__ import annotations

import re
from typing import Any

import pandas as pd

_QUOTED_ARCHETYPE_MONSTER = re.compile(r"[\"“”']([^\"“”']+)[\"“”']\s+monster", re.IGNORECASE)
_COUNTED_QUOTED_ARCHETYPE_MONSTER = re.compile(
    r"(\d+)\s+[\"“”']([^\"“”']+)[\"“”']\s+monsters?",
    re.IGNORECASE,
)
_COUNTED_QUOTED_ARCHETYPE = re.compile(r"(\d+)\s+[\"“”']([^\"“”']+)[\"“”']", re.IGNORECASE)
_COUNTED_TYPED_TUNER = re.compile(
    r"(\d+)\s+([a-z][a-z\s-]*?)\s*(?:-?\s*type)?\s+tuners?(?:\s+monsters?)?",
    re.IGNORECASE,
)
_ARTICLE_TYPED_TUNER = re.compile(
    r"\b(?:a|an|one)\s+([a-z][a-z\s-]*?)\s*(?:-?\s*type)?\s+tuners?(?:\s+monsters?)?",
    re.IGNORECASE,
)
_ATTRS = ("dark", "light", "wind", "water", "fire", "earth", "divine")
_COUNTED_ATTRIBUTE_MONSTER = re.compile(
    r"(\d+)\s+(dark|light|wind|water|fire|earth|divine)\s+monsters?",
    re.IGNORECASE,
)
_ARTICLE_ATTRIBUTE_MONSTER = re.compile(
    r"\b(?:a|an|one)\s+(dark|light|wind|water|fire|earth|divine)\s+monsters?",
    re.IGNORECASE,
)
_COUNTED_ATTRIBUTE_TUNER = re.compile(
    r"(\d+)\s+(dark|light|wind|water|fire|earth|divine)\s+tuners?(?:\s+monsters?)?",
    re.IGNORECASE,
)
_ARTICLE_ATTRIBUTE_TUNER = re.compile(
    r"\b(?:a|an|one)\s+(dark|light|wind|water|fire|earth|divine)\s+tuners?(?:\s+monsters?)?",
    re.IGNORECASE,
)


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def extract_required_monster_tag_counts(extra_rows: list[pd.Series]) -> dict[str, int]:
    requirements: dict[str, int] = {}
    for row in extra_rows:
        desc = _text(row.get("desc"))
        for count_text, attr_text in _COUNTED_ATTRIBUTE_TUNER.findall(desc):
            attr = _text(attr_text)
            if attr in _ATTRS:
                key = f"attr_tuner:{attr}"
                requirements[key] = max(requirements.get(key, 0), max(1, int(count_text)))
        for attr_text in _ARTICLE_ATTRIBUTE_TUNER.findall(desc):
            attr = _text(attr_text)
            if attr in _ATTRS:
                key = f"attr_tuner:{attr}"
                requirements[key] = max(requirements.get(key, 0), 1)
        for count_text, attr_text in _COUNTED_ATTRIBUTE_MONSTER.findall(desc):
            attr = _text(attr_text)
            if attr in _ATTRS:
                key = f"attr_monster:{attr}"
                requirements[key] = max(requirements.get(key, 0), max(1, int(count_text)))
        for attr_text in _ARTICLE_ATTRIBUTE_MONSTER.findall(desc):
            attr = _text(attr_text)
            if attr in _ATTRS:
                key = f"attr_monster:{attr}"
                requirements[key] = max(requirements.get(key, 0), 1)
        for count_text, race_text in _COUNTED_TYPED_TUNER.findall(desc):
            race = _text(race_text).replace("  ", " ")
            if race:
                key = f"type_tuner:{race}"
                count = max(1, int(count_text))
                requirements[key] = max(requirements.get(key, 0), count)
        for race_text in _ARTICLE_TYPED_TUNER.findall(desc):
            race = _text(race_text).replace("  ", " ")
            if race:
                key = f"type_tuner:{race}"
                requirements[key] = max(requirements.get(key, 0), 1)
        for count_text, tag_text in _COUNTED_QUOTED_ARCHETYPE_MONSTER.findall(desc):
            tag = _text(tag_text)
            if tag:
                count = max(1, int(count_text))
                requirements[tag] = max(requirements.get(tag, 0), count)
        for count_text, tag_text in _COUNTED_QUOTED_ARCHETYPE.findall(desc):
            tag = _text(tag_text)
            if tag and tag not in requirements:
                count = max(1, int(count_text))
                requirements[tag] = count
        for match in _QUOTED_ARCHETYPE_MONSTER.findall(desc):
            tag = _text(match)
            if tag and tag not in requirements:
                requirements[tag] = 1
    return requirements


def main_monster_matches_tag(row: pd.Series, tag: str) -> bool:
    tag_norm = _text(tag)
    if not tag_norm:
        return False
    card_type = _text(row.get("type"))
    if "monster" not in card_type:
        return False
    if tag_norm.startswith("attr_tuner:"):
        attr = tag_norm.split(":", 1)[1].strip()
        if "tuner" not in card_type:
            return False
        attr_text = _text(row.get("attribute"))
        if attr_text:
            return attr in attr_text
        return attr in _text(row.get("desc")) or attr in card_type
    if tag_norm.startswith("attr_monster:"):
        attr = tag_norm.split(":", 1)[1].strip()
        attr_text = _text(row.get("attribute"))
        if attr_text:
            return attr in attr_text
        return attr in _text(row.get("desc")) or attr in card_type
    if tag_norm.startswith("type_tuner:"):
        race = tag_norm.split(":", 1)[1].strip()
        if not race:
            return False
        race_text = _text(row.get("race"))
        if "tuner" not in card_type:
            return False
        if race_text and (race in race_text):
            return True
        return race in card_type or race in _text(row.get("name")) or race in _text(row.get("desc"))
    archetype = _text(row.get("archetype"))
    if archetype and (tag_norm == archetype or tag_norm in archetype):
        return True
    for field in ("name", "desc", "type"):
        if tag_norm in _text(row.get(field)):
            return True
    return False


def format_requirement_label(tag: str) -> str:
    tag_norm = _text(tag)
    if tag_norm.startswith("attr_tuner:"):
        attr = tag_norm.split(":", 1)[1].strip()
        return f"{attr} tuner"
    if tag_norm.startswith("attr_monster:"):
        attr = tag_norm.split(":", 1)[1].strip()
        return f"{attr} monster"
    if tag_norm.startswith("type_tuner:"):
        race = tag_norm.split(":", 1)[1].strip()
        return f"{race} tuner"
    return tag_norm
