from __future__ import annotations

import re
from typing import Any

import pandas as pd

_RACES = (
    "aqua",
    "beast",
    "beast-warrior",
    "cyberse",
    "dinosaur",
    "divine-beast",
    "dragon",
    "fairy",
    "fiend",
    "fish",
    "insect",
    "machine",
    "plant",
    "psychic",
    "pyro",
    "reptile",
    "rock",
    "sea serpent",
    "spellcaster",
    "thunder",
    "warrior",
    "winged beast",
    "wyrm",
    "zombie",
)
_RACE_PATTERN = "(?:" + "|".join(re.escape(r) for r in _RACES) + ")"
_QUOTED_ARCHETYPE_MONSTER = re.compile(r"[\"“”']([^\"“”']+)[\"“”']\s+monster", re.IGNORECASE)
_COUNTED_QUOTED_ARCHETYPE_MONSTER = re.compile(
    r"(?<![#\w])(\d+)\+?\s*(?:or\s+more\s+)?[\"“”']([^\"“”']+)[\"“”']\s+monsters?",
    re.IGNORECASE,
)
_COUNTED_QUOTED_ARCHETYPE = re.compile(
    r"(?<![#\w])(\d+)\+?\s*(?:or\s+more\s+)?[\"“”']([^\"“”']+)[\"“”']",
    re.IGNORECASE,
)
_COUNTED_TYPED_TUNER = re.compile(
    rf"(\d+)\+?\s*(?:or\s+more\s+)?({_RACE_PATTERN})\s*(?:-?\s*type)?\s+tuners?(?:\s+monsters?)?",
    re.IGNORECASE,
)
_ARTICLE_TYPED_TUNER = re.compile(
    rf"\b(?:a|an|one)\s+({_RACE_PATTERN})\s*(?:-?\s*type)?\s+tuners?(?:\s+monsters?)?",
    re.IGNORECASE,
)
_ATTRS = ("dark", "light", "wind", "water", "fire", "earth", "divine")
_COUNTED_ATTRIBUTE_MONSTER = re.compile(
    r"(\d+)\+?\s*(?:or\s+more\s+)?(dark|light|wind|water|fire|earth|divine)\s+monsters?",
    re.IGNORECASE,
)
_ARTICLE_ATTRIBUTE_MONSTER = re.compile(
    r"\b(?:a|an|one)\s+(dark|light|wind|water|fire|earth|divine)\s+monsters?",
    re.IGNORECASE,
)
_COUNTED_ATTRIBUTE_TUNER = re.compile(
    r"(\d+)\+?\s*(?:or\s+more\s+)?(dark|light|wind|water|fire|earth|divine)\s+tuners?(?:\s+monsters?)?",
    re.IGNORECASE,
)
_ARTICLE_ATTRIBUTE_TUNER = re.compile(
    r"\b(?:a|an|one)\s+(dark|light|wind|water|fire|earth|divine)\s+tuners?(?:\s+monsters?)?",
    re.IGNORECASE,
)
_COUNTED_QUALIFIED_ATTRIBUTE = re.compile(
    r"(\d+)\+?\s*(?:or\s+more\s+)?(dark|light|wind|water|fire|earth|divine)\s+(non[\s-]?tuners?|tuners?|monsters?)",
    re.IGNORECASE,
)
_ARTICLE_QUALIFIED_ATTRIBUTE = re.compile(
    r"\b(?:a|an|one)\s+(dark|light|wind|water|fire|earth|divine)\s+(non[\s-]?tuners?|tuners?|monsters?)",
    re.IGNORECASE,
)
_COUNTED_QUALIFIED_TYPE = re.compile(
    rf"(\d+)\+?\s*(?:or\s+more\s+)?({_RACE_PATTERN})\s*(?:-?\s*type)?\s+(non[\s-]?tuners?|tuners?|monsters?)",
    re.IGNORECASE,
)
_ARTICLE_QUALIFIED_TYPE = re.compile(
    rf"\b(?:a|an|one)\s+({_RACE_PATTERN})\s*(?:-?\s*type)?\s+(non[\s-]?tuners?|tuners?|monsters?)",
    re.IGNORECASE,
)
_COUNTED_QUALIFIED_QUOTED_ARCHETYPE = re.compile(
    r"(\d+)\+?\s*(?:or\s+more\s+)?(non[\s-]?tuners?|tuners?|monsters?)\s+[\"“”']([^\"“”']+)[\"“”']\s+monsters?",
    re.IGNORECASE,
)
_COUNTED_QUOTED_ARCHETYPE_QUALIFIED = re.compile(
    r"(\d+)\+?\s*(?:or\s+more\s+)?[\"“”']([^\"“”']+)[\"“”']\s+(non[\s-]?tuners?|tuners?|monsters?)",
    re.IGNORECASE,
)
_ARTICLE_QUALIFIED_QUOTED_ARCHETYPE = re.compile(
    r"\b(?:a|an|one)\s+(non[\s-]?tuners?|tuners?|monsters?)\s+[\"“”']([^\"“”']+)[\"“”']\s+monsters?",
    re.IGNORECASE,
)
_ARTICLE_QUOTED_ARCHETYPE_QUALIFIED = re.compile(
    r"\b(?:a|an|one)\s+[\"“”']([^\"“”']+)[\"“”']\s+(non[\s-]?tuners?|tuners?|monsters?)",
    re.IGNORECASE,
)
_MATERIAL_SPLIT = re.compile(r"\s*\+\s*")
_GENERIC_MATERIAL_PATTERNS = (
    re.compile(r"^\d+\+?\s+", re.IGNORECASE),
    re.compile(r"\b(?:non[\s-]?tuner|tuner|monster|monsters|fusion material)\b", re.IGNORECASE),
    re.compile(r"\b(?:type|attribute|or more|and)\b", re.IGNORECASE),
)


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_name(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value))


def _normalize_role(role_text: str | None) -> str:
    role = _text(role_text)
    if role.startswith("non"):
        return "non_tuner"
    if "tuner" in role:
        return "tuner"
    return "monster"


def _make_tag(prefix: str, value: str, role: str = "monster") -> str:
    val = _normalize_name(value)
    r = _normalize_role(role)
    if not val:
        return ""
    return f"{prefix}:{val}:{r}"


def _add_requirement(requirements: dict[str, int], tag: str, count: int) -> None:
    if not tag:
        return
    requirements[tag] = max(requirements.get(tag, 0), max(1, int(count)))


def _role_matches(card_type: str, role: str) -> bool:
    role_norm = _normalize_role(role)
    if "monster" not in card_type:
        return False
    if role_norm == "tuner":
        return "tuner" in card_type
    if role_norm == "non_tuner":
        return "tuner" not in card_type
    return True


def _extract_named_material_tokens(desc: str) -> list[str]:
    if "+" not in desc:
        return []
    candidates: list[str] = []
    for part in _MATERIAL_SPLIT.split(desc):
        token = part.strip(" \t\r\n.,;:()[]{}\"'`")
        if not token:
            continue
        token_norm = _normalize_name(token)
        if not token_norm:
            continue
        if any(p.search(token_norm) for p in _GENERIC_MATERIAL_PATTERNS):
            continue
        if any(ch.isdigit() for ch in token_norm):
            continue
        if len(token_norm.split()) < 2:
            continue
        candidates.append(token_norm)
    return candidates


def _material_clauses(desc: str) -> list[str]:
    text = _text(desc)
    if not text:
        return []
    # Summoning requirements are typically in the first sentence/segment.
    head = re.split(r"[;\n]", text, maxsplit=1)[0].strip()
    if not head:
        return []
    if "monster" not in head and "tuner" not in head and "+" not in head and "\"" not in head and "'" not in head:
        return []
    return [head]


def extract_required_monster_tag_counts(extra_rows: list[pd.Series]) -> dict[str, int]:
    requirements: dict[str, int] = {}
    for row in extra_rows:
        for desc in _material_clauses(row.get("desc")):
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
                if race and race not in _ATTRS:
                    key = f"type_tuner:{race}"
                    count = max(1, int(count_text))
                    requirements[key] = max(requirements.get(key, 0), count)
            for race_text in _ARTICLE_TYPED_TUNER.findall(desc):
                race = _text(race_text).replace("  ", " ")
                if race and race not in _ATTRS:
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
            for card_name in _extract_named_material_tokens(desc):
                key = f"card_name:{card_name}"
                requirements[key] = max(requirements.get(key, 0), 1)
            for count_text, attr_text, role_text in _COUNTED_QUALIFIED_ATTRIBUTE.findall(desc):
                attr = _text(attr_text)
                if attr in _ATTRS:
                    _add_requirement(
                        requirements,
                        _make_tag("attr", attr, role_text),
                        int(count_text),
                    )
            for attr_text, role_text in _ARTICLE_QUALIFIED_ATTRIBUTE.findall(desc):
                attr = _text(attr_text)
                if attr in _ATTRS:
                    _add_requirement(requirements, _make_tag("attr", attr, role_text), 1)
            for count_text, race_text, role_text in _COUNTED_QUALIFIED_TYPE.findall(desc):
                race = _normalize_name(race_text)
                if race and race not in _ATTRS:
                    _add_requirement(
                        requirements,
                        _make_tag("type", race, role_text),
                        int(count_text),
                    )
            for race_text, role_text in _ARTICLE_QUALIFIED_TYPE.findall(desc):
                race = _normalize_name(race_text)
                if race and race not in _ATTRS:
                    _add_requirement(requirements, _make_tag("type", race, role_text), 1)
            for count_text, role_text, arch_text in _COUNTED_QUALIFIED_QUOTED_ARCHETYPE.findall(desc):
                arch = _normalize_name(arch_text)
                if arch:
                    _add_requirement(
                        requirements,
                        _make_tag("archetype", arch, role_text),
                        int(count_text),
                    )
            for count_text, arch_text, role_text in _COUNTED_QUOTED_ARCHETYPE_QUALIFIED.findall(desc):
                arch = _normalize_name(arch_text)
                if arch:
                    _add_requirement(
                        requirements,
                        _make_tag("archetype", arch, role_text),
                        int(count_text),
                    )
            for role_text, arch_text in _ARTICLE_QUALIFIED_QUOTED_ARCHETYPE.findall(desc):
                arch = _normalize_name(arch_text)
                if arch:
                    _add_requirement(requirements, _make_tag("archetype", arch, role_text), 1)
            for arch_text, role_text in _ARTICLE_QUOTED_ARCHETYPE_QUALIFIED.findall(desc):
                arch = _normalize_name(arch_text)
                if arch:
                    _add_requirement(requirements, _make_tag("archetype", arch, role_text), 1)
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
    if tag_norm.startswith("card_name:"):
        required_name = _normalize_name(tag_norm.split(":", 1)[1])
        return bool(required_name) and _normalize_name(row.get("name")) == required_name
    if tag_norm.startswith("attr:"):
        _, attr, role = (tag_norm.split(":", 2) + ["monster"])[:3]
        if not _role_matches(card_type, role):
            return False
        attr_text = _text(row.get("attribute"))
        if attr_text:
            return attr in attr_text
        return attr in _text(row.get("desc")) or attr in card_type
    if tag_norm.startswith("type:"):
        _, race, role = (tag_norm.split(":", 2) + ["monster"])[:3]
        if not race or not _role_matches(card_type, role):
            return False
        race_text = _text(row.get("race"))
        if race_text and (race in race_text):
            return True
        return race in card_type or race in _text(row.get("name")) or race in _text(row.get("desc"))
    if tag_norm.startswith("archetype:"):
        _, arch, role = (tag_norm.split(":", 2) + ["monster"])[:3]
        if not arch or not _role_matches(card_type, role):
            return False
        archetype = _text(row.get("archetype"))
        if archetype and (arch == archetype or arch in archetype):
            return True
        for field in ("name", "desc", "type"):
            if arch in _text(row.get(field)):
                return True
        return False
    archetype = _text(row.get("archetype"))
    if archetype and (tag_norm == archetype or tag_norm in archetype):
        return True
    for field in ("name", "desc", "type"):
        if tag_norm in _text(row.get(field)):
            return True
    return False


def format_requirement_label(tag: str) -> str:
    tag_norm = _text(tag)
    role_suffix = ""
    if ":" in tag_norm:
        parts = tag_norm.split(":")
        if len(parts) == 3:
            role = _normalize_role(parts[2])
            if role == "tuner":
                role_suffix = " tuner"
            elif role == "non_tuner":
                role_suffix = " non-tuner"
            else:
                role_suffix = " monster"
    if tag_norm.startswith("attr_tuner:"):
        attr = tag_norm.split(":", 1)[1].strip()
        return f"{attr} tuner"
    if tag_norm.startswith("attr_monster:"):
        attr = tag_norm.split(":", 1)[1].strip()
        return f"{attr} monster"
    if tag_norm.startswith("type_tuner:"):
        race = tag_norm.split(":", 1)[1].strip()
        return f"{race} tuner"
    if tag_norm.startswith("card_name:"):
        return tag_norm.split(":", 1)[1].strip()
    if tag_norm.startswith("attr:"):
        attr = tag_norm.split(":", 2)[1].strip()
        return f"{attr}{role_suffix}"
    if tag_norm.startswith("type:"):
        race = tag_norm.split(":", 2)[1].strip()
        return f"{race}{role_suffix}"
    if tag_norm.startswith("archetype:"):
        arch = tag_norm.split(":", 2)[1].strip()
        return f"{arch}{role_suffix}"
    return tag_norm
