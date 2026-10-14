"""Reason text: the structure check of configs/reason_phrases.yaml and the renderer.

The structure check runs before any rendering; the renderer refuses a phrases
file with any problem. It checks that every slot is filled, that the groups and
fact keys are the ones configs/explain.yaml defines, that every placeholder is
one the renderer supports in that slot, and the wording rules stated in the
file's header (shape of group phrases; digits, comparison, causal and advice
words; nothing hidden named; no decision ID or private file named anywhere in
the file, comments included).

One explanation per customer, snapshot and horizon, in this order:
  1. position (higher or lower with its strength, or close);
  2. reasons pushing risk up, or a nothing-notable line;
  3. reasons pulling risk down, or a nothing-notable line (a close position
     with nothing listed either way gets one line for both);
  4. other factors, only when at least one reason is listed;
  5. the partly-cancel line, only when the position is weaker than the
     strongest listed reason and a pull the other way is visible;
  6. the probability label, then the caveats that apply.
"""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from retention.explain.config import STRENGTHS, ExplainConfig
from retention.explain.selection import Selection, strength_codes
from retention.models.metrics import CLIP, logit

PHRASES = Path("configs/reason_phrases.yaml")

TOP_SLOTS = {
    "position": ("higher", "lower", "close"),
    "nothing_notable": (
        "raising",
        "lowering",
        "many_small_raising",
        "many_small_lowering",
        "none_stand_out",
    ),
    "other_factors": ("raises", "lowers", "small"),
    "partly_cancel": ("close", "higher", "lower"),
}
CAVEATS = ("october_2016_calibration_half", "months_used_for_settings", "february_2017_gap")
# Placeholders each slot must use (exactly these; none elsewhere).
SLOT_PLACEHOLDERS = {
    "reason_line": {"phrase", "strength", "facts"},
    "position.higher": {"strength"},
    "position.lower": {"strength"},
    "other_factors.raises": {"strength"},
    "other_factors.lowers": {"strength"},
    "probability_label": {"horizon", "probability"},
}
FACT_PLACEHOLDERS = {"value", "days", "minutes", "tenure"}
WHEN_KEYS = ("when_zero", "when_missing", "when_tiny")
BOOLEAN_KEYS = ("when_true", "when_false")
CATEGORY_LEVELS = {"last_row_kind": ("paid", "discount100", "promo", "trial")}

COMPARISON_WORDS = r"\b(more|less|higher|lower|than|below|above|over|under|fewer)\b"
CAUSAL_WORDS = r"\b(causes?|caused|because|due to|leads? to|drives?|results? in)\b"
ADVICE_WORDS = r"\b(should|turn on|switch|increase|improve|reduce|try|consider)\b"
HIDDEN_WORDS = r"\b(registered|registration|channel)\b"
DECISION_ID = r"\bD-\d{3}\b"
EXCLUDE = Path(".git/info/exclude")
RAISES_SHAPE = re.compile(r"[A-Z].* is pushing risk up")
LOWERS_SHAPE = re.compile(r"[A-Z].* is pulling risk down")


# --- the structure check ------------------------------------------------------------------


def placeholders(text: str) -> list[tuple[str, str]]:
    """(name, format spec) of every placeholder in a format string."""
    return [(f, spec or "") for _, f, spec, _ in string.Formatter().parse(text) if f is not None]


def _values(node: Any, path: str = ""):
    """(dotted path, value) of every leaf of the phrases mapping."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _values(v, f"{path}.{k}" if path else str(k))
    else:
        yield path, node


def private_names(exclude: Path = EXCLUDE) -> list[str]:
    """Names of the private files kept out of the repository by the local git exclude file.

    Each excluded entry gives its name without extension; an excluded folder that
    exists also gives the names of the files inside it. The names are read here,
    never written in the code. Without the file (a fresh clone, CI) the list is empty.
    """
    if not exclude.exists():
        return []
    root = exclude.parent.parent.parent
    names: set[str] = set()
    for line in exclude.read_text().splitlines():
        entry = line.strip().strip("/")
        if not entry or line.lstrip().startswith("#"):
            continue
        names.add(Path(entry).stem.lstrip("."))
        folder = root / entry
        if folder.is_dir():
            names |= {f.stem for f in folder.iterdir() if f.is_file()}
    return sorted(n for n in names if n)


def structure_problems(
    raw_text: str, cfg: ExplainConfig, private: list[str] | None = None
) -> list[str]:
    """Every problem with the phrases file; empty when it may be used."""
    problems: list[str] = []
    for m in re.finditer(DECISION_ID, raw_text):
        problems.append(f"the file names a decision ID: {m.group(0)!r}")
    for name in private_names() if private is None else private:
        if re.search(rf"\b{re.escape(name)}\b", raw_text, flags=re.IGNORECASE):
            problems.append("the file names a private file")
    doc = yaml.safe_load(raw_text)
    for key in ("reason_line", "facts_separator", "missing_value", "probability_label"):
        if not isinstance(doc.get(key), str) or (key != "facts_separator" and not doc[key]):
            problems.append(f"{key} is not filled")
    for section, slots in TOP_SLOTS.items():
        for s in slots:
            if not doc.get(section, {}).get(s):
                problems.append(f"{section}.{s} is not filled")
    if set(doc.get("caveats", {})) != set(CAVEATS) or not all(doc["caveats"].values()):
        problems.append(f"caveats must be exactly {CAVEATS}, all filled")

    shown = [g for g in cfg.group_names if g not in cfg.hidden]
    groups = doc.get("groups", {})
    if set(groups) != set(shown):
        problems.append(f"groups {sorted(groups)} are not the shown groups {sorted(shown)}")
    for g, spec in groups.items():
        for d, shape in (("raises", RAISES_SHAPE), ("lowers", LOWERS_SHAPE)):
            text = spec.get(d) or ""
            if not shape.fullmatch(text):
                problems.append(f"groups.{g}.{d} does not have the shape: {text!r}")
            if re.search(r"\d", text):
                problems.append(f"groups.{g}.{d} has a digit")
            if re.search(COMPARISON_WORDS, text, re.IGNORECASE):
                problems.append(f"groups.{g}.{d} has a comparison word")
        facts = spec.get("facts", {})
        expected = cfg.facts.get(g, [])
        if sorted(facts) != sorted(expected):
            problems.append(f"groups.{g} facts {sorted(facts)} are not explain.yaml's {expected}")
        for col, fspec in facts.items():
            problems += _fact_problems(f"groups.{g}.facts.{col}", col, fspec)

    for path, value in _values(doc):
        if value is None:
            if path.split(".")[-1] not in (*WHEN_KEYS, *BOOLEAN_KEYS):
                problems.append(f"{path} is empty")
            continue
        if not isinstance(value, str):
            problems.append(f"{path} is not text")
            continue
        for words, what in ((CAUSAL_WORDS, "causal"), (ADVICE_WORDS, "advice")):
            if re.search(words, value, re.IGNORECASE):
                problems.append(f"{path} has a {what} word")
        hidden_cols = [c for h in cfg.hidden for c in cfg.groups[h]]
        if re.search(HIDDEN_WORDS, value, re.IGNORECASE) or any(c in value for c in hidden_cols):
            problems.append(f"{path} names something hidden")
        top = path.split(".")[0]
        if top in ("position", "nothing_notable", "other_factors", "partly_cancel"):
            if re.search(r"\d", value):
                problems.append(f"{path} has a digit")
        if top not in ("position", "partly_cancel") and re.search(
            r"\b(higher|lower) than\b", value, re.IGNORECASE
        ):
            problems.append(f"{path} says 'higher than' or 'lower than'")
        if ".facts." in path:
            continue
        names = {n for n, _ in placeholders(value)}
        want = SLOT_PLACEHOLDERS.get(path, set())
        if names != want:
            problems.append(f"{path} has placeholders {sorted(names)}, expected {sorted(want)}")
    return problems


def _fact_problems(path: str, col: str, spec: Any) -> list[str]:
    out = []
    if isinstance(spec, str):
        spec = {"format": spec}
    if not isinstance(spec, dict):
        return [f"{path} is neither a format string nor a mapping"]
    if col in CATEGORY_LEVELS:
        missing = [lv for lv in CATEGORY_LEVELS[col] if not spec.get(lv)]
        return [f"{path} has no text for levels {missing}"] if missing else []
    if any(k in spec for k in BOOLEAN_KEYS):
        if set(spec) != set(BOOLEAN_KEYS):
            out.append(f"{path} must have exactly when_true and when_false")
        return out
    unknown = set(spec) - {"format", *WHEN_KEYS}
    if unknown:
        out.append(f"{path} has unknown keys {sorted(unknown)}")
    fmt = spec.get("format")
    if not fmt:
        return [*out, f"{path} has no format"]
    for name, fspec in placeholders(fmt):
        if name not in FACT_PLACEHOLDERS:
            out.append(f"{path} uses unsupported placeholder {{{name}}}")
        elif fspec and name != "value":
            out.append(f"{path}: only {{value}} takes a format spec")
        elif fspec:
            try:
                format(1.0, fspec)
            except ValueError:
                out.append(f"{path}: bad format spec {fspec!r}")
    if "when_tiny" in spec and "minutes" not in {n for n, _ in placeholders(fmt)}:
        out.append(f"{path}: when_tiny is only for minutes")
    for k in WHEN_KEYS:
        if spec.get(k) is not None and placeholders(spec[k]):
            out.append(f"{path}.{k} may not use placeholders")
    return out


# --- rendering facts --------------------------------------------------------------------------


def _missing(v: Any) -> bool:
    return v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v))


def _plain(v: Any) -> Any:
    if _missing(v):
        return None
    return v.item() if hasattr(v, "item") else v


def plural(n: int, unit: str) -> str:
    return f"{n} {unit}" if n == 1 else f"{n} {unit}s"


def whole_minutes(seconds: float) -> int:
    """Seconds as whole minutes, rounded half up (30 s -> 1, 89 s -> 1, 90 s -> 2)."""
    return int(math.floor(seconds / 60 + 0.5))


def tenure_words(days: int) -> str:
    """Whole years and months from a day count: 213 -> '7 months', 825 -> '2 years and 3 months'."""
    total = int(days) * 12 // 365
    years, months = divmod(total, 12)
    if total == 0:
        return "less than a month"
    parts = ([plural(years, "year")] if years else []) + (
        [plural(months, "month")] if months else []
    )
    return " and ".join(parts)


def render_fact(spec: Any, value: Any, missing_text: str, column: str = "") -> str | None:
    """One fact as text, or None when the spec leaves it out."""
    value = _plain(value)
    if isinstance(spec, str):
        spec = {"format": spec}
    if column in CATEGORY_LEVELS:
        return missing_text if value is None else spec[str(value)]
    if "when_true" in spec:
        if value is None:
            return missing_text
        return spec["when_true"] if int(value) == 1 else spec["when_false"]
    if value is None:
        return spec["when_missing"] if "when_missing" in spec else missing_text
    if value == 0 and "when_zero" in spec:
        return spec["when_zero"]
    minutes = whole_minutes(float(value))
    if "when_tiny" in spec and value > 0 and minutes == 0:
        return spec["when_tiny"]
    names = {n for n, _ in placeholders(spec["format"])}
    fields: dict[str, Any] = {"value": value}
    if "days" in names:
        fields["days"] = plural(int(value), "day")
    if "minutes" in names:
        fields["minutes"] = plural(minutes, "minute")
    if "tenure" in names:
        fields["tenure"] = tenure_words(int(value))
    return spec["format"].format(**fields)


def facts_text(parts: list[str | None], separator: str) -> str:
    """Join the facts that are not left out; capital first letter; ends with a full stop."""
    kept = [p for p in parts if p]
    if not kept:
        return ""
    text = separator.join(kept)
    text = text[0].upper() + text[1:]
    return text if text.endswith(".") else text + "."


# --- rendering one explanation ----------------------------------------------------------------


@dataclass
class Rendered:
    lines: list[str]
    listed: list[tuple[str, int, float]] = field(default_factory=list)  # group, strength, c
    other: float = 0.0
    d: float = 0.0
    partly_cancel: bool = False


def probability_text(p: float) -> str:
    """A probability for display: one decimal, except below 0.1% ('under 0.1%') and above 99.9%
    ('over 99.9%'), so a small chance never reads as 0.0% and a large one never as 100.0%."""
    if p < 0.001:
        return "under 0.1%"
    if p > 0.999:
        return "over 99.9%"
    return f"{100 * p:.1f}%"


def platt_probability(raw_score: np.ndarray, a: float, b: float) -> np.ndarray:
    """Platt-calibrated probability from the raw score, clipped as the test stage clips it
    (before the cross-horizon ordering fix)."""
    p = 1 / (1 + np.exp(-np.asarray(raw_score, dtype=float)))
    z, _ = logit(p)
    return np.clip(1 / (1 + np.exp(-(a + b * z))), CLIP, 1 - CLIP)


class Renderer:
    def __init__(self, cfg: ExplainConfig, phrases_text: str):
        problems = structure_problems(phrases_text, cfg)
        if problems:
            raise ValueError(
                "configs/reason_phrases.yaml has problems:\n  " + "\n  ".join(problems)
            )
        self.cfg = cfg
        self.p = yaml.safe_load(phrases_text)

    @classmethod
    def from_files(cls, cfg: ExplainConfig, path: Path = PHRASES) -> Renderer:
        return cls(cfg, path.read_text())

    def reason_line(self, group: str, direction: str, strength: int, facts: dict[str, Any]) -> str:
        spec = self.p["groups"][group]
        parts = [
            render_fact(spec["facts"][col], facts.get(col), self.p["missing_value"], col)
            for col in self.cfg.facts[group]
        ]
        text = self.p["reason_line"].format(
            phrase=spec[direction],
            strength=STRENGTHS[strength - 1],
            facts=facts_text(parts, self.p["facts_separator"]),
        )
        return text.rstrip()

    def render(
        self,
        sel: Selection,
        i: int,
        facts: dict[str, Any],
        horizon: int,
        probability: float,
        caveats: tuple[str, ...] = (),
    ) -> Rendered:
        p, names = self.p, self.cfg.group_names
        lines: list[str] = []
        pos = int(sel.position[i])
        if pos == 0:
            lines.append(p["position"]["close"])
        else:
            s = int(strength_codes(np.abs(sel.d[i : i + 1]), self.cfg.cutoffs())[0])
            key = "higher" if pos == 1 else "lower"
            lines.append(p["position"][key].format(strength=STRENGTHS[s - 1]))
        listed: list[tuple[str, int, float]] = []
        nn = p["nothing_notable"]
        # Row values only: the Selection's whole-array helpers would cost a pass over
        # every row for each rendered row.
        n_up, n_dn = int((sel.raising[i] >= 0).sum()), int((sel.lowering[i] >= 0).sum())
        if pos == 0 and n_up == 0 and n_dn == 0:
            lines.append(nn["none_stand_out"])
        else:
            for idx, strengths, direction, many, nothing in (
                (sel.raising, sel.raising_strength, "raises", pos == 1 and n_up == 0, "raising"),
                (
                    sel.lowering,
                    sel.lowering_strength,
                    "lowers",
                    pos == -1 and n_dn == 0,
                    "lowering",
                ),
            ):
                slots = [
                    (int(j), int(s)) for j, s in zip(idx[i], strengths[i], strict=True) if j >= 0
                ]
                for j, s in slots:
                    lines.append(self.reason_line(names[j], direction, s, facts))
                    listed.append((names[j], s, float(sel.c[i, j])))
                if not slots:
                    lines.append(nn[f"many_small_{nothing}"] if many else nn[nothing])
        if listed:
            o, os_ = float(sel.other[i]), int(sel.other_strength[i])
            if os_ == 0:
                lines.append(p["other_factors"]["small"])
            else:
                key = "raises" if o > 0 else "lowers"
                lines.append(p["other_factors"][key].format(strength=STRENGTHS[os_ - 1]))
        cancel = partly_cancel_shown(sel, i, self.cfg)
        if cancel:
            lines.append(p["partly_cancel"][{0: "close", 1: "higher", -1: "lower"}[pos]])
        lines.append(
            p["probability_label"].format(
                horizon=horizon, probability=probability_text(probability)
            )
        )
        lines += [p["caveats"][c] for c in caveats]
        return Rendered(lines, listed, float(sel.other[i]), float(sel.d[i]), cancel)


def partly_cancel_shown(sel: Selection, i: int, cfg: ExplainConfig) -> bool:
    """Position weaker than the strongest listed reason, and a visible pull the other way."""
    top = max(int(sel.raising_strength[i, 0]), int(sel.lowering_strength[i, 0]))
    if top == 0:
        return False
    pos = int(sel.position[i])
    pos_s = 0 if pos == 0 else int(strength_codes(np.abs(sel.d[i : i + 1]), cfg.cutoffs())[0])
    if pos_s >= top:
        return False
    c_up = float(sel.c[i, sel.raising[i, 0]]) if sel.raising[i, 0] >= 0 else 0.0
    c_dn = float(-sel.c[i, sel.lowering[i, 0]]) if sel.lowering[i, 0] >= 0 else 0.0
    strongest = 1 if c_up >= c_dn else -1
    other_side = sel.lowering[i] if strongest == 1 else sel.raising[i]
    listed_other_way = bool((other_side >= 0).any())
    other_other_way = bool(sel.other_strength[i] > 0 and np.sign(sel.other[i]) == -strongest)
    return bool(listed_other_way or other_other_way)
