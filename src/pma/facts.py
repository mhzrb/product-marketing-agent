"""The fact base: what the copy is *allowed* to say, derived from the (sanitised) product data.

Everything the evaluator calls "supported" is defined here:

* every number in the copy must occur in the product data (normalised, so ``1.299,00`` /
  ``1,299.00`` / ``1299`` are the same number), and
* every ``number + unit`` pair (``16 GB``, ``27 inch``, ``65 W``) must occur as the same pair.

Conflicting spec values (the same attribute given twice with different values) are *excluded* from
the fact base and reported, so neither version may appear in the copy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from pma.schemas import ProductInput

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")

_UNIT_ALIASES = {
    "inch": "inch",
    "inches": "inch",
    '"': "inch",
    "″": "inch",
    "hour": "h",
    "hours": "h",
    "hr": "h",
    "hrs": "h",
    "h": "h",
    "uur": "h",
    "uren": "h",
}
_UNITS = [
    "gbps", "mbps", "ghz", "mhz", "tb", "gb", "mb", "kb", "hz", "wh", "mah", "w", "mm", "cm", "kg",
    "mp", "nits", "fps", "db", "inches", "inch", "hours", "hour", "hrs", "hr", "uren", "uur", "h",
    "%", '"', "″",
]  # fmt: skip
_UNIT_RE = re.compile(
    r"(\d+(?:[.,]\d+)*)[\s\-]?(" + "|".join(re.escape(u) for u in _UNITS) + r")(?![a-z0-9])",
    re.IGNORECASE,
)

# attribute synonyms (EN + NL) used to detect conflicting duplicates under different key names
_KEY_GROUPS = {
    "ram": {"ram", "memory", "geheugen", "werkgeheugen", "internalmemory"},
    "storage": {"storage", "ssd", "opslag", "harddisk", "hdd", "opslagcapaciteit", "diskcapacity"},
    "screen": {
        "screen",
        "display",
        "screensize",
        "schermgrootte",
        "scherm",
        "diagonal",
        "diagonaal",
    },
    "weight": {"weight", "gewicht"},
    "warranty": {"warranty", "garantie", "garantieperiode"},
    "battery": {"battery", "batterylife", "accu", "batterij", "accuduur", "batterijduur"},
    "price": {"price", "prijs"},
    "cpu": {"cpu", "processor", "processortype"},
    "power": {"power", "wattage", "vermogen", "powerconsumption"},
    "ports": {"ports", "poorten", "aansluitingen"},
}


def normalize_number(token: str) -> str:
    """Canonical decimal string for a number written in EN or NL notation."""
    t = token.strip()
    if "." in t and "," in t:
        dec = "." if t.rfind(".") > t.rfind(",") else ","
        thou = "," if dec == "." else "."
        t = t.replace(thou, "").replace(dec, ".")
    elif "." in t or "," in t:
        sep = "." if "." in t else ","
        parts = t.split(sep)
        if len(parts) > 2:
            t = "".join(parts)  # 1.234.567
        else:
            head, tail = parts
            if len(tail) == 3 and 1 <= len(head) <= 3 and not head.startswith("0"):
                t = head + tail  # 1.299 / 1,299 -> thousands separator
            else:
                t = head + "." + tail
    try:
        return format(Decimal(t).normalize(), "f")
    except InvalidOperation:
        return token


def extract_numbers(text: str) -> set[str]:
    return {normalize_number(m.group(0)) for m in NUMBER_RE.finditer(text)}


def extract_unit_pairs(text: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for m in _UNIT_RE.finditer(text):
        unit = m.group(2).lower()
        pairs.add((normalize_number(m.group(1)), _UNIT_ALIASES.get(unit, unit)))
    return pairs


def _norm_key(key: str) -> str:
    base = re.sub(r"[^a-z0-9]", "", key.lower())
    for canonical, names in _KEY_GROUPS.items():
        if base in names:
            return canonical
    return base


def _norm_value(value: str) -> str:
    """Comparison form of a spec value: numbers canonicalised, case and spacing removed."""
    lowered = value.lower()
    lowered = NUMBER_RE.sub(lambda m: normalize_number(m.group(0)), lowered)
    return re.sub(r"[\s\-]", "", lowered)


def format_price(price: float | None, currency: str) -> str | None:
    if price is None:
        return None
    return f"{Decimal(str(price)):.2f} {currency}"


@dataclass
class FactBase:
    product: ProductInput
    facts: dict[str, str] = field(default_factory=dict)  # normalised key -> value text
    display_keys: dict[str, str] = field(default_factory=dict)  # normalised key -> original key
    conflicts: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    allowed_numbers: set[str] = field(default_factory=set)
    allowed_pairs: set[tuple[str, str]] = field(default_factory=set)

    @classmethod
    def build(cls, product: ProductInput) -> FactBase:
        fb = cls(product=product)

        # --- spec grouping and conflict detection
        grouped: dict[str, list[tuple[str, str]]] = {}
        for key, value in product.specs.items():
            grouped.setdefault(_norm_key(key), []).append((key, value))
        for nkey, entries in grouped.items():
            distinct = {_norm_value(v) for _, v in entries}
            if len(distinct) > 1:
                fb.conflicts[nkey] = [f"{k}: {v}" for k, v in entries]
                fb.warnings.append(
                    f"conflicting values for '{nkey}' ({' vs '.join(v for _, v in entries)}); "
                    "excluded from the copy"
                )
            else:
                key, value = entries[0]
                fb.facts[nkey] = value
                fb.display_keys[nkey] = key

        # --- core fields
        fb.facts["name"] = product.name
        fb.facts["category"] = product.category
        if product.audience:
            fb.facts["audience"] = product.audience
        if product.notes:
            fb.facts["notes"] = product.notes
        price_text = format_price(product.price, product.currency)
        if price_text:
            fb.facts["price"] = price_text
        else:
            fb.warnings.append("no price provided; the copy must not mention a price")
        if not product.specs:
            fb.warnings.append("no specs provided; keep the copy generic and add no numbers")
        if not product.audience:
            fb.warnings.append("no audience provided; use a general B2B IT-buyer tone")

        # --- allowed numbers / unit pairs from everything that is a usable fact
        haystack = " \n ".join(fb.facts.values())
        fb.allowed_numbers = extract_numbers(haystack)
        fb.allowed_pairs = extract_unit_pairs(haystack)
        if product.price is not None:
            fb.allowed_numbers.add(normalize_number(str(Decimal(str(product.price)).normalize())))
        return fb

    # -- tool + prompt views ----------------------------------------------------------------
    def lookup(self, key: str) -> dict[str, object]:
        """Fuzzy lookup used by the ``lookup_product_fact`` tool."""
        wanted = key.strip()
        if wanted in ("*", ""):
            return {
                "found": True,
                "keys": sorted(self.facts),
                "conflicting_keys": sorted(self.conflicts),
            }
        nkey = _norm_key(wanted)
        if nkey in self.conflicts:
            return {
                "found": False,
                "key": nkey,
                "status": "conflicting",
                "values": self.conflicts[nkey],
                "instruction": "do not state this attribute in the copy",
            }
        if nkey in self.facts:
            return {"found": True, "key": nkey, "value": self.facts[nkey]}
        for candidate in self.facts:  # loose containment fallback ("screen size" -> "screen")
            if nkey and (nkey in candidate or candidate in nkey):
                return {"found": True, "key": candidate, "value": self.facts[candidate]}
        return {"found": False, "key": nkey, "status": "missing"}

    def prompt_view(self) -> dict[str, object]:
        """The JSON object shown to the model (already sanitised)."""
        p = self.product
        return {
            "id": p.id,
            "name": p.name,
            "category": p.category,
            "audience": p.audience,
            "price": format_price(p.price, p.currency),
            "specs": {
                self.display_keys[k]: v for k, v in self.facts.items() if k in self.display_keys
            },
            "notes": p.notes,
            "data_warnings": self.warnings,
        }
