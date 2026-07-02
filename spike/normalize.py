"""Spike 4: scientific-notation normalizer with offset mapping.

Only rewrites what misaki's G2P gets wrong (verified 2026-07-02):
scientific notation, powers of ten, comparison operators, unit slashes,
Greek letters, and math/unicode symbols. Plain numbers, decimals,
percents, and comma-grouped integers are left alone -- misaki handles
those natively via num2words.

normalize(text) -> (spoken, to_orig_start, to_orig_end)

The two arrays map every char position of `spoken` back to a char range
of `text`, so word-level speech marks computed on the spoken string can
highlight the original expression (e.g. all five words of "one point
five times ten to the negative ninth" map to the on-screen "1.5e-9").
"""

import re

from num2words import num2words


def _ordinal(n: int) -> str:
    words = num2words(abs(n), to="ordinal")
    return f"negative {words}" if n < 0 else words


SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")

GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho",
    "σ": "sigma", "τ": "tau", "υ": "upsilon", "φ": "phi", "χ": "chi",
    "ψ": "psi", "ω": "omega",
    "Γ": "capital gamma", "Δ": "delta", "Θ": "capital theta",
    "Λ": "capital lambda", "Ξ": "capital xi", "Π": "capital pi",
    "Σ": "sigma", "Φ": "capital phi", "Ψ": "capital psi", "Ω": "omega",
}

SYMBOLS = {
    "≤": "less than or equal to", "≥": "greater than or equal to",
    "<": "less than", ">": "greater than", "≠": "not equal to",
    "≈": "approximately", "±": "plus or minus", "→": "goes to",
    "∝": "proportional to", "∞": "infinity", "∇": "del", "∂": "partial",
    "√": "square root of", "∑": "the sum of", "∏": "the product of",
    "∫": "the integral of", "·": "times", "×": "times", "∈": "in",
}

UNITS = {
    "m": "meter", "km": "kilometer", "cm": "centimeter", "mm": "millimeter",
    "µm": "micrometer", "um": "micrometer", "nm": "nanometer",
    "s": "second", "ms": "millisecond", "µs": "microsecond",
    "us": "microsecond", "ns": "nanosecond", "min": "minute", "h": "hour",
    "g": "gram", "kg": "kilogram", "mg": "milligram", "µg": "microgram",
    "Hz": "hertz", "kHz": "kilohertz", "MHz": "megahertz", "GHz": "gigahertz",
    "V": "volt", "mV": "millivolt", "A": "amp", "mA": "milliamp",
    "W": "watt", "mW": "milliwatt", "eV": "electron volt", "K": "kelvin",
    "Pa": "pascal", "kPa": "kilopascal", "dB": "decibel", "mol": "mole",
    "L": "liter", "mL": "milliliter",
}

NO_PLURAL = {"hertz", "kilohertz", "megahertz", "gigahertz", "kelvin",
             "decibel"}  # decibel does pluralize, but "3 decibel" is fine spoken


def _plural(unit_word: str, value: str) -> str:
    if unit_word in NO_PLURAL:
        return unit_word
    try:
        singular = float(value) == 1.0
    except ValueError:
        singular = False
    return unit_word if singular else unit_word + "s"


def _sci_notation(m: re.Match) -> str:
    return f"{m.group(1)} times ten to the {_ordinal(int(m.group(2)))}"


def _pow_ten(m: re.Match) -> str:
    mant, exp = m.group(1), m.group(2).translate(SUPERSCRIPTS)
    lead = f"{mant} times " if mant else ""
    return f"{lead}ten to the {_ordinal(int(exp))}"


def _power(m: re.Match) -> str:
    base, exp = m.group(1), int(m.group(2))
    if exp == 2:
        return f"{base} squared"
    if exp == 3:
        return f"{base} cubed"
    return f"{base} to the {_ordinal(exp)}"


def _unit_slash(m: re.Match) -> str:
    num = (m.group(1) or "").strip()
    u1, u2 = UNITS[m.group(2)], UNITS[m.group(3)]
    if not num:  # bare "m/s" -> plural reading
        return f"{_plural(u1, '')} per {u2}"
    return f"{num} {_plural(u1, num)} per {u2}"


def _num_unit(m: re.Match) -> str:
    num, unit = m.group(1), UNITS[m.group(2)]
    return f"{num} {_plural(unit, num)}"


_UNIT_ALT = "|".join(sorted(UNITS, key=len, reverse=True))

RULES = [
    # 1.5e-9 / 2E+8  (exponent needs a sign or 1-2 digits; keeps "2018e" out)
    (re.compile(r"\b(\d+(?:\.\d+)?)[eE]([+-]\d+|\d\d?)\b"), _sci_notation),
    # 3 × 10^8, 3 x 10**8, 10⁻⁹, bare 10^8
    (re.compile(r"(?:\b(\d+(?:\.\d+)?)\s*[x×·*]\s*)?\b10\s*(?:\^|\*\*)\s*([+-]?\d+)\b"), _pow_ten),
    (re.compile(r"(?:\b(\d+(?:\.\d+)?)\s*[x×·*]\s*)?\b10([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)"), _pow_ten),
    # x^2, mc^2, r^3
    (re.compile(r"\b([A-Za-z]+)\s*\^\s*(\d+)\b"), _power),
    # 3 m/s, 9.8 m/s
    (re.compile(rf"\b(\d+(?:\.\d+)?)\s*({_UNIT_ALT})/({_UNIT_ALT})\b"), _unit_slash),
    # bare m/s (e.g. after a power-of-ten expression that consumed the number)
    (re.compile(rf"\b()({_UNIT_ALT})/({_UNIT_ALT})\b"), _unit_slash),
    # 2.5 µm, 5 GHz (unit must be followed by non-letter)
    (re.compile(rf"\b(\d+(?:\.\d+)?)\s*({_UNIT_ALT})(?![A-Za-z])"), _num_unit),
    # degrees
    (re.compile(r"°\s*C\b"), lambda m: " degrees Celsius"),
    (re.compile(r"°\s*F\b"), lambda m: " degrees Fahrenheit"),
    (re.compile(r"°"), lambda m: " degrees "),
    # greek + symbols, padded with spaces so words don't fuse
    (re.compile("|".join(map(re.escape, GREEK))), lambda m: f" {GREEK[m.group()]} "),
    (re.compile("|".join(map(re.escape, SYMBOLS))), lambda m: f" {SYMBOLS[m.group()]} "),
]


def normalize(text: str):
    """Return (spoken, to_orig_start, to_orig_end)."""
    # collect matches from all rules; earlier rules win on overlap
    taken = []  # (start, end, replacement)
    occupied = [False] * (len(text) + 1)
    for pattern, fn in RULES:
        for m in pattern.finditer(text):
            if any(occupied[m.start():m.end()]):
                continue
            taken.append((m.start(), m.end(), fn(m)))
            for i in range(m.start(), m.end()):
                occupied[i] = True
    taken.sort()

    spoken_parts = []
    to_start, to_end = [], []
    pos = 0
    for start, end, repl in taken:
        for o in range(pos, start):  # identity segment
            spoken_parts.append(text[o])
            to_start.append(o)
            to_end.append(o + 1)
        for _ in repl:  # replaced segment: whole spoken span -> orig span
            to_start.append(start)
            to_end.append(end)
        spoken_parts.append(repl)
        pos = end
    for o in range(pos, len(text)):
        spoken_parts.append(text[o])
        to_start.append(o)
        to_end.append(o + 1)

    spoken = "".join(spoken_parts)
    # collapse double spaces introduced by padded replacements, keeping maps aligned
    out, ts, te = [], [], []
    for i, ch in enumerate(spoken):
        if ch == " " and out and out[-1] == " ":
            continue
        out.append(ch)
        ts.append(to_start[i])
        te.append(to_end[i])
    return "".join(out), ts, te


if __name__ == "__main__":
    samples = [
        "The rate constant is 1.5e-9 per second.",
        "Light travels at 3 × 10^8 m/s in vacuum.",
        "We set α = 0.05 and found p < 0.001.",
        "The beam width is 2.5 µm at 5 GHz.",
        "Energy scales as E = mc^2, roughly 10⁻⁹ J.",
        "Heated to 37 °C, i.e. Δ T ≈ 12 degrees.",
        "Accuracy improved by 4.6% on 12,345 samples.",  # should be untouched
    ]
    for s in samples:
        spoken, ts, te = normalize(s)
        print(f"  in: {s}\n out: {spoken}\n")
