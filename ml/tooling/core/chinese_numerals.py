"""Read the numbers out of an answer written the way a person writes one.

She answers a maths question in prose, and prose spells numbers several ways in
the same sentence: "六十一点五四公里", "19.41 天", "三分之一", "六十九万九千克".
Comparing an answer to a reference means finding the reference's value in that
prose, so every spelling has to be readable.

The extraction is deliberately generous. A number this misses is a correct
answer marked wrong; a number it invents is at worst a wrong answer marked
right, and the reference still has to appear for that to happen. Two earlier
attempts at this comparison reported 91% and 7.3% error rates that were entirely
the parser's fault -- rounding and Chinese numerals -- so the bias is set on
purpose and the tests pin it.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

DIGITS: dict[str, int] = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "壹": 1,
    "二": 2,
    "两": 2,
    "贰": 2,
    "三": 3,
    "叁": 3,
    "四": 4,
    "肆": 4,
    "五": 5,
    "伍": 5,
    "六": 6,
    "陆": 6,
    "七": 7,
    "柒": 7,
    "八": 8,
    "捌": 8,
    "九": 9,
    "玖": 9,
}
SMALL_UNITS: dict[str, int] = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000}
BIG_UNITS: dict[str, int] = {"万": 10**4, "亿": 10**8}

_NUMERAL_CHARS = "".join(DIGITS) + "".join(SMALL_UNITS) + "".join(BIG_UNITS)
# A run of numeral characters, optionally with a decimal tail after 点.
CJK_RUN = re.compile(f"[{_NUMERAL_CHARS}]+(?:点[{_NUMERAL_CHARS}]+)?")
ARABIC = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
# "X 分之 Y" in either spelling, so a fractional answer is comparable.
CJK_FRACTION = re.compile(
    f"([{_NUMERAL_CHARS}]+)\\s*分之\\s*([{_NUMERAL_CHARS}]+)"
)
ARABIC_FRACTION = re.compile(r"(\d+)\s*/\s*(\d+)")


def _integer(text: str) -> int | None:
    """Parse the integer part of a Chinese numeral, or None if it is not one."""

    total = 0
    section = 0
    digit = 0
    seen = False
    for char in text:
        if char in DIGITS:
            digit = DIGITS[char]
            seen = True
        elif char in SMALL_UNITS:
            # A bare 十 means ten, as in 十五.
            section += (digit or 1) * SMALL_UNITS[char]
            digit = 0
            seen = True
        elif char in BIG_UNITS:
            section += digit
            total += (section or 1) * BIG_UNITS[char]
            section = 0
            digit = 0
            seen = True
        else:
            return None
    if not seen:
        return None
    return total + section + digit


def parse_cjk_number(text: str) -> Decimal | None:
    """Parse one Chinese numeral, decimal tail included.

    After 点 the digits are read one by one -- 六十一点五四 is 61.54, not
    61 and 54 -- which is how the decimal is actually spoken.
    """

    head, dot, tail = text.partition("点")
    whole = _integer(head)
    if whole is None:
        return None
    if not dot:
        return Decimal(whole)
    places = []
    for char in tail:
        if char not in DIGITS:
            break
        places.append(str(DIGITS[char]))
    if not places:
        return Decimal(whole)
    return Decimal(f"{whole}.{''.join(places)}")


def numeric_values(text: str) -> set[Decimal]:
    """Every number this text could be read as stating.

    Prefixes of a Chinese numeral run are included as well. A run can absorb a
    following unit that happens to be a numeral character, and reading one
    character too far should not lose the number that was there.
    """

    text = str(text or "")
    found: set[Decimal] = set()

    for token in ARABIC.findall(text):
        try:
            found.add(Decimal(token.replace(",", "")))
        except (InvalidOperation, ValueError):
            continue

    for numerator, denominator in ARABIC_FRACTION.findall(text):
        try:
            if Decimal(denominator) != 0:
                found.add(Decimal(numerator) / Decimal(denominator))
        except (InvalidOperation, ValueError):
            continue

    for run in CJK_RUN.findall(text):
        value = parse_cjk_number(run)
        if value is not None:
            found.add(value)
        for cut in range(1, len(run)):
            prefix = parse_cjk_number(run[:cut])
            if prefix is not None:
                found.add(prefix)

    for denominator_text, numerator_text in CJK_FRACTION.findall(text):
        denominator = parse_cjk_number(denominator_text)
        numerator = parse_cjk_number(numerator_text)
        if denominator and numerator is not None and denominator != 0:
            found.add(numerator / denominator)

    return found


def agrees(reference: Decimal, candidates: set[Decimal]) -> bool:
    """Does any candidate state the reference, at the precision it was written?

    She rounds: a reference of 19.4118 is answered "19.41 天", and a checker
    that demands the reference's digits calls that wrong. The tolerance is half
    a unit of the candidate's own last place, so 19.41 agrees and 19.5 does not.
    """

    for candidate in candidates:
        if candidate == reference:
            return True
        exponent = candidate.as_tuple().exponent
        places = -exponent if isinstance(exponent, int) and exponent < 0 else 0
        tolerance = Decimal(10) ** Decimal(-places) / 2
        if abs(candidate - reference) <= tolerance:
            return True
    return False


__all__ = ["agrees", "numeric_values", "parse_cjk_number"]
