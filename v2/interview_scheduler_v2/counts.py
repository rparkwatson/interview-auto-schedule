"""Lossless validation for counts entered in Excel or the review editor."""

from decimal import Decimal, InvalidOperation


def nonnegative_integer(value: object) -> int:
    """Accept whole, finite numeric values; never truncate, round, or accept bools."""

    if isinstance(value, bool):
        raise ValueError("Counts must be whole numbers of zero or more.")
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise ValueError("Counts must be whole numbers of zero or more.") from None
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise ValueError("Counts must be whole numbers of zero or more.")
    return int(number)
