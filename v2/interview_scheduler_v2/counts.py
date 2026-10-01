"""Lossless validation for counts entered in Excel or the review editor."""

from decimal import Decimal, InvalidOperation


def whole_number(value: object) -> int:
    """Accept whole, finite numeric values; never truncate, round, or accept bools."""

    if isinstance(value, bool):
        raise ValueError("Counts must be whole numbers.")
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise ValueError("Counts must be whole numbers.") from None
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError("Counts must be whole numbers.")
    return int(number)


def nonnegative_integer(value: object) -> int:
    """Accept whole, finite numeric values of zero or more; never truncate or round."""

    number = whole_number(value)
    if number < 0:
        raise ValueError("Counts must be whole numbers of zero or more.")
    return number
