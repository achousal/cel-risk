"""
CLI seed parsing utilities.

Provides functions for parsing and validating seed lists from
CLI arguments.
"""


def parse_seed_list(seed_string: str) -> list[int]:
    """
    Parse comma-separated seed list from CLI argument.

    Parameters
    ----------
    seed_string : str
        Comma-separated seed values (e.g., '0,1,2,3,4')

    Returns
    -------
    List[int]
        List of seed integers

    Raises
    ------
    ValueError
        If seed_string is empty or contains invalid integers

    Examples
    --------
    >>> parse_seed_list('0,1,2')
    [0, 1, 2]

    >>> parse_seed_list('72, 73, 74')
    [72, 73, 74]

    >>> parse_seed_list('')
    ValueError: Empty seed string provided
    """
    if not seed_string or not seed_string.strip():
        raise ValueError("Empty seed string provided")

    try:
        seeds = [int(s.strip()) for s in seed_string.split(",")]
    except ValueError as e:
        raise ValueError(
            f"Invalid seed format in '{seed_string}'. "
            "Expected comma-separated integers (e.g., '0,1,2')."
        ) from e

    if not seeds:
        raise ValueError("No seeds parsed from seed string")

    return seeds


def parse_seed_range(start: int, count: int) -> list[int]:
    """
    Generate list of consecutive seeds from start value.

    Parameters
    ----------
    start : int
        Starting seed value
    count : int
        Number of seeds to generate

    Returns
    -------
    List[int]
        List of consecutive seed integers

    Raises
    ------
    ValueError
        If count is less than 1

    Examples
    --------
    >>> parse_seed_range(0, 3)
    [0, 1, 2]

    >>> parse_seed_range(72, 5)
    [72, 73, 74, 75, 76]

    >>> parse_seed_range(0, 0)
    ValueError: Seed count must be at least 1
    """
    if count < 1:
        raise ValueError(f"Seed count must be at least 1, got {count}")

    return list(range(start, start + count))
