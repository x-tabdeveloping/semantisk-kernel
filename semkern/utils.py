from typing import List


def distinct(l: List) -> List:
    """
    Function that takes a list and transforms it into a list of unique values,
    similar to SQL's DISTINCT command
    """
    return list(set(l))
