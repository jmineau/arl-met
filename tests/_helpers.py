"""Helpers shared by test modules."""

import functools
import warnings

from arlmet import ARLFormatWarning


def writes_small_uniform_fields(func):
    """
    Let a fixture write uniform fields below the packing precision.

    The tests read those fields back as the packed file holds them, so the
    writer's warning is not what they are about.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ARLFormatWarning)
            return func(*args, **kwargs)

    return wrapper
