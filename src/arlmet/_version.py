"""The installed arlmet version, kept apart so any module can import it."""

import importlib.metadata

__version__ = importlib.metadata.version("arlmet")
