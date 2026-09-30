"""Safe output-path helpers shared by the ARL writers."""

from __future__ import annotations

import contextlib
import os
import secrets
from collections.abc import Iterator
from pathlib import Path


def reject_same_file(
    source: str | os.PathLike[str], destination: str | os.PathLike[str]
) -> None:
    """
    Raise if ``destination`` is the same file as ``source``.

    Writing over the file being read truncates it before its records are read,
    destroying it. Symlinks and hard links to the source are caught too.
    """
    source, destination = Path(source), Path(destination)
    same = source.resolve() == destination.resolve()
    if not same and source.exists() and destination.exists():
        same = os.path.samefile(source, destination)
    if same:
        raise ValueError(
            f"destination {destination} is the same file as the source {source}; "
            "write to a new path instead."
        )


@contextlib.contextmanager
def atomic_output(destination: str | os.PathLike[str]) -> Iterator[Path]:
    """
    Yield a temporary path that replaces ``destination`` only on success.

    The temporary file sits next to ``destination`` (so the final
    ``os.replace`` is an atomic rename on the same filesystem) and is removed
    if the ``with`` block raises. An interrupted write therefore never leaves a
    truncated file under the final name, which a later run could mistake for a
    finished output.
    """
    destination = Path(destination)
    # A random name rather than tempfile.mkstemp: mkstemp creates the file with
    # mode 0600, which would strip group read from outputs in shared
    # directories. The writer creates this file normally, honoring the umask.
    tmp_path = destination.with_name(
        f".{destination.name}.{secrets.token_hex(6)}.partial"
    )
    try:
        yield tmp_path
        os.replace(tmp_path, destination)
    finally:
        tmp_path.unlink(missing_ok=True)
