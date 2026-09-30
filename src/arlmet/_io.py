"""Safe output-path helpers shared by the ARL writers."""

from __future__ import annotations

import contextlib
import os
import secrets
from collections.abc import Iterator
from pathlib import Path


def reject_same_file(
    path: str | os.PathLike[str], dest: str | os.PathLike[str]
) -> None:
    """
    Raise if the output ``dest`` is the same file as the input ``path``.

    Writing over the file being read truncates it before its records are read,
    destroying it. Symlinks and hard links to the input are caught too.
    """
    path, dest = Path(path), Path(dest)
    same = path.resolve() == dest.resolve()
    if not same and path.exists() and dest.exists():
        same = os.path.samefile(path, dest)
    if same:
        raise ValueError(
            f"dest {dest} is the same file as the input {path}; "
            "write to a new path instead."
        )


@contextlib.contextmanager
def atomic_output(dest: str | os.PathLike[str]) -> Iterator[Path]:
    """
    Yield a temporary path that replaces ``dest`` only on success.

    The temporary file sits next to ``dest`` (so the final
    ``os.replace`` is an atomic rename on the same filesystem) and is removed
    if the ``with`` block raises. An interrupted write therefore never leaves a
    truncated file under the final name, which a later run could mistake for a
    finished output.
    """
    dest = Path(dest)
    # A random name rather than tempfile.mkstemp: mkstemp creates the file with
    # mode 0600, which would strip group read from outputs in shared
    # directories. The writer creates this file normally, honoring the umask.
    tmp_path = dest.with_name(f".{dest.name}.{secrets.token_hex(6)}.partial")
    try:
        yield tmp_path
        os.replace(tmp_path, dest)
    finally:
        tmp_path.unlink(missing_ok=True)
