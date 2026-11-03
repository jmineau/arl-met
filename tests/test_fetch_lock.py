"""
Concurrent ``Archive.fetch`` calls download a file once (no network access).

The archive is real; only ``fsspec.open`` is faked, by a source that sleeps
and counts how often it is opened.
"""

import io
import os
import subprocess
import sys
import textwrap
import threading
import time
import types
from pathlib import Path

import pytest

from arlmet import archives
from arlmet.archives import HRRRArchive

DELAY = 0.5  # seconds one fake download takes
START = "2024-07-18 00:00"  # the 00-05 block
OTHER = "2024-07-18 06:00"  # the 06-11 block
NAME = "20240718_00-05_hrrr"
HEX = "abcdef012345"


class _SlowFile(io.BytesIO):
    """A source whose first read takes ``delay`` seconds."""

    def __init__(self, delay: float):
        super().__init__(b"arl-bytes")
        self.delay = delay

    def read(self, size: int | None = -1) -> bytes:
        time.sleep(self.delay)
        self.delay = 0
        return super().read(size)


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch):
    monkeypatch.setattr(archives, "_FETCH_POLL", 0.02)


@pytest.fixture
def opened(monkeypatch):
    """Fake ``fsspec`` in this process; the list holds one URL per download."""
    urls: list[str] = []
    guard = threading.Lock()

    def fake_open(url, mode, **opts):
        with guard:
            urls.append(url)
        return _SlowFile(DELAY)

    monkeypatch.setitem(sys.modules, "fsspec", types.SimpleNamespace(open=fake_open))
    return urls


def _visible(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def _in_threads(*works):
    threads = [threading.Thread(target=work) for work in works]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def _left_by_a_killed_caller(folder: Path) -> None:
    """Make what ``kill -9`` leaves: the lock folder and a temp file."""
    (folder / f".{NAME}.lock").mkdir(parents=True)
    (folder / f".{NAME}.{HEX}.partial").write_bytes(b"half")


def test_threads_fetching_one_file_download_it_once(tmp_path, opened):
    results: list[list[Path]] = []

    def work():
        results.append(HRRRArchive().fetch(START, START, dest_dir=tmp_path))

    _in_threads(*[work] * 6)

    assert len(opened) == 1
    assert results == [[tmp_path / NAME]] * 6
    assert _visible(tmp_path) == [NAME]


def test_threads_fetching_a_crop_download_and_crop_it_once(
    tmp_path, opened, monkeypatch
):
    crops = []

    def fake_extract_subset(path, dest, bbox, levels):
        crops.append(Path(dest))
        Path(dest).write_bytes(Path(path).read_bytes() + b"-cropped")

    monkeypatch.setitem(
        sys.modules,
        "arlmet.ops.subset",
        types.SimpleNamespace(extract_subset=fake_extract_subset),
    )
    box = (-112.0, 40.0, -111.0, 41.0)

    def work():
        HRRRArchive().fetch(START, START, dest_dir=tmp_path, bbox=box)

    _in_threads(*[work] * 4)

    assert len(opened) == len(crops) == 1
    (only,) = _visible(tmp_path)
    assert only.startswith(f"{NAME}.crop_")
    assert (tmp_path / only).read_bytes() == b"arl-bytes-cropped"


def test_different_files_download_at_the_same_time(tmp_path, monkeypatch):
    # Each source holds its read until the other is reading too.
    both = threading.Barrier(2, timeout=30)

    class Together(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            if self.tell() == 0:
                both.wait()
            return super().read(size)

    monkeypatch.setitem(
        sys.modules,
        "fsspec",
        types.SimpleNamespace(open=lambda url, mode, **opts: Together(b"arl-bytes")),
    )

    _in_threads(
        lambda: HRRRArchive().fetch(START, START, dest_dir=tmp_path),
        lambda: HRRRArchive().fetch(OTHER, OTHER, dest_dir=tmp_path),
    )

    assert _visible(tmp_path) == [NAME, "20240718_06-11_hrrr"]


def test_a_finished_file_needs_no_lock(tmp_path, opened):
    (tmp_path / NAME).write_bytes(b"there")
    _left_by_a_killed_caller(tmp_path)

    start = time.monotonic()
    HRRRArchive().fetch(START, START, dest_dir=tmp_path)

    assert time.monotonic() - start < 5
    assert opened == []


def test_a_file_finished_just_before_the_lock_is_not_downloaded_again(
    tmp_path, opened, monkeypatch
):
    make_lock = archives._make_lock

    def finished_meanwhile(lock):
        (tmp_path / NAME).write_bytes(b"holder's")
        return make_lock(lock)

    monkeypatch.setattr(archives, "_make_lock", finished_meanwhile)

    HRRRArchive().fetch(START, START, dest_dir=tmp_path)

    assert opened == []
    assert (tmp_path / NAME).read_bytes() == b"holder's"
    assert _visible(tmp_path) == [NAME]


def test_a_failed_download_leaves_nothing_and_the_next_caller_starts_at_once(
    tmp_path, monkeypatch
):
    def fail(url, mode, **opts):
        raise RuntimeError("no network")

    monkeypatch.setitem(sys.modules, "fsspec", types.SimpleNamespace(open=fail))

    start = time.monotonic()
    for _ in range(20):
        with pytest.raises(RuntimeError, match="no network"):
            HRRRArchive().fetch(START, START, dest_dir=tmp_path)

    assert time.monotonic() - start < 10
    assert _visible(tmp_path) == []


def test_a_waiter_downloads_when_the_holder_failed(tmp_path, monkeypatch):
    calls = []

    def fake_open(url, mode, **opts):
        calls.append(url)
        if len(calls) == 1:
            time.sleep(DELAY)
            raise RuntimeError("no network")
        return io.BytesIO(b"arl-bytes")

    monkeypatch.setitem(sys.modules, "fsspec", types.SimpleNamespace(open=fake_open))
    outcomes = []

    def work():
        try:
            HRRRArchive().fetch(START, START, dest_dir=tmp_path)
            outcomes.append("ok")
        except RuntimeError:
            outcomes.append("failed")

    _in_threads(work, work)

    assert sorted(outcomes) == ["failed", "ok"]
    assert len(calls) == 2
    assert _visible(tmp_path) == [NAME]


def test_a_lock_with_nothing_growing_is_taken_over(
    tmp_path, opened, monkeypatch, caplog
):
    monkeypatch.setattr(archives, "_FETCH_STALL", 0.2)
    _left_by_a_killed_caller(tmp_path)

    with caplog.at_level("WARNING", logger=archives.logger.name):
        HRRRArchive().fetch(START, START, dest_dir=tmp_path)

    assert len(opened) == 1
    assert (tmp_path / NAME).read_bytes() == b"arl-bytes"
    assert _visible(tmp_path) == [NAME]  # the lock and the dead temp file are gone
    assert "taking it over" in caplog.text


def test_a_lock_is_not_taken_over_while_the_download_grows(
    tmp_path, opened, monkeypatch
):
    monkeypatch.setattr(archives, "_FETCH_STALL", 0.3)
    _left_by_a_killed_caller(tmp_path)
    temp = tmp_path / f".{NAME}.{HEX}.partial"

    def holder():
        for _ in range(30):  # 1.5 s, five times the stall limit
            with open(temp, "ab") as f:
                f.write(b"more")
            time.sleep(0.05)
        (tmp_path / NAME).write_bytes(b"holder's")
        temp.unlink()
        (tmp_path / f".{NAME}.lock").rmdir()

    def waiter():
        HRRRArchive().fetch(START, START, dest_dir=tmp_path)

    _in_threads(holder, waiter)

    assert opened == []
    assert (tmp_path / NAME).read_bytes() == b"holder's"


def test_a_crop_step_counts_as_growth(tmp_path):
    """The waiter sums every temp file of its file and no other file's."""
    dest = tmp_path / NAME
    (tmp_path / f".{NAME}.{HEX}.partial").write_bytes(b"12")
    (tmp_path / f"..{NAME}.{HEX}.partial.{HEX}.partial").write_bytes(b"123")
    (tmp_path / f"...{NAME}.{HEX}.partial.{HEX}.partial.{HEX}.partial").touch()
    (tmp_path / f".{NAME}.crop_1_2_3_4.{HEX}.partial").write_bytes(b"other crop")
    (tmp_path / f".{NAME}.lock").mkdir()

    assert sorted(archives._temp_sizes(dest).values()) == [0, 2, 3]


def test_a_finished_file_is_not_replaced(tmp_path, monkeypatch):
    dest = tmp_path / NAME

    class FirstFinishes(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            if self.tell() == 0:
                dest.write_bytes(b"first copy")  # a caller paused past the limit
            return super().read(size)

    monkeypatch.setitem(
        sys.modules,
        "fsspec",
        types.SimpleNamespace(open=lambda url, mode, **opts: FirstFinishes(b"second")),
    )

    HRRRArchive().fetch(START, START, dest_dir=tmp_path)
    assert dest.read_bytes() == b"first copy"
    assert _visible(tmp_path) == [NAME]

    HRRRArchive().fetch(START, START, dest_dir=tmp_path, overwrite=True)
    assert dest.read_bytes() == b"second"
    assert _visible(tmp_path) == [NAME]


@pytest.mark.parametrize("finished", [True, False])
def test_a_caller_whose_temp_file_was_removed_uses_the_finished_file(
    tmp_path, monkeypatch, finished
):
    dest = tmp_path / NAME

    class TakenForDead(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            if self.tell() == 0:
                for temp in tmp_path.glob("*.partial"):
                    temp.unlink()
                if finished:
                    dest.write_bytes(b"taker's")
            return super().read(size)

    monkeypatch.setitem(
        sys.modules,
        "fsspec",
        types.SimpleNamespace(open=lambda url, mode, **opts: TakenForDead(b"mine")),
    )

    if finished:
        HRRRArchive().fetch(START, START, dest_dir=tmp_path)
        assert dest.read_bytes() == b"taker's"
    else:
        with pytest.raises(FileNotFoundError):
            HRRRArchive().fetch(START, START, dest_dir=tmp_path)
    assert not (tmp_path / f".{NAME}.lock").exists()


def test_overwrite_callers_download_one_at_a_time(tmp_path, monkeypatch):
    (tmp_path / NAME).write_bytes(b"old")
    at_once = []
    running = []
    guard = threading.Lock()

    class Counted(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            if self.tell() == 0:
                with guard:
                    running.append(1)
                    at_once.append(len(running))
                time.sleep(0.1)
                with guard:
                    running.pop()
            return super().read(size)

    monkeypatch.setitem(
        sys.modules,
        "fsspec",
        types.SimpleNamespace(open=lambda url, mode, **opts: Counted(b"new")),
    )

    def work():
        HRRRArchive().fetch(START, START, dest_dir=tmp_path, overwrite=True)

    _in_threads(*[work] * 3)

    assert at_once == [1, 1, 1]
    assert (tmp_path / NAME).read_bytes() == b"new"
    assert _visible(tmp_path) == [NAME]


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs a folder this user cannot write",
)
def test_an_unwritable_folder_raises_promptly(tmp_path, opened):
    folder = tmp_path / "met"
    folder.mkdir()
    folder.chmod(0o500)
    try:
        start = time.monotonic()
        with pytest.raises(PermissionError):
            HRRRArchive().fetch(START, START, dest_dir=folder)
        assert time.monotonic() - start < 10
    finally:
        folder.chmod(0o700)
    assert opened == []


# -- separate processes -----------------------------------------------------

WORKER = textwrap.dedent(
    """
    import io, sys, time, types, uuid
    from pathlib import Path

    dest_dir, calls, go, delay, when = sys.argv[1:6]

    class Slow(io.BytesIO):
        def read(self, size=-1):
            time.sleep(float(delay))
            return super().read(size)

    def fake_open(url, mode, **opts):
        (Path(calls) / uuid.uuid4().hex).touch()
        return Slow(b"arl-bytes")

    sys.modules["fsspec"] = types.SimpleNamespace(open=fake_open)
    from arlmet.archives import HRRRArchive

    while not Path(go).exists():  # start together, whatever the import time
        time.sleep(0.01)
    HRRRArchive().fetch(when, when, dest_dir=dest_dir)
    """
)


def _spawn(tmp_path: Path, delay: float) -> subprocess.Popen:
    calls = tmp_path / "calls"
    calls.mkdir(exist_ok=True)
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            WORKER,
            str(tmp_path / "met"),
            str(calls),
            str(tmp_path / "go"),
            str(delay),
            START,
        ],
        stderr=subprocess.PIPE,
    )


def test_processes_fetching_one_file_download_it_once(tmp_path):
    procs = [_spawn(tmp_path, 1.5) for _ in range(4)]
    time.sleep(0.5)
    (tmp_path / "go").touch()
    for p in procs:
        _, err = p.communicate(timeout=120)
        assert p.returncode == 0, err.decode()

    assert len(list((tmp_path / "calls").iterdir())) == 1
    assert _visible(tmp_path / "met") == [NAME]


def test_a_killed_downloader_does_not_block_others(tmp_path, monkeypatch):
    monkeypatch.setattr(archives, "_FETCH_STALL", 1.0)
    monkeypatch.setitem(
        sys.modules,
        "fsspec",
        types.SimpleNamespace(open=lambda url, mode, **opts: io.BytesIO(b"mine")),
    )
    met = tmp_path / "met"
    victim = _spawn(tmp_path, 600)
    (tmp_path / "go").touch()
    deadline = time.monotonic() + 120
    while not list(tmp_path.glob("met/.*.partial")):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    victim.kill()  # SIGKILL: no chance to clean up
    victim.wait()
    victim.stderr.close()
    assert (met / f".{NAME}.lock").exists()

    start = time.monotonic()
    HRRRArchive().fetch(START, START, dest_dir=met)

    assert time.monotonic() - start < 30
    assert (met / NAME).read_bytes() == b"mine"
    assert _visible(met) == [NAME]
