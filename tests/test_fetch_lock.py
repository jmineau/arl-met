"""
Concurrent ``Archive.fetch`` calls download a file once (no network access).

The archive is real; only ``fsspec.open`` is faked, by a source that sleeps
and counts how often it is opened.
"""

import io
import os
import signal
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

DELAY = 0.6  # seconds one fake download takes
START = "2024-07-18 00:00"  # the 00-05 block
OTHER = "2024-07-18 06:00"  # the 06-11 block


class _SlowFile(io.BytesIO):
    """A source whose first read takes ``DELAY`` seconds."""

    def __init__(self, delay: float):
        super().__init__(b"arl-bytes")
        self.delay = delay

    def read(self, size: int | None = -1) -> bytes:
        time.sleep(self.delay)
        self.delay = 0
        return super().read(size)


@pytest.fixture
def fast_locks(monkeypatch):
    monkeypatch.setattr(archives, "_FETCH_LOCK_POLL", 0.02, raising=False)
    monkeypatch.setattr(archives, "_FETCH_LOCK_BEAT", 0.05, raising=False)


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


def test_threads_fetching_one_file_download_it_once(tmp_path, opened, fast_locks):
    results: list[list[Path]] = []

    def work():
        results.append(HRRRArchive().fetch(START, START, dest_dir=tmp_path))

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(opened) == 1
    assert len(results) == 6
    assert all(r == [tmp_path / "20240718_00-05_hrrr"] for r in results)
    assert _visible(tmp_path) == ["20240718_00-05_hrrr"]


def test_threads_fetching_a_crop_download_it_once(
    tmp_path, opened, fast_locks, monkeypatch
):
    crops = []

    def fake_crop(self, url, dest, opts, *, bbox, levels):
        crops.append(dest)
        self._download(url, dest, opts)

    monkeypatch.setattr(HRRRArchive, "_fetch_and_crop", fake_crop)
    box = (-112.0, 40.0, -111.0, 41.0)
    threads = [
        threading.Thread(
            target=lambda: HRRRArchive().fetch(
                START, START, dest_dir=tmp_path, bbox=box
            )
        )
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(crops) == 1
    assert len(_visible(tmp_path)) == 1


def test_different_files_download_in_parallel(tmp_path, monkeypatch, fast_locks):
    # Both downloads must be under way at once, else the barrier breaks.
    barrier = threading.Barrier(2, timeout=5)

    def fake_open(url, mode, **opts):
        barrier.wait()
        return io.BytesIO(b"arl-bytes")

    monkeypatch.setitem(sys.modules, "fsspec", types.SimpleNamespace(open=fake_open))
    errors: list[BaseException] = []

    def work(when):
        try:
            HRRRArchive().fetch(when, when, dest_dir=tmp_path)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(w,)) for w in (START, OTHER)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert _visible(tmp_path) == ["20240718_00-05_hrrr", "20240718_06-11_hrrr"]


def test_interrupted_download_leaves_no_lock_or_partial(tmp_path, monkeypatch):
    def fake_open(url, mode, **opts):
        raise KeyboardInterrupt

    monkeypatch.setitem(sys.modules, "fsspec", types.SimpleNamespace(open=fake_open))

    with pytest.raises(KeyboardInterrupt):
        HRRRArchive().fetch(START, START, dest_dir=tmp_path)

    assert _visible(tmp_path) == []


def test_interrupted_download_leaves_no_lock_during_a_crop(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "fsspec",
        types.SimpleNamespace(open=lambda url, mode, **opts: io.BytesIO(b"x")),
    )

    def fake_crop(self, url, dest, opts, *, bbox, levels):
        self._download(url, archives._partial_path(dest), opts)
        raise KeyboardInterrupt

    monkeypatch.setattr(HRRRArchive, "_fetch_and_crop", fake_crop)

    with pytest.raises(KeyboardInterrupt):
        HRRRArchive().fetch(
            START, START, dest_dir=tmp_path, bbox=(-112.0, 40.0, -111.0, 41.0)
        )

    # The lock goes; the partial file is the crop code's to remove.
    assert not [n for n in _visible(tmp_path) if n.endswith(".lock")]


def _old_lock(tmp_path: Path, age: float) -> Path:
    lock = tmp_path / ".20240718_00-05_hrrr.lock"
    lock.write_text("host gone, process 1\n")
    old = time.time() - age
    os.utime(lock, (old, old))
    return lock


def test_a_stale_lock_is_taken_over(tmp_path, opened, fast_locks, monkeypatch):
    monkeypatch.setattr(archives, "_FETCH_LOCK_STALE", 30.0)
    _old_lock(tmp_path, age=3600)

    assert HRRRArchive().fetch(START, START, dest_dir=tmp_path) == [
        tmp_path / "20240718_00-05_hrrr"
    ]
    assert len(opened) == 1
    assert _visible(tmp_path) == ["20240718_00-05_hrrr"]


def test_a_fresh_lock_is_waited_for_then_timeout_downloads_anyway(
    tmp_path, opened, fast_locks, monkeypatch
):
    monkeypatch.setattr(archives, "_FETCH_LOCK_STALE", 3600.0)
    monkeypatch.setattr(archives, "_FETCH_LOCK_WAIT", 0.3)
    lock = _old_lock(tmp_path, age=0)

    start = time.monotonic()
    HRRRArchive().fetch(START, START, dest_dir=tmp_path)

    assert time.monotonic() - start >= 0.3
    assert len(opened) == 1
    # The other caller's lock is not ours to remove.
    assert lock.exists()
    assert (tmp_path / "20240718_00-05_hrrr").read_bytes() == b"arl-bytes"


def test_a_waiter_uses_the_file_when_the_holder_finishes(
    tmp_path, opened, fast_locks, monkeypatch
):
    monkeypatch.setattr(archives, "_FETCH_LOCK_STALE", 3600.0)
    lock = _old_lock(tmp_path, age=0)
    dest = tmp_path / "20240718_00-05_hrrr"

    def holder():
        time.sleep(0.2)
        dest.write_bytes(b"from the holder")
        lock.unlink()

    t = threading.Thread(target=holder)
    t.start()
    assert HRRRArchive().fetch(START, START, dest_dir=tmp_path) == [dest]
    t.join()

    assert opened == []
    assert dest.read_bytes() == b"from the holder"


def test_a_waiter_downloads_when_the_holder_failed(
    tmp_path, opened, fast_locks, monkeypatch
):
    monkeypatch.setattr(archives, "_FETCH_LOCK_STALE", 3600.0)
    lock = _old_lock(tmp_path, age=0)
    t = threading.Timer(0.2, lock.unlink)
    t.start()

    HRRRArchive().fetch(START, START, dest_dir=tmp_path)
    t.join()

    assert len(opened) == 1
    assert _visible(tmp_path) == ["20240718_00-05_hrrr"]


def test_the_lock_is_kept_fresh_while_downloading(tmp_path, monkeypatch, fast_locks):
    ages = []

    def fake_open(url, mode, **opts):
        lock = tmp_path / ".20240718_00-05_hrrr.lock"
        time.sleep(0.4)
        ages.append(time.time() - lock.stat().st_mtime)
        return io.BytesIO(b"x")

    monkeypatch.setitem(sys.modules, "fsspec", types.SimpleNamespace(open=fake_open))
    HRRRArchive().fetch(START, START, dest_dir=tmp_path)

    assert ages[0] < 0.2


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


def _spawn(tmp_path: Path, when: str, delay: float) -> subprocess.Popen:
    calls = tmp_path / "calls"
    calls.mkdir(exist_ok=True)
    met = tmp_path / "met"
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            WORKER,
            str(met),
            str(calls),
            str(tmp_path / "go"),
            str(delay),
            when,
        ],
        stderr=subprocess.PIPE,
    )


def test_processes_fetching_one_file_download_it_once(tmp_path):
    procs = [_spawn(tmp_path, START, 1.5) for i in range(4)]
    time.sleep(0.5)
    (tmp_path / "go").touch()
    for p in procs:
        _, err = p.communicate(timeout=120)
        assert p.returncode == 0, err.decode()

    assert len(list((tmp_path / "calls").iterdir())) == 1
    assert _visible(tmp_path / "met") == ["20240718_00-05_hrrr"]


def test_a_killed_downloader_does_not_block_others(tmp_path, monkeypatch):
    monkeypatch.setattr(archives, "_FETCH_LOCK_POLL", 0.05)
    monkeypatch.setattr(archives, "_FETCH_LOCK_BEAT", 0.1)
    monkeypatch.setattr(archives, "_FETCH_LOCK_STALE", 1.0)
    monkeypatch.setitem(
        sys.modules,
        "fsspec",
        types.SimpleNamespace(open=lambda url, mode, **opts: io.BytesIO(b"mine")),
    )
    victim = _spawn(tmp_path, START, 600)
    (tmp_path / "go").touch()
    lock = tmp_path / "met" / ".20240718_00-05_hrrr.lock"
    deadline = time.monotonic() + 120
    while not lock.exists():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    victim.send_signal(signal.SIGKILL)  # no chance to clean up
    victim.wait()
    victim.stderr.close()
    assert lock.exists()

    start = time.monotonic()
    HRRRArchive().fetch(START, START, dest_dir=tmp_path / "met")

    assert time.monotonic() - start < 30
    assert (tmp_path / "met" / "20240718_00-05_hrrr").read_bytes() == b"mine"
    assert not lock.exists()
