"""
Tests for ``Archive.sync`` against a fake NOAA mirror.

The mirror is a local folder read through fsspec's ``file://``; no network access.
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

from arlmet import archives
from arlmet.archives import (
    Archive,
    GDASArchive,
    HRRRArchive,
    HRRRv1Archive,
    NARRArchive,
    ReanalysisArchive,
    SyncResult,
)

DAY = {"start": "2024-07-18 00:00", "end": "2024-07-18 23:00"}
HRRR_DAY = [
    "20240718_00-05_hrrr",
    "20240718_06-11_hrrr",
    "20240718_12-17_hrrr",
    "20240718_18-23_hrrr",
]


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


class FakeMirror:
    """A folder laid out like NOAA's bucket, with a ``listing.md5.txt`` per product."""

    def __init__(self, root: Path, archive: Archive, monkeypatch, start, end) -> None:
        """Serve *archive*'s files for ``[start, end]`` from *root*."""
        self.root = root
        self.product = archive._archive_path(archive.start_date).split("/")[0]
        self.listing: dict[str, str] = {}  # name -> md5 NOAA lists
        self.paths = {Path(p).name: p for p in archive.paths_for_range(start, end)}
        monkeypatch.setattr(
            archive, "_url", lambda path, mirror: f"file://{root}/{path}"
        )
        monkeypatch.setattr(archive, "_storage_options", lambda mirror: {})
        self.archive = archive

    def put(self, name: str, data: bytes, listed_md5: str | None = None) -> None:
        """Publish a file at its archive path; list *listed_md5* (default: its real MD5)."""
        path = self.root / self._path_of(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.listing[name] = listed_md5 or md5(data)
        self._write_listing()

    def _path_of(self, name: str) -> str:
        return self.paths[name]

    def _write_listing(self) -> None:
        text = "".join(f"{m}  {n}\n" for n, m in self.listing.items())
        folder = self.root / self.product
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "listing.md5.txt").write_text(text)


def content(name: str) -> bytes:
    return f"ARL data of {name}\n".encode() * 50


@pytest.fixture
def hrrr(tmp_path, monkeypatch):
    """A HRRR archive reading a fake mirror that holds the four files of 2024-07-18."""
    mirror = FakeMirror(tmp_path / "noaa", HRRRArchive(), monkeypatch, **DAY)
    for name in HRRR_DAY:
        mirror.put(name, content(name))
    return mirror


@pytest.fixture
def dest(tmp_path):
    return tmp_path / "local" / "hrrr"


def no_files(*args, **kwargs):
    raise AssertionError("a file was downloaded or read")


def hidden_files(folder: Path) -> list[Path]:
    return [p for p in folder.rglob(".*")]


# ---------------------------------------------------------------------------
# New files
# ---------------------------------------------------------------------------


def test_missing_files_are_fetched_into_the_archive_layout(hrrr, dest):
    result = hrrr.archive.sync(dest, **DAY)

    expected = [dest / "2024" / "07" / name for name in HRRR_DAY]
    assert result.downloaded == expected
    assert (
        result.adopted == result.unchanged == result.mismatched == result.failed == []
    )
    for name, path in zip(HRRR_DAY, expected, strict=True):
        assert path.read_bytes() == content(name)
    assert hidden_files(dest) == []  # no .partial file, no lock


def test_manifest_is_md5sum_format_and_matches_the_files(hrrr, dest):
    hrrr.archive.sync(dest, **DAY)

    lines = (dest / "arlmet-manifest.md5").read_text().splitlines()
    assert lines == [f"{md5(content(name))}  2024/07/{name}" for name in HRRR_DAY]


def test_a_corrupt_download_is_rejected_and_leaves_nothing(hrrr, dest):
    bad = HRRR_DAY[1]
    hrrr.put(bad, content(bad), listed_md5=md5(b"what NOAA says it should be"))

    result = hrrr.archive.sync(dest, **DAY)

    assert result.failed == [dest / "2024" / "07" / bad]
    assert not (dest / "2024" / "07" / bad).exists()
    assert hidden_files(dest) == []
    assert len(result.downloaded) == 3  # the others went on
    assert bad not in (dest / "arlmet-manifest.md5").read_text()


def test_a_corrupt_download_does_not_replace_the_old_copy(hrrr, dest):
    hrrr.archive.sync(dest, **DAY)
    name = HRRR_DAY[0]
    old = (dest / "2024" / "07" / name).read_bytes()
    # NOAA corrects the file, but what the server sends does not match its list.
    hrrr.put(name, b"new bytes", listed_md5=md5(b"other bytes"))

    result = hrrr.archive.sync(dest, **DAY)

    assert result.failed == [dest / "2024" / "07" / name]
    assert (dest / "2024" / "07" / name).read_bytes() == old
    assert hidden_files(dest) == []


def test_a_file_missing_on_the_mirror_fails_and_the_run_goes_on(hrrr, dest):
    (hrrr.root / hrrr._path_of(HRRR_DAY[2])).unlink()

    result = hrrr.archive.sync(dest, **DAY)

    assert result.failed == [dest / "2024" / "07" / HRRR_DAY[2]]
    assert len(result.downloaded) == 3


def test_a_name_noaa_does_not_list_is_not_fetched(hrrr, dest):
    del hrrr.listing[HRRR_DAY[3]]
    hrrr._write_listing()

    result = hrrr.archive.sync(dest, **DAY)

    assert [p.name for p in result.downloaded] == HRRR_DAY[:3]
    assert not (dest / "2024" / "07" / HRRR_DAY[3]).exists()


def test_start_and_end_limit_the_files(hrrr, dest):
    result = hrrr.archive.sync(dest, start="2024-07-18 07:00", end="2024-07-18 13:00")

    assert [p.name for p in result.downloaded] == HRRR_DAY[1:3]


def test_whole_archive_by_default(hrrr, dest):
    result = hrrr.archive.sync(dest)  # start_date to now: only the listed files count

    assert [p.name for p in result.downloaded] == HRRR_DAY


def test_time_zone_aware_range_is_read_as_utc(hrrr, dest):
    result = hrrr.archive.sync(
        dest, start="2024-07-17 20:00-04:00", end="2024-07-18 02:00-04:00"
    )  # 2024-07-18 00:00 to 06:00 UTC

    assert [p.name for p in result.downloaded] == HRRR_DAY[:2]


def test_an_empty_listing_raises(hrrr, dest):
    hrrr.listing.clear()
    hrrr._write_listing()

    with pytest.raises(ValueError, match="no 'md5  name' lines"):
        hrrr.archive.sync(dest, **DAY)


# ---------------------------------------------------------------------------
# A run with nothing to do
# ---------------------------------------------------------------------------


def test_an_unchanged_run_reads_nothing(hrrr, dest, monkeypatch):
    first = hrrr.archive.sync(dest, **DAY)
    monkeypatch.setattr(archives, "_md5_of_file", no_files)
    monkeypatch.setattr(hrrr.archive, "_download", no_files)

    second = hrrr.archive.sync(dest, **DAY)

    assert second.unchanged == first.downloaded
    assert second.downloaded == second.adopted == second.mismatched == []


def test_a_file_noaa_corrected_is_replaced_where_it_is(hrrr, dest):
    hrrr.archive.sync(dest, **DAY)
    name = HRRR_DAY[1]
    hrrr.put(name, b"corrected by NOAA")

    result = hrrr.archive.sync(dest, **DAY)

    path = dest / "2024" / "07" / name
    assert result.downloaded == [path]
    assert len(result.unchanged) == 3
    assert path.read_bytes() == b"corrected by NOAA"
    manifest = (dest / "arlmet-manifest.md5").read_text()
    assert f"{md5(b'corrected by NOAA')}  2024/07/{name}" in manifest
    assert manifest.count(name) == 1  # the old line is gone
    assert hidden_files(dest) == []


def test_a_deleted_file_is_fetched_again(hrrr, dest):
    hrrr.archive.sync(dest, **DAY)
    gone = dest / "2024" / "07" / HRRR_DAY[0]
    gone.unlink()

    result = hrrr.archive.sync(dest, **DAY)

    assert result.downloaded == [gone]
    assert gone.read_bytes() == content(HRRR_DAY[0])


# ---------------------------------------------------------------------------
# A folder that was filled another way
# ---------------------------------------------------------------------------


def test_v1_files_in_one_flat_folder_are_adopted_not_downloaded_again(
    tmp_path, dest, monkeypatch
):
    """The real mirror: HRRR v1 files sit flat, NOAA's layout is YYYY/MM/."""
    mirror = FakeMirror(
        tmp_path / "noaa",
        HRRRv1Archive(),
        monkeypatch,
        "2019-01-01",
        "2019-01-01 23:00",
    )
    names = [f"hysplit.20190101.{h:02d}z.hrrra" for h in (0, 6, 12, 18)]
    dest.mkdir(parents=True)
    for name in names:
        mirror.put(name, content(name))
        (dest / name).write_bytes(content(name))
    monkeypatch.setattr(mirror.archive, "_download", no_files)

    result = mirror.archive.sync(dest, start="2019-01-01", end="2019-01-01 23:00")

    assert result.adopted == [dest / name for name in names]
    assert result.downloaded == []
    assert not (dest / "2019").exists()  # nothing was made in NOAA's layout
    manifest = (dest / "arlmet-manifest.md5").read_text().splitlines()
    assert manifest == [f"{md5(content(n))}  {n}" for n in names]


def test_a_second_run_over_a_flat_folder_reads_nothing(tmp_path, dest, monkeypatch):
    mirror = FakeMirror(
        tmp_path / "noaa",
        HRRRv1Archive(),
        monkeypatch,
        "2019-01-01",
        "2019-01-01 05:00",
    )
    name = "hysplit.20190101.00z.hrrra"
    dest.mkdir(parents=True)
    mirror.put(name, content(name))
    (dest / name).write_bytes(content(name))
    kwargs = {"start": "2019-01-01", "end": "2019-01-01 05:00"}
    mirror.archive.sync(dest, **kwargs)
    monkeypatch.setattr(archives, "_md5_of_file", no_files)
    monkeypatch.setattr(mirror.archive, "_download", no_files)

    result = mirror.archive.sync(dest, **kwargs)

    assert result.unchanged == [dest / name]


def test_a_copy_in_some_other_folder_layout_is_found_by_name(hrrr, dest, monkeypatch):
    elsewhere = dest / "old" / "disk2" / "2024"
    elsewhere.mkdir(parents=True)
    for name in HRRR_DAY:
        (elsewhere / name).write_bytes(content(name))
    monkeypatch.setattr(hrrr.archive, "_download", no_files)

    result = hrrr.archive.sync(dest, **DAY)

    assert result.adopted == [elsewhere / name for name in HRRR_DAY]
    assert not (dest / "2024" / "07").exists()


def test_a_copy_in_a_linked_folder_is_found(hrrr, dest, tmp_path, monkeypatch):
    other_disk = tmp_path / "other_disk"
    other_disk.mkdir()
    for name in HRRR_DAY:
        (other_disk / name).write_bytes(content(name))
    dest.mkdir(parents=True)
    (dest / "2024").symlink_to(other_disk, target_is_directory=True)
    monkeypatch.setattr(hrrr.archive, "_download", no_files)

    result = hrrr.archive.sync(dest, **DAY)

    assert result.adopted == [dest / "2024" / name for name in HRRR_DAY]


def test_a_wrong_existing_file_is_reported_and_left_alone(hrrr, dest):
    name = HRRR_DAY[2]
    path = dest / name
    dest.mkdir(parents=True)
    path.write_bytes(b"short file from a killed transfer")

    result = hrrr.archive.sync(dest, **DAY)

    assert result.mismatched == [path]
    assert path.read_bytes() == b"short file from a killed transfer"
    assert not (dest / "2024" / "07" / name).exists()  # not fetched elsewhere either
    assert len(result.downloaded) == 3
    assert name not in (dest / "arlmet-manifest.md5").read_text()


def test_the_copy_at_the_archive_path_wins_over_another(hrrr, dest):
    name = HRRR_DAY[0]
    (dest / "2024" / "07").mkdir(parents=True)
    (dest / "0_old").mkdir()  # sorts before "2024"
    (dest / "0_old" / name).write_bytes(b"not the file")
    (dest / "2024" / "07" / name).write_bytes(content(name))

    result = hrrr.archive.sync(dest, **DAY)

    assert result.adopted == [dest / "2024" / "07" / name]
    assert result.mismatched == []


def test_hidden_files_and_folders_are_not_copies(hrrr, dest):
    name = HRRR_DAY[0]
    (dest / ".trash").mkdir(parents=True)
    (dest / ".trash" / name).write_bytes(content(name))
    (dest / f".{name}.abc123.partial").write_bytes(b"half")

    result = hrrr.archive.sync(dest, **DAY)

    assert dest / "2024" / "07" / name in result.downloaded
    assert (dest / f".{name}.abc123.partial").exists()  # not touched


def test_files_noaa_does_not_list_are_left_alone(hrrr, dest):
    dest.mkdir(parents=True)
    keep = [dest / "20240718_00-05_hrrr~20240719123000~", dest / "notes.txt"]
    for path in keep:
        path.write_bytes(b"mine")

    result = hrrr.archive.sync(dest, **DAY)

    assert all(path.read_bytes() == b"mine" for path in keep)
    assert len(result.downloaded) == 4


def test_the_manifest_drops_a_file_that_is_gone_and_keeps_the_rest(hrrr, dest):
    hrrr.archive.sync(dest, **DAY)
    other = dest / "2024" / "07" / "some_other_file"
    other.write_bytes(b"x")
    with open(dest / "arlmet-manifest.md5", "a") as f:
        f.write(f"{md5(b'x')}  2024/07/some_other_file\n")
        f.write(f"{md5(b'y')}  2024/07/deleted_file\n")

    hrrr.archive.sync(dest, **DAY)

    manifest = (dest / "arlmet-manifest.md5").read_text()
    assert "some_other_file" in manifest
    assert "deleted_file" not in manifest


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_dry_run_writes_and_reads_nothing(hrrr, dest, monkeypatch):
    monkeypatch.setattr(archives, "_md5_of_file", no_files)
    monkeypatch.setattr(hrrr.archive, "_download", no_files)

    result = hrrr.archive.sync(dest, dry_run=True, **DAY)

    assert result.dry_run
    assert result.downloaded == [dest / "2024" / "07" / name for name in HRRR_DAY]
    assert not dest.exists()  # not even the folder


def test_dry_run_lists_what_it_would_check_without_reading_it(hrrr, dest, monkeypatch):
    dest.mkdir(parents=True)
    (dest / HRRR_DAY[0]).write_bytes(content(HRRR_DAY[0]))
    monkeypatch.setattr(archives, "_md5_of_file", no_files)
    monkeypatch.setattr(hrrr.archive, "_download", no_files)

    result = hrrr.archive.sync(dest, dry_run=True, **DAY)

    assert result.adopted == [dest / HRRR_DAY[0]]
    assert len(result.downloaded) == 3
    assert sorted(p.name for p in dest.iterdir()) == [
        HRRR_DAY[0]
    ]  # no manifest, no lock


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------


def test_a_second_run_is_refused_while_the_first_holds_the_lock(hrrr, dest):
    lock = dest / ".arlmet-sync.lock"
    lock.mkdir(parents=True)
    (lock / "owner").write_text(
        "host node42, process 7, started 2024-07-18 05:30:00 UTC\n"
    )

    with pytest.raises(RuntimeError, match=r"node42.*started 2024-07-18 05:30:00 UTC"):
        hrrr.archive.sync(dest, **DAY)

    assert (lock / "owner").exists()  # the first run's lock is not removed
    assert list(dest.rglob("2024")) == []  # nothing fetched


def test_a_run_inside_a_run_is_refused(hrrr, dest, monkeypatch):
    real_download = hrrr.archive._download

    def download_then_start_another_run(*args, **kwargs):
        with pytest.raises(RuntimeError, match="Another sync holds the lock"):
            hrrr.archive.sync(dest, **DAY)
        return real_download(*args, **kwargs)

    monkeypatch.setattr(hrrr.archive, "_download", download_then_start_another_run)

    result = hrrr.archive.sync(dest, **DAY)

    assert len(result.downloaded) == 4
    assert not (dest / ".arlmet-sync.lock").exists()


def test_the_lock_is_removed_after_an_error(hrrr, dest, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(hrrr.archive, "_download", boom)

    with pytest.raises(RuntimeError, match="boom"):
        hrrr.archive.sync(dest, **DAY)

    assert not (dest / ".arlmet-sync.lock").exists()
    assert hidden_files(dest) == []


def test_the_manifest_keeps_what_was_checked_before_an_error(hrrr, dest, monkeypatch):
    real_download = hrrr.archive._download
    calls = []

    def second_one_fails(url, target, opts, **kwargs):
        calls.append(target)
        if len(calls) == 2:
            raise RuntimeError("killed")
        return real_download(url, target, opts, **kwargs)

    monkeypatch.setattr(hrrr.archive, "_download", second_one_fails)

    with pytest.raises(RuntimeError, match="killed"):
        hrrr.archive.sync(dest, **DAY)

    manifest = (dest / "arlmet-manifest.md5").read_text().splitlines()
    assert manifest == [f"{md5(content(HRRR_DAY[0]))}  2024/07/{HRRR_DAY[0]}"]


# ---------------------------------------------------------------------------
# Products whose files are not 6-hourly, and the helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("archive", "start", "end", "expected"),
    [
        (
            GDASArchive(),  # weekly: w1 is days 1 to 7, w5 days 29 to the end
            "2024-07-01",
            "2024-07-31",
            [f"2024/gdas1.jul24.w{n}" for n in range(1, 6)],
        ),
        (
            ReanalysisArchive(),
            "2024-01-15",
            "2024-03-02",
            [f"2024/RP2024{m:02d}.gbl" for m in (1, 2, 3)],
        ),
        (
            NARRArchive(),
            "2019-11-30",
            "2019-12-01",
            ["2019/NARR201911", "2019/NARR201912"],
        ),
    ],
    ids=["weekly", "monthly", "monthly-narr"],
)
def test_weekly_and_monthly_products_land_in_their_archive_layout(
    tmp_path, dest, monkeypatch, archive, start, end, expected
):
    mirror = FakeMirror(tmp_path / "noaa", archive, monkeypatch, start, end)
    for name in mirror.paths:
        mirror.put(name, content(name))

    result = archive.sync(dest, start=start, end=end)

    assert result.downloaded == [dest / rel for rel in expected]
    assert result.failed == []


def test_names_of_a_listing_with_no_archive_path_are_skipped(hrrr, dest):
    hrrr.listing["listing_prev.md5.txt"] = md5(b"x")
    hrrr.listing["somebody_elses_file"] = md5(b"y")
    hrrr._write_listing()

    result = hrrr.archive.sync(dest, **DAY)

    assert len(result.downloaded) == 4


def test_parse_md5_lines_reads_md5sum_output():
    text = (
        f"{'A' * 32}  plain\n"
        f"{'b' * 32} *binary mode\r\n"
        "\n"
        "not a line\n"
        f"{'c' * 31}  too short\n"
        f"{'d' * 32}  has two  spaces\n"
        f"{'e' * 32}  twice\n"
        f"{'f' * 32}  twice\n"
    )

    assert archives._parse_md5_lines(text) == {
        "plain": "a" * 32,
        "binary mode": "b" * 32,
        "has two  spaces": "d" * 32,
        "twice": "f" * 32,
    }


def test_download_without_a_checksum_is_unchanged(tmp_path):
    src = tmp_path / "src"
    src.write_bytes(b"abc")
    dest = tmp_path / "out"

    HRRRArchive()._download(f"file://{src}", dest, {})

    assert dest.read_bytes() == b"abc"


def test_repr_gives_counts_not_paths():
    result = SyncResult(downloaded=[Path("a"), Path("b")], failed=[Path("c")])

    assert repr(result) == (
        "SyncResult(downloaded=2, adopted=0, unchanged=0, mismatched=0, failed=1, "
        "dry_run=False)"
    )


def test_sync_needs_fsspec(hrrr, dest, monkeypatch):
    monkeypatch.setitem(sys.modules, "fsspec", None)

    with pytest.raises(ImportError, match=r"Archive\.sync"):
        hrrr.archive.sync(dest, **DAY)


def test_the_default_end_is_now(dest, monkeypatch):
    seen = []
    archive = HRRRArchive()

    def record(start, end):
        seen.append((start, end))
        return []

    monkeypatch.setattr(archive, "paths_for_range", record)
    monkeypatch.setattr(archive, "_url", lambda path, mirror: "file:///nonexistent")
    monkeypatch.setattr(archive, "_storage_options", lambda mirror: {})

    with pytest.raises(FileNotFoundError):
        archive.sync(dest, dry_run=True)

    start, end = seen[0]
    assert start == archive.start_date
    assert abs(end - pd.Timestamp.now("UTC").tz_localize(None)) < pd.Timedelta(
        minutes=1
    )
