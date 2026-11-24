"""Tests for the catalog of a directory of ARL files, read from their headers."""

import os
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import arlmet
from arlmet import ARLFormatWarning, Catalog, _catalog, catalog

AXIS = arlmet.SigmaAxis(levels=[1.0])


def write_met(path, times, *, source="TEST", nx=20):
    """Write a surface-only ARL file holding *times*, with ``write_dataset``."""
    times = pd.to_datetime(times)
    grid = xr.Variable(
        (),
        0,
        attrs={
            "nx": nx,
            "ny": 20,
            "pole_lat": 90.0,
            "pole_lon": 0.0,
            "tangent_lat": 1.0,
            "tangent_lon": 1.0,
            "grid_size": 0.0,
            "orientation": 0.0,
            "cone_angle": 0.0,
            "sync_x": 1.0,
            "sync_y": 1.0,
            "sync_lat": 30.0,
            "sync_lon": -120.0,
        },
    )
    data = np.full((len(times), 20, nx), 1000.0, dtype=np.float32)
    ds = xr.Dataset(
        {"PRSS": (("time", "lat", "lon"), data)},
        coords={"time": times, "arl_grid": grid},
        attrs={"source": source},
    )
    arlmet.write_dataset(ds, path, vertical_axis=AXIS)
    return path


def six_hour_blocks(directory, first, n_blocks):
    """Write *n_blocks* files of six hourly steps from *first*, named ``YYYYMMDD_HH``."""
    paths = []
    for i in range(n_blocks):
        start = pd.Timestamp(first) + pd.Timedelta(hours=6 * i)
        times = pd.date_range(start, periods=6, freq="h")
        paths.append(write_met(directory / f"{start:%Y%m%d_%H}", times))
    return paths


@pytest.fixture
def archive(tmp_path):
    """Four six-hour files of hourly steps, 2024-07-01 00:00 to 23:00."""
    # Resolved, as the table's paths are (a temporary folder can be reached
    # through a link, as /var is on macOS).
    folder = tmp_path.resolve() / "hrrr"
    folder.mkdir()
    six_hour_blocks(folder, "2024-07-01 00:00", 4)
    return folder


@pytest.fixture(autouse=True)
def user_cache(tmp_path, monkeypatch):
    """Point the user cache folder at a temporary folder for every test."""
    folder = tmp_path / "user-cache"
    monkeypatch.setenv("ARLMET_CACHE_DIR", str(folder))
    return folder


class TestTable:
    def test_one_row_per_file_from_the_headers(self, archive):
        cat = catalog(archive)

        assert isinstance(cat, Catalog)
        assert len(cat) == 4
        table = cat.table
        assert list(table.columns) == [
            "path",
            "source",
            "grid",
            "start",
            "end",
            "step",
            "error",
        ]
        assert [p.name for p in table["path"]] == [
            "20240701_00",
            "20240701_06",
            "20240701_12",
            "20240701_18",
        ]
        first = table.iloc[0]
        assert first["path"] == archive / "20240701_00"
        assert first["source"] == "TEST"
        assert first["grid"] == "Grid(latlon, 20×20)"
        assert first["start"] == pd.Timestamp("2024-07-01 00:00")
        assert first["end"] == pd.Timestamp("2024-07-01 05:00")
        assert first["step"] == pd.Timedelta(hours=1)
        assert first["error"] == ""
        assert table["end"].iloc[-1] == pd.Timestamp("2024-07-01 23:00")
        assert cat.step == pd.Timedelta(hours=1)

    def test_names_do_not_matter(self, tmp_path):
        # The times come from the headers, so any name works.
        write_met(
            tmp_path / "zzz", pd.date_range("2024-07-01 00:00", periods=3, freq="h")
        )
        write_met(
            tmp_path / "aaa", pd.date_range("2024-07-01 03:00", periods=3, freq="h")
        )

        cat = catalog(tmp_path)

        assert cat.covering("2024-07-01 01:00", "2024-07-01 04:00") == [
            tmp_path / "zzz",
            tmp_path / "aaa",
        ]

    def test_a_window_with_a_time_zone_is_converted_to_utc(self, archive):
        cat = catalog(archive)
        # 2024-06-30 18:00 to 2024-07-01 04:00 in Denver is 00:00 to 10:00 UTC.
        start = pd.Timestamp("2024-06-30 18:00", tz="America/Denver")
        end = pd.Timestamp("2024-07-01 04:00", tz="America/Denver")

        assert cat.covering(start, end) == cat.covering(
            "2024-07-01 00:00", "2024-07-01 10:00"
        )
        assert cat.missing(start, end) == []

    def test_a_file_that_is_not_arl_is_kept_with_its_error(self, archive):
        (archive / "README.TXT").write_text("HRRR 3 km archive\n" * 40)
        (archive / "empty").write_bytes(b"")

        cat = catalog(archive)

        errors = dict(
            zip([p.name for p in cat.table["path"]], cat.table["error"], strict=True)
        )
        assert "README.TXT" in errors["README.TXT"]  # the message names the file
        assert "holds no index record" in errors["empty"]
        rows = cat.table.set_index(cat.table["path"].map(lambda p: p.name))
        assert pd.isna(rows.loc["README.TXT", "start"])
        assert rows.loc["empty", "source"] == ""
        # Unusable rows sort last and are not used to cover a window.
        assert [p.name for p in cat.table["path"][-2:]] == ["README.TXT", "empty"]
        assert len(cat.covering("2024-07-01 00:00", "2024-07-01 23:00")) == 4

    def test_uneven_time_steps_are_an_error_and_not_used(self, tmp_path):
        times = pd.to_datetime(
            ["2024-07-01 00:00", "2024-07-01 01:00", "2024-07-01 03:00"]
        )
        write_met(tmp_path / "gap", times)

        cat = catalog(tmp_path)

        row = cat.table.iloc[0]
        assert row["error"] == (
            "Its time steps are not evenly spaced: 2024-07-01 01:00 to "
            "2024-07-01 03:00 is 2 h; the first two are 1 h apart."
        )
        assert row["start"] == pd.Timestamp("2024-07-01 00:00")
        assert row["end"] == pd.Timestamp("2024-07-01 03:00")
        assert pd.isna(row["step"])
        with pytest.raises(ValueError, match="No usable ARL file"):
            cat.covering("2024-07-01 00:00", "2024-07-01 01:00")

    def test_a_file_of_one_time_step_has_no_step(self, tmp_path):
        write_met(tmp_path / "one", ["2024-07-01 06:00"])

        cat = catalog(tmp_path)

        assert pd.isna(cat.table["step"].iloc[0])
        assert cat.step is None
        assert cat.covering("2024-07-01 06:00", "2024-07-01 06:00") == [
            tmp_path / "one"
        ]
        assert cat.missing("2024-07-01 06:00", "2024-07-01 07:00") == [
            pd.Timestamp("2024-07-01 07:00")
        ]

    def test_a_repeated_time_step_counts_once(self, tmp_path):
        once = write_met(
            tmp_path / "once", pd.date_range("2024-07-01 00:00", periods=2, freq="h")
        )
        (tmp_path / "twice").write_bytes(once.read_bytes() * 2)
        once.unlink()

        with pytest.warns(ARLFormatWarning, match="repeated"):
            cat = catalog(tmp_path)

        row = cat.table.iloc[0]
        assert row["error"] == ""
        assert (row["start"], row["end"]) == (
            pd.Timestamp("2024-07-01 00:00"),
            pd.Timestamp("2024-07-01 01:00"),
        )

    def test_hidden_files_backups_and_subfolders_are_not_read(self, archive):
        shutil.copy(archive / "20240701_00", archive / ".20240701_00.partial")
        shutil.copy(archive / "20240701_00", archive / "20240701_00~20260101~")
        (archive / "deprecated").mkdir()
        shutil.copy(archive / "20240701_00", archive / "deprecated" / "20240701_00")

        cat = catalog(archive)

        assert len(cat) == 4
        assert cat.covering("2024-07-01 00:00", "2024-07-01 02:00") == [
            archive / "20240701_00"
        ]

    @pytest.mark.parametrize("ending", [".bad", ".partial", ".lock", ".lock.1"])
    def test_a_copy_set_aside_is_not_read(self, archive, ending):
        shutil.copy(archive / "20240701_00", archive / f"20240701_00{ending}")

        cat = catalog(archive)

        assert len(cat) == 4
        assert cat.covering("2024-07-01 00:00", "2024-07-01 02:00") == [
            archive / "20240701_00"
        ]

    def test_a_name_that_only_starts_like_a_lock_file_is_read(self, archive):
        shutil.copy(archive / "20240701_00", archive / "20240701_00.locked")

        assert len(catalog(archive)) == 5

    def test_a_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No directory"):
            catalog(tmp_path / "nowhere")

    def test_repr_shows_file_names(self, archive):
        text = repr(catalog(archive))

        assert text.startswith(f"Catalog({str(archive)!r}: 4 files, 0 not used)")
        assert "20240701_18" in text
        assert str(archive / "20240701_18") not in text


class TestCovering:
    def test_brackets_the_window(self, archive):
        # 05:30 to 12:00 needs 05:00 (before the start) through 12:00.
        cat = catalog(archive)

        assert cat.covering("2024-07-01 05:30", "2024-07-01 12:00") == [
            archive / "20240701_00",
            archive / "20240701_06",
            archive / "20240701_12",
        ]
        assert cat.missing("2024-07-01 05:30", "2024-07-01 12:00") == []

    def test_a_window_on_whole_hours_needs_only_its_own(self, archive):
        cat = catalog(archive)

        assert cat.covering("2024-07-01 06:00", "2024-07-01 11:00") == [
            archive / "20240701_06"
        ]
        assert cat.covering("2024-07-01 11:00", "2024-07-01 11:00") == [
            archive / "20240701_06"
        ]
        # 11:00 to 11:30 needs 12:00, in the next file.
        assert cat.covering("2024-07-01 11:00", "2024-07-01 11:30") == [
            archive / "20240701_06",
            archive / "20240701_12",
        ]

    def test_missing_hours_are_listed(self, archive):
        (archive / "20240701_06").unlink()
        cat = catalog(archive)

        hours = pd.date_range("2024-07-01 06:00", periods=6, freq="h")
        assert cat.missing("2024-07-01 05:30", "2024-07-01 12:00") == list(hours)
        with pytest.raises(ValueError) as exc:
            cat.covering("2024-07-01 05:30", "2024-07-01 12:00")
        assert str(exc.value) == (
            f"No file in {archive} holds 2024-07-01 06:00, 2024-07-01 07:00, "
            "2024-07-01 08:00, 2024-07-01 09:00, 2024-07-01 10:00, "
            "2024-07-01 11:00, which 2024-07-01 05:00 to 2024-07-01 12:00 needs."
        )

    def test_hours_past_the_archive_are_missing(self, archive):
        cat = catalog(archive)

        # The last time is 23:00; 00:00 and 01:00 the next day are needed.
        assert cat.missing("2024-07-01 22:00", "2024-07-02 00:30") == [
            pd.Timestamp("2024-07-02 00:00"),
            pd.Timestamp("2024-07-02 01:00"),
        ]
        # Before the first time, the steps are counted back from it.
        assert cat.missing("2024-06-30 22:30", "2024-07-01 00:00") == [
            pd.Timestamp("2024-06-30 22:00"),
            pd.Timestamp("2024-06-30 23:00"),
        ]

    def test_more_than_ten_missing_hours_are_counted(self, archive):
        cat = catalog(archive)

        with pytest.raises(ValueError, match=r"2024-07-02 09:00 and 2 more, which"):
            cat.covering("2024-07-02 00:00", "2024-07-02 11:00")

    def test_two_files_with_one_hour_name_both(self, archive):
        shutil.copy(archive / "20240701_06", archive / "20240701_06.copy")
        cat = catalog(archive)

        # Hours the copy does not hold are found as before.
        assert cat.covering("2024-07-01 00:00", "2024-07-01 05:00") == [
            archive / "20240701_00"
        ]
        # An hour in both is not missing, but the files are ambiguous.
        assert cat.missing("2024-07-01 05:00", "2024-07-01 07:00") == []
        with pytest.raises(ValueError) as exc:
            cat.covering("2024-07-01 05:00", "2024-07-01 07:00")
        assert str(exc.value) == (
            f"2024-07-01 06:00 is in two files, {archive / '20240701_06'} and "
            f"{archive / '20240701_06.copy'}. Move one of them out of {archive}."
        )

    def test_a_short_file_inside_a_longer_one_is_not_used(self, archive):
        # Two hours of the 06:00 file, as a crop of the same grid would hold.
        write_met(archive / "crop", ["2024-07-01 07:00", "2024-07-01 08:00"])
        cat = catalog(archive)

        assert len(cat) == 5
        assert cat.covering("2024-07-01 07:00", "2024-07-01 08:00") == [
            archive / "20240701_06"
        ]

    def test_a_short_file_of_another_grid_still_names_both(self, archive):
        write_met(archive / "crop", ["2024-07-01 07:00", "2024-07-01 08:00"], nx=21)
        cat = catalog(archive)

        with pytest.raises(ValueError, match="is in two files"):
            cat.covering("2024-07-01 07:00", "2024-07-01 08:00")

    def test_missing_ok_returns_the_files_for_the_hours_held(self, archive):
        (archive / "20240701_12").unlink()
        cat = catalog(archive)

        assert cat.covering(
            "2024-07-01 10:00", "2024-07-01 19:00", missing_ok=True
        ) == [archive / "20240701_06", archive / "20240701_18"]
        assert cat.missing("2024-07-01 10:00", "2024-07-01 19:00") == list(
            pd.date_range("2024-07-01 12:00", periods=6, freq="h")
        )

    def test_the_step_is_the_most_common_spacing(self, tmp_path):
        # Three-hourly files, plus one hourly step that does not change it.
        write_met(
            tmp_path / "a", pd.date_range("2024-07-01 00:00", periods=4, freq="3h")
        )
        write_met(
            tmp_path / "b", pd.date_range("2024-07-01 12:00", periods=4, freq="3h")
        )
        write_met(tmp_path / "c", ["2024-07-02 00:00", "2024-07-02 01:00"])

        cat = catalog(tmp_path)

        assert cat.step == pd.Timedelta(hours=3)
        assert cat.covering("2024-07-01 07:00", "2024-07-01 13:00") == [
            tmp_path / "a",
            tmp_path / "b",
        ]

    def test_start_after_end_raises(self, archive):
        cat = catalog(archive)

        with pytest.raises(ValueError, match="is after end"):
            cat.covering("2024-07-01 06:00", "2024-07-01 05:00")

    def test_an_empty_directory_covers_nothing(self, tmp_path):
        cat = catalog(tmp_path)

        assert len(cat) == 0
        with pytest.raises(ValueError, match="No usable ARL file"):
            cat.missing("2024-07-01 00:00", "2024-07-01 01:00")


class TestCache:
    def test_cache_is_beside_a_writable_directory(self, archive, user_cache):
        cat = catalog(archive)

        assert cat.cache == archive / ".arlmet-catalog.csv"
        assert cat.cache.is_file()
        assert not user_cache.exists()

    def test_cache_is_reused_and_refreshed(self, archive, monkeypatch):
        catalog(archive)
        read = []
        real_read_file = _catalog._read_file
        monkeypatch.setattr(
            _catalog,
            "_read_file",
            lambda path: read.append(path.name) or real_read_file(path),
        )

        # Nothing changed: no file is opened.
        again = catalog(archive)
        assert read == []
        assert len(again) == 4
        assert again.table["start"].iloc[0] == pd.Timestamp("2024-07-01 00:00")
        assert again.table["step"].iloc[0] == pd.Timedelta(hours=1)

        # A new file, a rewritten file and a removed file.
        six_hour_blocks(archive, "2024-07-02 00:00", 1)
        write_met(
            archive / "20240701_12",
            pd.date_range("2024-07-01 12:00", periods=3, freq="h"),
        )
        (archive / "20240701_18").unlink()
        refreshed = catalog(archive)

        assert sorted(read) == ["20240701_12", "20240702_00"]
        assert [p.name for p in refreshed.table["path"]] == [
            "20240701_00",
            "20240701_06",
            "20240701_12",
            "20240702_00",
        ]
        assert refreshed.table["end"].iloc[2] == pd.Timestamp("2024-07-01 14:00")

    def test_an_interrupted_index_keeps_what_it_saved(self, archive, monkeypatch):
        monkeypatch.setattr(_catalog, "_SAVE_EVERY", 2)
        real_read_file = _catalog._read_file

        def stops_at_the_last(path):
            if path.name == "20240701_18":
                raise RuntimeError("interrupted")
            return real_read_file(path)

        monkeypatch.setattr(_catalog, "_read_file", stops_at_the_last)
        with pytest.raises(RuntimeError, match="interrupted"):
            catalog(archive)

        # The cache was saved after the first two files; the rest are read now.
        read = []
        monkeypatch.setattr(
            _catalog,
            "_read_file",
            lambda path: read.append(path.name) or real_read_file(path),
        )
        cat = catalog(archive)
        assert sorted(read) == ["20240701_12", "20240701_18"]
        assert len(cat) == 4

    def test_errors_are_cached_too(self, archive, monkeypatch):
        (archive / "README.TXT").write_text("not ARL\n")
        first = catalog(archive)
        monkeypatch.setattr(_catalog, "_read_file", pytest.fail)

        again = catalog(archive)

        assert list(again.table["error"]) == list(first.table["error"])

    def test_a_cache_of_another_layout_is_rebuilt(self, archive):
        # The first version keyed rows by file name; its cache is ignored.
        stat = (archive / "20240701_00").stat()
        (archive / ".arlmet-catalog.csv").write_text(
            "name,size,mtime_ns,source,grid,start,end,step,error\n"
            f"20240701_00,{stat.st_size},{stat.st_mtime_ns},OLD,g,"
            "2000-01-01 00:00:00,2000-01-01 05:00:00,0 days 01:00:00,\n"
        )

        cat = catalog(archive)

        assert cat.table["start"].iloc[0] == pd.Timestamp("2024-07-01 00:00")
        assert set(cat.table["source"]) == {"TEST"}
        assert cat.cache is not None
        assert cat.cache.read_text().startswith("path,size,mtime_ns,")

    def test_a_recursive_index_has_its_own_cache(self, archive):
        flat = catalog(archive)
        deep = catalog(archive, recursive=True)

        assert flat.cache == archive / ".arlmet-catalog.csv"
        assert deep.cache == archive / ".arlmet-catalog-recursive.csv"

    def test_read_only_directory_uses_the_user_cache(self, archive, user_cache):
        before = sorted(p.name for p in archive.iterdir())
        archive.chmod(0o555)
        try:
            if os.access(archive, os.W_OK):
                pytest.skip("running as a user who can write anywhere")
            cat = catalog(archive)
        finally:
            archive.chmod(0o755)

        assert cat.cache is not None
        assert cat.cache.parent == user_cache / "catalog"
        assert cat.cache.name.startswith("hrrr-") and cat.cache.suffix == ".csv"
        assert cat.cache.is_file()
        assert sorted(p.name for p in archive.iterdir()) == before

    def test_a_published_table_is_the_starting_point_for_another_cache(
        self, archive, tmp_path, monkeypatch
    ):
        published = catalog(archive)  # the maintainer's run: table beside the files
        before = published.cache.read_bytes()
        reads = []
        real = _catalog._read_file
        monkeypatch.setattr(
            _catalog, "_read_file", lambda p: reads.append(p) or real(p)
        )

        own = catalog(archive, cache_dir=tmp_path / "mine")

        assert reads == []
        assert own.cache.parent == tmp_path / "mine"
        assert own.cache.is_file()
        assert own.table["path"].tolist() == published.table["path"].tolist()
        assert published.cache.read_bytes() == before

    def test_a_stale_published_table_costs_only_the_changed_files(
        self, archive, tmp_path, monkeypatch
    ):
        catalog(archive)
        new = write_met(
            archive / "20240702_00", pd.date_range("2024-07-02", periods=2, freq="h")
        )
        reads = []
        real = _catalog._read_file
        monkeypatch.setattr(
            _catalog, "_read_file", lambda p: reads.append(p) or real(p)
        )

        own = catalog(archive, cache_dir=tmp_path / "mine")

        assert [p.resolve() for p in reads] == [new.resolve()]
        assert new.resolve() in set(own.table["path"])

    def test_the_user_cache_follows_xdg_without_the_variable(
        self, archive, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("ARLMET_CACHE_DIR")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        monkeypatch.setattr(_catalog.sys, "platform", "linux")
        monkeypatch.setattr(_catalog.os, "access", lambda *args: False)

        cat = catalog(archive)

        assert cat.cache is not None
        assert cat.cache.parent == tmp_path / "xdg" / "arlmet" / "catalog"
        assert not (archive / ".arlmet-catalog.csv").exists()

    def test_cache_dir_argument_wins(self, archive, tmp_path):
        cat = catalog(archive, cache_dir=tmp_path / "mine")

        assert cat.cache is not None
        assert cat.cache.parent == tmp_path / "mine"
        assert cat.cache.is_file()
        assert not (archive / ".arlmet-catalog.csv").exists()

    def test_two_directories_of_one_name_get_two_caches(self, tmp_path):
        a = tmp_path / "a" / "hrrr"
        b = tmp_path / "b" / "hrrr"
        a.mkdir(parents=True)
        b.mkdir(parents=True)

        first = catalog(a, cache_dir=tmp_path / "caches")
        second = catalog(b, cache_dir=tmp_path / "caches")

        assert first.cache != second.cache

    def test_an_unwritable_cache_raises_a_clear_error(self, archive, tmp_path):
        blocker = tmp_path / "a-file"
        blocker.write_text("")

        with pytest.raises(
            OSError, match="Cannot write the catalog cache.*ARLMET_CACHE_DIR"
        ):
            catalog(archive, cache_dir=blocker)


class TestLinksAndFolders:
    def test_a_link_and_its_target_are_one_row(self, archive):
        os.symlink(archive / "20240701_06", archive / "zz-link")

        cat = catalog(archive)

        assert len(cat) == 4
        assert list(cat.table["path"]).count(archive / "20240701_06") == 1
        assert cat.covering("2024-07-01 06:00", "2024-07-01 07:00") == [
            archive / "20240701_06"
        ]

    def test_a_link_is_shown_by_its_real_path(self, tmp_path):
        # A flat folder of links into a YYYY/MM tree, as a shared archive is.
        top = tmp_path.resolve() / "hrrr"
        month = top / "2024" / "07"
        month.mkdir(parents=True)
        six_hour_blocks(month, "2024-07-01 00:00", 2)
        for name in ("20240701_00", "20240701_06"):
            os.symlink(month / name, top / name)

        flat = catalog(top)
        deep = catalog(top, recursive=True)

        expected = [month / "20240701_00", month / "20240701_06"]
        assert list(flat.table["path"]) == expected
        assert list(deep.table["path"]) == expected

    def test_a_link_to_a_link_is_followed(self, tmp_path, archive):
        other = tmp_path / "other"
        other.mkdir()
        os.symlink(archive / "20240701_12", other / "first")
        os.symlink(other / "first", other / "second")

        cat = catalog(other)

        assert list(cat.table["path"]) == [archive / "20240701_12"]
        assert cat.table["error"].iloc[0] == ""

    def test_a_broken_link_is_a_row_with_its_error(self, archive, tmp_path):
        os.symlink(tmp_path / "gone", archive / "broken")

        cat = catalog(archive)
        again = catalog(archive)  # the cache does not hide it

        for table in (cat.table, again.table):
            row = table.iloc[-1]
            assert row["path"] == archive / "broken"
            assert row["error"].startswith("Cannot follow the link: ")
            assert pd.isna(row["start"])
        assert len(cat.covering("2024-07-01 00:00", "2024-07-01 23:00")) == 4

    def test_a_link_loop_is_a_row_with_its_error(self, archive):
        os.symlink(archive / "loop-b", archive / "loop-a")
        os.symlink(archive / "loop-a", archive / "loop-b")
        # A link from a subfolder back to the top is not followed.
        (archive / "sub").mkdir()
        os.symlink(archive, archive / "sub" / "up")

        cat = catalog(archive, recursive=True)

        errors = cat.table.set_index(cat.table["path"].map(lambda p: p.name))["error"]
        assert errors["loop-a"].startswith("Cannot follow the link: ")
        assert errors["loop-b"].startswith("Cannot follow the link: ")
        assert "up" not in errors
        assert len(cat) == 6

    def test_recursive_reads_a_yyyy_mm_tree(self, tmp_path):
        top = tmp_path.resolve() / "hrrr"
        june, july = top / "2024" / "06", top / "2024" / "07"
        june.mkdir(parents=True)
        july.mkdir(parents=True)
        six_hour_blocks(june, "2024-06-30 18:00", 1)
        six_hour_blocks(july, "2024-07-01 00:00", 2)

        assert len(catalog(top)) == 0
        cat = catalog(top, recursive=True)

        assert len(cat) == 3
        assert cat.covering("2024-06-30 23:30", "2024-07-01 06:00") == [
            june / "20240630_18",
            july / "20240701_00",
            july / "20240701_06",
        ]

    def test_recursive_reads_superseded_copies_too(self, archive):
        # A subfolder of superseded copies is read, and its files overlap.
        (archive / "deprecated").mkdir()
        shutil.copy(archive / "20240701_06", archive / "deprecated" / "20240701_06")

        assert (
            len(catalog(archive).covering("2024-07-01 06:00", "2024-07-01 07:00")) == 1
        )
        cat = catalog(archive, recursive=True)
        with pytest.raises(ValueError, match="is in two files"):
            cat.covering("2024-07-01 06:00", "2024-07-01 07:00")

    def test_hidden_folders_are_not_read(self, archive):
        (archive / ".hidden").mkdir()
        shutil.copy(archive / "20240701_06", archive / ".hidden" / "20240701_06")

        assert len(catalog(archive, recursive=True)) == 4


class TestNames:
    def test_catalog_is_the_function_and_its_module_is_private(self):
        import importlib

        assert callable(arlmet.catalog)
        assert arlmet.catalog is _catalog.catalog
        assert arlmet.Catalog is _catalog.Catalog
        for gone in ("index", "DirectoryIndex"):
            assert not hasattr(arlmet, gone)
        for gone in (
            "arlmet.index",
            "arlmet.directoryindex",
            "arlmet.header",
            "arlmet.indexrecord",
        ):
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module(gone)

    def test_the_codec_is_public_only_in_record_and_packing(self):
        from arlmet import packing, record, vertical

        codec = {
            record: ["Header", "IndexRecord", "LvlInfo", "VarInfo"],
            packing: ["calculate_checksum", "pack", "unpack"],
        }
        for module, names in codec.items():
            for name in names:
                assert name in module.__all__
                assert name not in arlmet.__all__
                assert not hasattr(arlmet, name)
        assert {"R_D", "G"} <= set(vertical.__all__)
        assert {"catalog", "Catalog", "probe", "Probe"} <= set(arlmet.__all__)

    def test_only_modules_with_names_of_their_own_are_public(self):
        import pkgutil

        from arlmet import converters

        public = {
            m.name
            for m in pkgutil.iter_modules(arlmet.__path__)
            if not m.name.startswith("_")
        }
        assert public == {
            "archives",
            "collection",
            "converters",
            "packing",
            "record",
            "vertical",
        }
        assert all(
            m.name.startswith("_") for m in pkgutil.iter_modules(converters.__path__)
        )
