"""Tests for indexing a directory of ARL files by their headers."""

import os
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import arlmet
from arlmet import ARLFormatWarning, DirectoryIndex, index
from arlmet import directory as directory_module

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
        idx = index(archive)

        assert isinstance(idx, DirectoryIndex)
        assert len(idx) == 4
        table = idx.table
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
        assert idx.step == pd.Timedelta(hours=1)

    def test_names_do_not_matter(self, tmp_path):
        # The times come from the headers, so any name works.
        write_met(
            tmp_path / "zzz", pd.date_range("2024-07-01 00:00", periods=3, freq="h")
        )
        write_met(
            tmp_path / "aaa", pd.date_range("2024-07-01 03:00", periods=3, freq="h")
        )

        idx = index(tmp_path)

        assert idx.covering("2024-07-01 01:00", "2024-07-01 04:00") == [
            tmp_path / "zzz",
            tmp_path / "aaa",
        ]

    def test_a_file_that_is_not_arl_is_kept_with_its_error(self, archive):
        (archive / "README.TXT").write_text("HRRR 3 km archive\n" * 40)
        (archive / "empty").write_bytes(b"")

        idx = index(archive)

        errors = dict(
            zip([p.name for p in idx.table["path"]], idx.table["error"], strict=True)
        )
        assert "README.TXT" in errors["README.TXT"]  # the message names the file
        assert errors["empty"] == "The file holds no time step."
        rows = idx.table.set_index(idx.table["path"].map(lambda p: p.name))
        assert pd.isna(rows.loc["README.TXT", "start"])
        assert rows.loc["empty", "source"] == ""
        # Unusable rows sort last and are not used to cover a window.
        assert [p.name for p in idx.table["path"][-2:]] == ["README.TXT", "empty"]
        assert len(idx.covering("2024-07-01 00:00", "2024-07-01 23:00")) == 4

    def test_uneven_time_steps_are_an_error_and_not_used(self, tmp_path):
        times = pd.to_datetime(
            ["2024-07-01 00:00", "2024-07-01 01:00", "2024-07-01 03:00"]
        )
        write_met(tmp_path / "gap", times)

        idx = index(tmp_path)

        row = idx.table.iloc[0]
        assert row["error"] == (
            "Its time steps are not evenly spaced: 2024-07-01 01:00 to "
            "2024-07-01 03:00 is 2 h; the first two are 1 h apart."
        )
        assert row["start"] == pd.Timestamp("2024-07-01 00:00")
        assert row["end"] == pd.Timestamp("2024-07-01 03:00")
        assert pd.isna(row["step"])
        with pytest.raises(ValueError, match="No usable ARL file"):
            idx.covering("2024-07-01 00:00", "2024-07-01 01:00")

    def test_a_file_of_one_time_step_has_no_step(self, tmp_path):
        write_met(tmp_path / "one", ["2024-07-01 06:00"])

        idx = index(tmp_path)

        assert pd.isna(idx.table["step"].iloc[0])
        assert idx.step is None
        assert idx.covering("2024-07-01 06:00", "2024-07-01 06:00") == [
            tmp_path / "one"
        ]
        assert idx.missing("2024-07-01 06:00", "2024-07-01 07:00") == [
            pd.Timestamp("2024-07-01 07:00")
        ]

    def test_a_repeated_time_step_counts_once(self, tmp_path):
        once = write_met(
            tmp_path / "once", pd.date_range("2024-07-01 00:00", periods=2, freq="h")
        )
        (tmp_path / "twice").write_bytes(once.read_bytes() * 2)
        once.unlink()

        with pytest.warns(ARLFormatWarning, match="repeated"):
            idx = index(tmp_path)

        row = idx.table.iloc[0]
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

        idx = index(archive)

        assert len(idx) == 4
        assert idx.covering("2024-07-01 00:00", "2024-07-01 02:00") == [
            archive / "20240701_00"
        ]

    def test_a_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No directory"):
            index(tmp_path / "nowhere")

    def test_repr_shows_file_names(self, archive):
        text = repr(index(archive))

        assert text.startswith(f"DirectoryIndex({str(archive)!r}: 4 files, 0 not used)")
        assert "20240701_18" in text
        assert str(archive / "20240701_18") not in text


class TestCovering:
    def test_brackets_the_window(self, archive):
        # 05:30 to 12:00 needs 05:00 (before the start) through 12:00.
        idx = index(archive)

        assert idx.covering("2024-07-01 05:30", "2024-07-01 12:00") == [
            archive / "20240701_00",
            archive / "20240701_06",
            archive / "20240701_12",
        ]
        assert idx.missing("2024-07-01 05:30", "2024-07-01 12:00") == []

    def test_a_window_on_whole_hours_needs_only_its_own(self, archive):
        idx = index(archive)

        assert idx.covering("2024-07-01 06:00", "2024-07-01 11:00") == [
            archive / "20240701_06"
        ]
        assert idx.covering("2024-07-01 11:00", "2024-07-01 11:00") == [
            archive / "20240701_06"
        ]
        # 11:00 to 11:30 needs 12:00, in the next file.
        assert idx.covering("2024-07-01 11:00", "2024-07-01 11:30") == [
            archive / "20240701_06",
            archive / "20240701_12",
        ]

    def test_missing_hours_are_listed(self, archive):
        (archive / "20240701_06").unlink()
        idx = index(archive)

        hours = pd.date_range("2024-07-01 06:00", periods=6, freq="h")
        assert idx.missing("2024-07-01 05:30", "2024-07-01 12:00") == list(hours)
        with pytest.raises(ValueError) as exc:
            idx.covering("2024-07-01 05:30", "2024-07-01 12:00")
        assert str(exc.value) == (
            f"No file in {archive} holds 2024-07-01 06:00, 2024-07-01 07:00, "
            "2024-07-01 08:00, 2024-07-01 09:00, 2024-07-01 10:00, "
            "2024-07-01 11:00, which 2024-07-01 05:00 to 2024-07-01 12:00 needs."
        )

    def test_hours_past_the_archive_are_missing(self, archive):
        idx = index(archive)

        # The last time is 23:00; 00:00 and 01:00 the next day are needed.
        assert idx.missing("2024-07-01 22:00", "2024-07-02 00:30") == [
            pd.Timestamp("2024-07-02 00:00"),
            pd.Timestamp("2024-07-02 01:00"),
        ]
        # Before the first time, the steps are counted back from it.
        assert idx.missing("2024-06-30 22:30", "2024-07-01 00:00") == [
            pd.Timestamp("2024-06-30 22:00"),
            pd.Timestamp("2024-06-30 23:00"),
        ]

    def test_more_than_ten_missing_hours_are_counted(self, archive):
        idx = index(archive)

        with pytest.raises(ValueError, match=r"2024-07-02 09:00 and 2 more, which"):
            idx.covering("2024-07-02 00:00", "2024-07-02 11:00")

    def test_two_files_with_one_hour_name_both(self, archive):
        shutil.copy(archive / "20240701_06", archive / "20240701_06.bad")
        idx = index(archive)

        # Hours the copy does not hold are found as before.
        assert idx.covering("2024-07-01 00:00", "2024-07-01 05:00") == [
            archive / "20240701_00"
        ]
        # An hour in both is not missing, but the files are ambiguous.
        assert idx.missing("2024-07-01 05:00", "2024-07-01 07:00") == []
        with pytest.raises(ValueError) as exc:
            idx.covering("2024-07-01 05:00", "2024-07-01 07:00")
        assert str(exc.value) == (
            f"2024-07-01 06:00 is in two files, {archive / '20240701_06'} and "
            f"{archive / '20240701_06.bad'}. Move one of them out of {archive}."
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

        idx = index(tmp_path)

        assert idx.step == pd.Timedelta(hours=3)
        assert idx.covering("2024-07-01 07:00", "2024-07-01 13:00") == [
            tmp_path / "a",
            tmp_path / "b",
        ]

    def test_start_after_end_raises(self, archive):
        idx = index(archive)

        with pytest.raises(ValueError, match="is after end"):
            idx.covering("2024-07-01 06:00", "2024-07-01 05:00")

    def test_an_empty_directory_covers_nothing(self, tmp_path):
        idx = index(tmp_path)

        assert len(idx) == 0
        with pytest.raises(ValueError, match="No usable ARL file"):
            idx.missing("2024-07-01 00:00", "2024-07-01 01:00")


class TestCache:
    def test_cache_is_beside_a_writable_directory(self, archive, user_cache):
        idx = index(archive)

        assert idx.cache == archive / ".arlmet-index.csv"
        assert idx.cache.is_file()
        assert not user_cache.exists()

    def test_cache_is_reused_and_refreshed(self, archive, monkeypatch):
        index(archive)
        read = []
        real_read_file = directory_module._read_file
        monkeypatch.setattr(
            directory_module,
            "_read_file",
            lambda path: read.append(path.name) or real_read_file(path),
        )

        # Nothing changed: no file is opened.
        again = index(archive)
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
        refreshed = index(archive)

        assert sorted(read) == ["20240701_12", "20240702_00"]
        assert [p.name for p in refreshed.table["path"]] == [
            "20240701_00",
            "20240701_06",
            "20240701_12",
            "20240702_00",
        ]
        assert refreshed.table["end"].iloc[2] == pd.Timestamp("2024-07-01 14:00")

    def test_an_interrupted_index_keeps_what_it_saved(self, archive, monkeypatch):
        monkeypatch.setattr(directory_module, "_SAVE_EVERY", 2)
        real_read_file = directory_module._read_file

        def stops_at_the_last(path):
            if path.name == "20240701_18":
                raise RuntimeError("interrupted")
            return real_read_file(path)

        monkeypatch.setattr(directory_module, "_read_file", stops_at_the_last)
        with pytest.raises(RuntimeError, match="interrupted"):
            index(archive)

        # The cache was saved after the first two files; the rest are read now.
        read = []
        monkeypatch.setattr(
            directory_module,
            "_read_file",
            lambda path: read.append(path.name) or real_read_file(path),
        )
        idx = index(archive)
        assert sorted(read) == ["20240701_12", "20240701_18"]
        assert len(idx) == 4

    def test_errors_are_cached_too(self, archive, monkeypatch):
        (archive / "README.TXT").write_text("not ARL\n")
        first = index(archive)
        monkeypatch.setattr(directory_module, "_read_file", pytest.fail)

        again = index(archive)

        assert list(again.table["error"]) == list(first.table["error"])

    def test_a_cache_of_another_layout_is_rebuilt(self, archive):
        # The first version keyed rows by file name; its cache is ignored.
        stat = (archive / "20240701_00").stat()
        (archive / ".arlmet-index.csv").write_text(
            "name,size,mtime_ns,source,grid,start,end,step,error\n"
            f"20240701_00,{stat.st_size},{stat.st_mtime_ns},OLD,g,"
            "2000-01-01 00:00:00,2000-01-01 05:00:00,0 days 01:00:00,\n"
        )

        idx = index(archive)

        assert idx.table["start"].iloc[0] == pd.Timestamp("2024-07-01 00:00")
        assert set(idx.table["source"]) == {"TEST"}
        assert idx.cache is not None
        assert idx.cache.read_text().startswith("path,size,mtime_ns,")

    def test_a_recursive_index_has_its_own_cache(self, archive):
        flat = index(archive)
        deep = index(archive, recursive=True)

        assert flat.cache == archive / ".arlmet-index.csv"
        assert deep.cache == archive / ".arlmet-index-recursive.csv"

    def test_read_only_directory_uses_the_user_cache(self, archive, user_cache):
        before = sorted(p.name for p in archive.iterdir())
        archive.chmod(0o555)
        try:
            if os.access(archive, os.W_OK):
                pytest.skip("running as a user who can write anywhere")
            idx = index(archive)
        finally:
            archive.chmod(0o755)

        assert idx.cache is not None
        assert idx.cache.parent == user_cache / "index"
        assert idx.cache.name.startswith("hrrr-") and idx.cache.suffix == ".csv"
        assert idx.cache.is_file()
        assert sorted(p.name for p in archive.iterdir()) == before

    def test_the_user_cache_follows_xdg_without_the_variable(
        self, archive, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("ARLMET_CACHE_DIR")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        monkeypatch.setattr(directory_module.sys, "platform", "linux")
        monkeypatch.setattr(directory_module.os, "access", lambda *args: False)

        idx = index(archive)

        assert idx.cache is not None
        assert idx.cache.parent == tmp_path / "xdg" / "arlmet" / "index"
        assert not (archive / ".arlmet-index.csv").exists()

    def test_cache_dir_argument_wins(self, archive, tmp_path):
        idx = index(archive, cache_dir=tmp_path / "mine")

        assert idx.cache is not None
        assert idx.cache.parent == tmp_path / "mine"
        assert idx.cache.is_file()
        assert not (archive / ".arlmet-index.csv").exists()

    def test_two_directories_of_one_name_get_two_caches(self, tmp_path):
        a = tmp_path / "a" / "hrrr"
        b = tmp_path / "b" / "hrrr"
        a.mkdir(parents=True)
        b.mkdir(parents=True)

        first = index(a, cache_dir=tmp_path / "caches")
        second = index(b, cache_dir=tmp_path / "caches")

        assert first.cache != second.cache

    def test_an_unwritable_cache_raises_a_clear_error(self, archive, tmp_path):
        blocker = tmp_path / "a-file"
        blocker.write_text("")

        with pytest.raises(
            OSError, match="Cannot write the index cache.*ARLMET_CACHE_DIR"
        ):
            index(archive, cache_dir=blocker)


class TestLinksAndFolders:
    def test_a_link_and_its_target_are_one_row(self, archive):
        os.symlink(archive / "20240701_06", archive / "zz-link")

        idx = index(archive)

        assert len(idx) == 4
        assert list(idx.table["path"]).count(archive / "20240701_06") == 1
        assert idx.covering("2024-07-01 06:00", "2024-07-01 07:00") == [
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

        flat = index(top)
        deep = index(top, recursive=True)

        expected = [month / "20240701_00", month / "20240701_06"]
        assert list(flat.table["path"]) == expected
        assert list(deep.table["path"]) == expected

    def test_a_link_to_a_link_is_followed(self, tmp_path, archive):
        other = tmp_path / "other"
        other.mkdir()
        os.symlink(archive / "20240701_12", other / "first")
        os.symlink(other / "first", other / "second")

        idx = index(other)

        assert list(idx.table["path"]) == [archive / "20240701_12"]
        assert idx.table["error"].iloc[0] == ""

    def test_a_broken_link_is_a_row_with_its_error(self, archive, tmp_path):
        os.symlink(tmp_path / "gone", archive / "broken")

        idx = index(archive)
        again = index(archive)  # the cache does not hide it

        for table in (idx.table, again.table):
            row = table.iloc[-1]
            assert row["path"] == archive / "broken"
            assert row["error"].startswith("Cannot follow the link: ")
            assert pd.isna(row["start"])
        assert len(idx.covering("2024-07-01 00:00", "2024-07-01 23:00")) == 4

    def test_a_link_loop_is_a_row_with_its_error(self, archive):
        os.symlink(archive / "loop-b", archive / "loop-a")
        os.symlink(archive / "loop-a", archive / "loop-b")
        # A link from a subfolder back to the top is not followed.
        (archive / "sub").mkdir()
        os.symlink(archive, archive / "sub" / "up")

        idx = index(archive, recursive=True)

        errors = idx.table.set_index(idx.table["path"].map(lambda p: p.name))["error"]
        assert errors["loop-a"].startswith("Cannot follow the link: ")
        assert errors["loop-b"].startswith("Cannot follow the link: ")
        assert "up" not in errors
        assert len(idx) == 6

    def test_recursive_reads_a_yyyy_mm_tree(self, tmp_path):
        top = tmp_path.resolve() / "hrrr"
        june, july = top / "2024" / "06", top / "2024" / "07"
        june.mkdir(parents=True)
        july.mkdir(parents=True)
        six_hour_blocks(june, "2024-06-30 18:00", 1)
        six_hour_blocks(july, "2024-07-01 00:00", 2)

        assert len(index(top)) == 0
        idx = index(top, recursive=True)

        assert len(idx) == 3
        assert idx.covering("2024-06-30 23:30", "2024-07-01 06:00") == [
            june / "20240630_18",
            july / "20240701_00",
            july / "20240701_06",
        ]

    def test_recursive_reads_superseded_copies_too(self, archive):
        # A subfolder of superseded copies is read, and its files overlap.
        (archive / "deprecated").mkdir()
        shutil.copy(archive / "20240701_06", archive / "deprecated" / "20240701_06")

        assert len(index(archive).covering("2024-07-01 06:00", "2024-07-01 07:00")) == 1
        idx = index(archive, recursive=True)
        with pytest.raises(ValueError, match="is in two files"):
            idx.covering("2024-07-01 06:00", "2024-07-01 07:00")

    def test_hidden_folders_are_not_read(self, archive):
        (archive / ".snapshot").mkdir()
        shutil.copy(archive / "20240701_06", archive / ".snapshot" / "20240701_06")

        assert len(index(archive, recursive=True)) == 4


class TestNames:
    def test_index_is_the_function_and_the_record_module_moved(self):
        import importlib

        from arlmet.indexrecord import IndexRecord, LvlInfo, VarInfo

        assert callable(arlmet.index)
        assert arlmet.index is directory_module.index
        assert arlmet.IndexRecord is IndexRecord
        assert LvlInfo.__module__ == VarInfo.__module__ == "arlmet.indexrecord"
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("arlmet.index")
