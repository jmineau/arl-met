"""Unit tests for arlmet.sources — no network access required."""

import io
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from arlmet.sources import (
    GDAS0p5Source,
    GDASSource,
    GFSSource,
    HRRRSource,
    HRRRv1Source,
    NAMSource,
    NAMSSource,
    NARRSource,
    ReanalysisSource,
)

# ---------------------------------------------------------------------------
# HRRRSource
# ---------------------------------------------------------------------------


class TestHRRRSource:
    def setup_method(self):
        self.src = HRRRSource()

    @pytest.mark.parametrize(
        "hour, expected",
        [
            (0, "20240718_00-05_hrrr"),
            (3, "20240718_00-05_hrrr"),
            (5, "20240718_00-05_hrrr"),
            (6, "20240718_06-11_hrrr"),
            (11, "20240718_06-11_hrrr"),
            (12, "20240718_12-17_hrrr"),
            (17, "20240718_12-17_hrrr"),
            (18, "20240718_18-23_hrrr"),
            (23, "20240718_18-23_hrrr"),
        ],
    )
    def test_filename_blocks(self, hour, expected):
        t = pd.Timestamp(f"2024-07-18 {hour:02d}:00")
        assert self.src._filename(t) == expected

    def test_s3_key_uses_year_month_subdir(self):
        t = pd.Timestamp("2024-07-18 03:00")
        assert self.src._s3_key(t) == "hrrr/2024/07/20240718_00-05_hrrr"

    def test_keys_for_range_single_block(self):
        keys = self.src.keys_for_range("2024-07-18 01:00", "2024-07-18 04:00")
        assert keys == ["hrrr/2024/07/20240718_00-05_hrrr"]

    def test_keys_for_range_two_blocks(self):
        keys = self.src.keys_for_range("2024-07-18 04:00", "2024-07-18 08:00")
        assert keys == [
            "hrrr/2024/07/20240718_00-05_hrrr",
            "hrrr/2024/07/20240718_06-11_hrrr",
        ]

    def test_keys_for_range_all_four_blocks(self):
        keys = self.src.keys_for_range("2024-07-18 00:00", "2024-07-18 23:00")
        assert keys == [
            "hrrr/2024/07/20240718_00-05_hrrr",
            "hrrr/2024/07/20240718_06-11_hrrr",
            "hrrr/2024/07/20240718_12-17_hrrr",
            "hrrr/2024/07/20240718_18-23_hrrr",
        ]

    def test_keys_for_range_day_boundary(self):
        keys = self.src.keys_for_range("2024-07-18 22:00", "2024-07-19 02:00")
        assert keys == [
            "hrrr/2024/07/20240718_18-23_hrrr",
            "hrrr/2024/07/20240719_00-05_hrrr",
        ]

    def test_keys_for_range_backward_trajectory(self):
        # start > end: backward STILT trajectory
        keys = self.src.keys_for_range("2024-07-18 08:00", "2024-07-18 04:00")
        assert keys == [
            "hrrr/2024/07/20240718_00-05_hrrr",
            "hrrr/2024/07/20240718_06-11_hrrr",
        ]

    def test_keys_for_range_no_duplicates(self):
        # All times within the same block → single key
        keys = self.src.keys_for_range("2024-07-18 06:00", "2024-07-18 11:00")
        assert keys == ["hrrr/2024/07/20240718_06-11_hrrr"]

    def test_repr(self):
        assert repr(self.src) == "HRRRSource()"


# ---------------------------------------------------------------------------
# NAMSource
# ---------------------------------------------------------------------------


class TestNAMSource:
    def setup_method(self):
        self.src = NAMSource()

    def test_filename(self):
        assert self.src._filename(pd.Timestamp("2024-07-18")) == "20240718_nam12"

    def test_s3_key(self):
        assert (
            self.src._s3_key(pd.Timestamp("2024-07-18"))
            == "nam12/2024/07/20240718_nam12"
        )

    def test_keys_for_range_single_day(self):
        keys = self.src.keys_for_range("2024-07-18 06:00", "2024-07-18 18:00")
        assert keys == ["nam12/2024/07/20240718_nam12"]

    def test_keys_for_range_two_days(self):
        keys = self.src.keys_for_range("2024-07-18 22:00", "2024-07-19 06:00")
        assert keys == [
            "nam12/2024/07/20240718_nam12",
            "nam12/2024/07/20240719_nam12",
        ]

    def test_keys_for_range_backward(self):
        keys = self.src.keys_for_range("2024-07-19 06:00", "2024-07-18 22:00")
        assert keys == [
            "nam12/2024/07/20240718_nam12",
            "nam12/2024/07/20240719_nam12",
        ]


# ---------------------------------------------------------------------------
# GDASSource
# ---------------------------------------------------------------------------


class TestGDASSource:
    def setup_method(self):
        self.src = GDASSource()

    @pytest.mark.parametrize(
        "date_str, expected_week",
        [
            ("2025-09-01", 1),
            ("2025-09-07", 1),
            ("2025-09-08", 2),
            ("2025-09-14", 2),
            ("2025-09-15", 3),
            ("2025-09-21", 3),
            ("2025-09-22", 4),
            ("2025-09-28", 4),
            ("2025-09-29", 5),
            ("2025-09-30", 5),
            ("2025-01-31", 5),  # 31-day month
        ],
    )
    def test_week_boundaries(self, date_str, expected_week):
        assert self.src._week(pd.Timestamp(date_str)) == expected_week

    @pytest.mark.parametrize(
        "date_str, expected",
        [
            ("2025-09-01", "gdas1.sep25.w1"),
            ("2025-09-07", "gdas1.sep25.w1"),
            ("2025-09-08", "gdas1.sep25.w2"),
            ("2025-09-30", "gdas1.sep25.w5"),
            ("2024-01-15", "gdas1.jan24.w3"),
            ("2024-12-31", "gdas1.dec24.w5"),
        ],
    )
    def test_filename(self, date_str, expected):
        assert self.src._filename(pd.Timestamp(date_str)) == expected

    def test_s3_key(self):
        assert (
            self.src._s3_key(pd.Timestamp("2025-09-01")) == "gdas1/2025/gdas1.sep25.w1"
        )

    def test_keys_for_range_within_one_week(self):
        keys = self.src.keys_for_range("2025-09-03", "2025-09-05")
        assert keys == ["gdas1/2025/gdas1.sep25.w1"]

    def test_keys_for_range_week_boundary(self):
        keys = self.src.keys_for_range("2025-09-06", "2025-09-09")
        assert keys == [
            "gdas1/2025/gdas1.sep25.w1",
            "gdas1/2025/gdas1.sep25.w2",
        ]

    def test_keys_for_range_month_boundary(self):
        keys = self.src.keys_for_range("2025-09-29", "2025-10-03")
        assert keys == [
            "gdas1/2025/gdas1.sep25.w5",
            "gdas1/2025/gdas1.oct25.w1",
        ]

    def test_keys_for_range_year_boundary(self):
        keys = self.src.keys_for_range("2024-12-30", "2025-01-02")
        assert keys == [
            "gdas1/2024/gdas1.dec24.w5",
            "gdas1/2025/gdas1.jan25.w1",
        ]

    def test_keys_for_range_backward(self):
        keys = self.src.keys_for_range("2025-09-09", "2025-09-06")
        assert keys == [
            "gdas1/2025/gdas1.sep25.w1",
            "gdas1/2025/gdas1.sep25.w2",
        ]


# ---------------------------------------------------------------------------
# GFSSource
# ---------------------------------------------------------------------------


class TestGFSSource:
    def setup_method(self):
        self.src = GFSSource()

    def test_filename(self):
        assert self.src._filename(pd.Timestamp("2024-07-18")) == "20240718_gfs0p25"

    def test_s3_key(self):
        assert (
            self.src._s3_key(pd.Timestamp("2024-07-18"))
            == "gfs0p25/2024/07/20240718_gfs0p25"
        )

    def test_keys_for_range_single_day(self):
        keys = self.src.keys_for_range("2024-07-18 06:00", "2024-07-18 18:00")
        assert keys == ["gfs0p25/2024/07/20240718_gfs0p25"]

    def test_keys_for_range_two_days(self):
        keys = self.src.keys_for_range("2024-07-18 20:00", "2024-07-19 06:00")
        assert keys == [
            "gfs0p25/2024/07/20240718_gfs0p25",
            "gfs0p25/2024/07/20240719_gfs0p25",
        ]


# ---------------------------------------------------------------------------
# NAMSSource
# ---------------------------------------------------------------------------


class TestNAMSSource:
    def setup_method(self):
        self.src = NAMSSource()

    def test_filename_conus(self):
        assert (
            self.src._filename(pd.Timestamp("2024-07-18"))
            == "20240718_hysplit.t00z.namsa"
        )

    def test_filename_ak(self):
        src = NAMSSource(domain="ak")
        assert (
            src._filename(pd.Timestamp("2024-07-18"))
            == "20240718_hysplit.t00z.namsa.AK"
        )

    def test_filename_hi(self):
        src = NAMSSource(domain="hi")
        assert (
            src._filename(pd.Timestamp("2024-07-18"))
            == "20240718_hysplit.t00z.namsa.HI"
        )

    def test_invalid_domain_raises(self):
        with pytest.raises(ValueError, match="domain"):
            NAMSSource(domain="eu")

    def test_s3_key(self):
        assert (
            self.src._s3_key(pd.Timestamp("2024-07-18"))
            == "nams/2024/07/20240718_hysplit.t00z.namsa"
        )

    def test_keys_for_range_single_day(self):
        keys = self.src.keys_for_range("2024-07-18 06:00", "2024-07-18 18:00")
        assert keys == ["nams/2024/07/20240718_hysplit.t00z.namsa"]

    def test_keys_for_range_two_days(self):
        keys = self.src.keys_for_range("2024-07-18 22:00", "2024-07-19 06:00")
        assert keys == [
            "nams/2024/07/20240718_hysplit.t00z.namsa",
            "nams/2024/07/20240719_hysplit.t00z.namsa",
        ]

    def test_keys_for_range_backward(self):
        keys = self.src.keys_for_range("2024-07-19 06:00", "2024-07-18 22:00")
        assert keys == [
            "nams/2024/07/20240718_hysplit.t00z.namsa",
            "nams/2024/07/20240719_hysplit.t00z.namsa",
        ]

    def test_repr(self):
        assert repr(self.src) == "NAMSSource(domain='conus')"


# ---------------------------------------------------------------------------
# ReanalysisSource
# ---------------------------------------------------------------------------


class TestReanalysisSource:
    def setup_method(self):
        self.src = ReanalysisSource()

    @pytest.mark.parametrize(
        "date_str, expected",
        [
            ("2024-07-01", "RP202407.gbl"),
            ("2024-07-18", "RP202407.gbl"),
            ("2024-07-31", "RP202407.gbl"),
            ("1948-01-01", "RP194801.gbl"),
            ("2000-12-31", "RP200012.gbl"),
        ],
    )
    def test_filename(self, date_str, expected):
        assert self.src._filename(pd.Timestamp(date_str)) == expected

    def test_s3_key(self):
        assert (
            self.src._s3_key(pd.Timestamp("2024-07-18"))
            == "reanalysis/2024/RP202407.gbl"
        )

    def test_keys_for_range_within_one_month(self):
        keys = self.src.keys_for_range("2024-07-05", "2024-07-20")
        assert keys == ["reanalysis/2024/RP202407.gbl"]

    def test_keys_for_range_month_boundary(self):
        keys = self.src.keys_for_range("2024-07-28", "2024-08-03")
        assert keys == [
            "reanalysis/2024/RP202407.gbl",
            "reanalysis/2024/RP202408.gbl",
        ]

    def test_keys_for_range_year_boundary(self):
        keys = self.src.keys_for_range("2023-12-20", "2024-01-10")
        assert keys == [
            "reanalysis/2023/RP202312.gbl",
            "reanalysis/2024/RP202401.gbl",
        ]

    def test_keys_for_range_backward(self):
        keys = self.src.keys_for_range("2024-08-03", "2024-07-28")
        assert keys == [
            "reanalysis/2024/RP202407.gbl",
            "reanalysis/2024/RP202408.gbl",
        ]

    def test_repr(self):
        assert repr(self.src) == "ReanalysisSource()"


# ---------------------------------------------------------------------------
# HRRRv1Source
# ---------------------------------------------------------------------------


class TestHRRRv1Source:
    def setup_method(self):
        self.src = HRRRv1Source()

    @pytest.mark.parametrize(
        "hour, expected",
        [
            (0, "hysplit.20170601.00z.hrrra"),
            (5, "hysplit.20170601.00z.hrrra"),
            (6, "hysplit.20170601.06z.hrrra"),
            (11, "hysplit.20170601.06z.hrrra"),
            (12, "hysplit.20170601.12z.hrrra"),
            (18, "hysplit.20170601.18z.hrrra"),
            (23, "hysplit.20170601.18z.hrrra"),
        ],
    )
    def test_filename_blocks(self, hour, expected):
        t = pd.Timestamp(f"2017-06-01 {hour:02d}:00")
        assert self.src._filename(t) == expected

    def test_s3_key(self):
        t = pd.Timestamp("2017-06-01 06:00")
        assert self.src._s3_key(t) == "hrrr.v1/2017/06/hysplit.20170601.06z.hrrra"

    def test_keys_for_range_single_block(self):
        keys = self.src.keys_for_range("2017-06-01 01:00", "2017-06-01 04:00")
        assert keys == ["hrrr.v1/2017/06/hysplit.20170601.00z.hrrra"]

    def test_keys_for_range_two_blocks(self):
        keys = self.src.keys_for_range("2017-06-01 04:00", "2017-06-01 08:00")
        assert keys == [
            "hrrr.v1/2017/06/hysplit.20170601.00z.hrrra",
            "hrrr.v1/2017/06/hysplit.20170601.06z.hrrra",
        ]

    def test_repr(self):
        assert repr(self.src) == "HRRRv1Source()"


# ---------------------------------------------------------------------------
# GDAS0p5Source
# ---------------------------------------------------------------------------


class TestGDAS0p5Source:
    def setup_method(self):
        self.src = GDAS0p5Source()

    def test_filename(self):
        assert self.src._filename(pd.Timestamp("2024-07-18")) == "20240718_gdas0p5"

    def test_s3_key(self):
        assert (
            self.src._s3_key(pd.Timestamp("2024-07-18"))
            == "gdas0p5/2024/07/20240718_gdas0p5"
        )

    def test_keys_for_range_single_day(self):
        keys = self.src.keys_for_range("2024-07-18 06:00", "2024-07-18 18:00")
        assert keys == ["gdas0p5/2024/07/20240718_gdas0p5"]

    def test_keys_for_range_two_days(self):
        keys = self.src.keys_for_range("2024-07-18 20:00", "2024-07-19 06:00")
        assert keys == [
            "gdas0p5/2024/07/20240718_gdas0p5",
            "gdas0p5/2024/07/20240719_gdas0p5",
        ]

    def test_repr(self):
        assert repr(self.src) == "GDAS0p5Source()"


# ---------------------------------------------------------------------------
# NARRSource
# ---------------------------------------------------------------------------


class TestNARRSource:
    def setup_method(self):
        self.src = NARRSource()

    @pytest.mark.parametrize(
        "date_str, expected",
        [
            ("2024-07-01", "NARR202407"),
            ("2024-07-18", "NARR202407"),
            ("2024-07-31", "NARR202407"),
            ("1979-01-01", "NARR197901"),
            ("2000-12-31", "NARR200012"),
        ],
    )
    def test_filename(self, date_str, expected):
        assert self.src._filename(pd.Timestamp(date_str)) == expected

    def test_s3_key(self):
        assert self.src._s3_key(pd.Timestamp("2024-07-18")) == "narr/2024/NARR202407"

    def test_keys_for_range_within_one_month(self):
        keys = self.src.keys_for_range("2024-07-05", "2024-07-20")
        assert keys == ["narr/2024/NARR202407"]

    def test_keys_for_range_month_boundary(self):
        keys = self.src.keys_for_range("2024-07-28", "2024-08-03")
        assert keys == [
            "narr/2024/NARR202407",
            "narr/2024/NARR202408",
        ]

    def test_keys_for_range_year_boundary(self):
        keys = self.src.keys_for_range("2023-12-20", "2024-01-10")
        assert keys == [
            "narr/2023/NARR202312",
            "narr/2024/NARR202401",
        ]

    def test_repr(self):
        assert repr(self.src) == "NARRSource()"


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


class TestUrls:
    def setup_method(self):
        self.src = HRRRSource()

    def test_s3_url(self):
        key = "hrrr/2024/20240718_00-05_hrrr"
        assert self.src._url(key, "s3") == (
            "s3://noaa-oar-arl-hysplit-pds/hrrr/2024/20240718_00-05_hrrr"
        )

    def test_ftp_url(self):
        key = "hrrr/2024/20240718_00-05_hrrr"
        assert self.src._url(key, "ftp") == (
            "ftp://anonymous@ftp.arl.noaa.gov/archives/hrrr/2024/20240718_00-05_hrrr"
        )

    def test_http_url(self):
        key = "hrrr/2024/20240718_00-05_hrrr"
        assert self.src._url(key, "http") == (
            "https://www.ready.noaa.gov/data/archives/hrrr/2024/20240718_00-05_hrrr"
        )

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            self.src._url("hrrr/2024/foo", "gcs")

    def test_s3_storage_options_anonymous(self):
        assert self.src._storage_options("s3") == {"anon": True}

    def test_non_s3_storage_options_empty(self):
        assert self.src._storage_options("ftp") == {}
        assert self.src._storage_options("http") == {}


class TestFetchHelpers:
    def setup_method(self):
        self.src = HRRRSource()

    def test_dest_path_adds_crop_tag(self, tmp_path):
        plain = self.src._dest_path(tmp_path, "file.arl", None)
        cropped = self.src._dest_path(
            tmp_path, "file.arl", (-111.5, 40.5, -110.0, 41.0)
        )

        assert plain == tmp_path / "file.arl"
        assert cropped == tmp_path / "file.arl.crop_-111.50_40.50_-110.00_41.00"

    def test_dest_path_adds_levels_tag(self, tmp_path):
        bbox = (-111.5, 40.5, -110.0, 41.0)
        both = self.src._dest_path(tmp_path, "file.arl", bbox, list(range(20)))
        levels_only = self.src._dest_path(tmp_path, "file.arl", None, [5, 0, 1, 2])

        assert both.name == "file.arl.crop_-111.50_40.50_-110.00_41.00.levels_0-19"
        assert levels_only.name == "file.arl.levels_0-2_5"

    def test_dest_path_two_decimal_bbox_keeps_legacy_tag(self, tmp_path):
        # Caches written before finer bboxes were supported (e.g. by PYSTILT)
        # must keep matching, including float noise from bbox arithmetic.
        dest = self.src._dest_path(
            tmp_path, "file", (-112.0, 40.25, -111.1 - 0.2, 40.1 - 0.05)
        )
        assert dest.name == "file.crop_-112.00_40.25_-111.30_40.05"

    def test_dest_path_bboxes_differing_past_two_decimals_do_not_collide(
        self, tmp_path
    ):
        a = self.src._dest_path(tmp_path, "file", (-111.925, 40.5, -110.0, 41.0))
        b = self.src._dest_path(tmp_path, "file", (-111.924, 40.5, -110.0, 41.0))
        c = self.src._dest_path(tmp_path, "file", (-111.92, 40.5, -110.0, 41.0))

        assert len({a, b, c}) == 3
        assert a.name == "file.crop_-111.925_40.50_-110.00_41.00"
        assert b.name == "file.crop_-111.924_40.50_-110.00_41.00"
        assert c.name == "file.crop_-111.92_40.50_-110.00_41.00"

    def test_download_copies_bytes_and_renames_tmp_file(self, tmp_path, monkeypatch):
        class FakeOpen:
            def __init__(self, data: bytes):
                self.data = data

            def __call__(self, url, mode, **opts):
                assert url == "s3://bucket/test"
                assert mode == "rb"
                assert opts == {"anon": True}
                return io.BytesIO(self.data)

        monkeypatch.setitem(
            sys.modules, "fsspec", types.SimpleNamespace(open=FakeOpen(b"arl-bytes"))
        )

        dest = tmp_path / "download.arl"
        self.src._download("s3://bucket/test", dest, {"anon": True})

        assert dest.read_bytes() == b"arl-bytes"
        # The hidden temp file was renamed onto dest.
        assert list(tmp_path.iterdir()) == [dest]

    def test_download_cleans_up_tmp_file_on_failure(self, tmp_path, monkeypatch):
        monkeypatch.setitem(
            sys.modules,
            "fsspec",
            types.SimpleNamespace(open=lambda url, mode, **opts: io.BytesIO(b"bytes")),
        )

        def boom(src, dst, length):
            raise RuntimeError("copy failed")

        monkeypatch.setattr("arlmet.sources.shutil.copyfileobj", boom)
        dest = tmp_path / "broken.arl"

        with pytest.raises(RuntimeError, match="copy failed"):
            self.src._download("s3://bucket/test", dest, {})

        assert not dest.exists()
        assert list(tmp_path.iterdir()) == []

    def test_download_writes_to_hidden_partial_file(self, tmp_path, monkeypatch):
        monkeypatch.setitem(
            sys.modules,
            "fsspec",
            types.SimpleNamespace(open=lambda url, mode, **opts: io.BytesIO(b"bytes")),
        )
        written = []

        def record(src, dst, length):
            written.append(Path(dst.name))
            dst.write(src.read())

        monkeypatch.setattr("arlmet.sources.shutil.copyfileobj", record)
        dest = tmp_path / "file.arl"
        self.src._download("s3://bucket/test", dest, {})

        # Never written under the final name, so an interrupted download
        # can't be mistaken for a cached file.
        assert written[0].parent == tmp_path
        assert written[0].name.startswith(".file.arl.")
        assert written[0].name.endswith(".partial")
        assert dest.read_bytes() == b"bytes"

    def test_fetch_and_crop_cleans_up_temp_input(self, tmp_path, monkeypatch):
        downloaded = []
        cropped = []

        def fake_download(url, dest, opts):
            downloaded.append((url, dest, opts))
            Path(dest).write_bytes(b"raw")

        def fake_extract_subset(path, dest, bbox, levels):
            cropped.append((Path(path), Path(dest), bbox, levels))
            Path(dest).write_bytes(Path(path).read_bytes() + b"-cropped")
            # extract_subset returns the output path.
            return Path(dest)

        monkeypatch.setattr(self.src, "_download", fake_download)
        monkeypatch.setitem(
            sys.modules,
            "arlmet.ops.subset",
            types.SimpleNamespace(extract_subset=fake_extract_subset),
        )

        dest = tmp_path / "cropped.arl"
        self.src._fetch_and_crop(
            "s3://bucket/test",
            dest,
            {},
            bbox=(-112.0, 40.0, -111.0, 41.0),
            levels=[0, 1, 2],
        )

        assert downloaded[0][0] == "s3://bucket/test"
        assert cropped[0][2:] == ((-112.0, 40.0, -111.0, 41.0), [0, 1, 2])
        assert dest.read_bytes() == b"raw-cropped"
        # The raw download is staged as a hidden file in the destination
        # directory, not the system temp dir, and cleaned up. The crop goes
        # straight to dest: extract_subset itself writes atomically.
        raw, out = cropped[0][0], cropped[0][1]
        assert raw.parent == tmp_path
        assert raw.name.startswith(".cropped.arl.")
        assert raw.name.endswith(".partial")
        assert out == dest
        assert list(tmp_path.iterdir()) == [dest]

    @pytest.mark.parametrize("fail_in", ["download", "crop"])
    def test_fetch_and_crop_cleans_up_on_failure(self, tmp_path, monkeypatch, fail_in):
        def fake_download(url, dest, opts):
            Path(dest).write_bytes(b"raw")
            if fail_in == "download":
                raise RuntimeError("download failed")

        def fake_extract_subset(path, dest, bbox, levels):
            # The real extract_subset leaves nothing at dst when it fails
            # (tested in test_subset.py).
            raise RuntimeError("crop failed")

        monkeypatch.setattr(self.src, "_download", fake_download)
        monkeypatch.setitem(
            sys.modules,
            "arlmet.ops.subset",
            types.SimpleNamespace(extract_subset=fake_extract_subset),
        )

        dest = tmp_path / "cropped.arl"
        with pytest.raises(RuntimeError, match=f"{fail_in} failed"):
            self.src._fetch_and_crop(
                "s3://bucket/test",
                dest,
                {},
                bbox=(-112.0, 40.0, -111.0, 41.0),
                levels=None,
            )

        # No truncated file under the cached name, and no leftover temp files.
        assert list(tmp_path.iterdir()) == []

    def test_fetch_uses_cache_and_dispatches_crop_download_and_overwrite(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setitem(
            sys.modules, "fsspec", types.SimpleNamespace(open=lambda *a, **k: None)
        )
        monkeypatch.setattr(
            self.src,
            "keys_for_range",
            lambda start, end: ["hrrr/2024/07/a", "hrrr/2024/07/b"],
        )

        downloads = []
        crops = []

        def fake_download(url, dest, opts):
            downloads.append((url, Path(dest), opts))
            Path(dest).write_text("downloaded")

        def fake_crop(url, dest, opts, *, bbox, levels):
            crops.append((url, Path(dest), bbox, levels, opts))
            Path(dest).write_text("cropped")

        monkeypatch.setattr(self.src, "_download", fake_download)
        monkeypatch.setattr(self.src, "_fetch_and_crop", fake_crop)

        cached = tmp_path / "a"
        cached.write_text("cached")

        results = self.src.fetch("2024-07-18", "2024-07-19", local_dir=tmp_path)
        assert results == [cached, tmp_path / "b"]
        assert downloads == [
            (
                "s3://noaa-oar-arl-hysplit-pds/hrrr/2024/07/b",
                tmp_path / "b",
                {"anon": True},
            )
        ]

        downloads.clear()
        results = self.src.fetch(
            "2024-07-18",
            "2024-07-19",
            local_dir=tmp_path,
            bbox=(-112.0, 40.0, -111.0, 41.0),
            overwrite=True,
        )
        assert len(results) == 2
        assert downloads == []
        assert [entry[1].name for entry in crops] == [
            "a.crop_-112.00_40.00_-111.00_41.00",
            "b.crop_-112.00_40.00_-111.00_41.00",
        ]

        # Levels alone also crop, and are passed sorted and unique.
        crops.clear()
        results = self.src.fetch(
            "2024-07-18", "2024-07-19", local_dir=tmp_path, levels=range(2, -1, -1)
        )
        assert downloads == []
        assert [(entry[1].name, entry[2], entry[3]) for entry in crops] == [
            ("a.levels_0-2", None, [0, 1, 2]),
            ("b.levels_0-2", None, [0, 1, 2]),
        ]
        assert results == [tmp_path / "a.levels_0-2", tmp_path / "b.levels_0-2"]


# ---------------------------------------------------------------------------
# start_date validation
# ---------------------------------------------------------------------------


_ALL_SOURCES = [
    HRRRSource(),
    HRRRv1Source(),
    NAMSource(),
    NAMSSource(),
    GDASSource(),
    GDAS0p5Source(),
    GFSSource(),
    NARRSource(),
    ReanalysisSource(),
]


class TestStartDate:
    @pytest.mark.parametrize("src", _ALL_SOURCES, ids=repr)
    def test_range_starting_at_start_date_is_allowed(self, src):
        assert src.keys_for_range(src.start_date, src.start_date) == [
            src._s3_key(src.start_date)
        ]

    @pytest.mark.parametrize("src", _ALL_SOURCES, ids=repr)
    def test_range_before_start_date_raises(self, src):
        before = src.start_date - pd.Timedelta(hours=1)
        with pytest.raises(ValueError, match="archive begins"):
            src.keys_for_range(before, src.start_date + pd.Timedelta(days=1))

    def test_backward_range_before_start_date_raises(self):
        # start > end (backward trajectory): the earlier end is what matters.
        src = HRRRSource()
        with pytest.raises(ValueError, match="HRRRSource archive begins 2019-06-12"):
            src.keys_for_range("2019-06-12 06:00", "2019-06-11 18:00")

    def test_tz_aware_range_is_compared_in_utc(self):
        src = HRRRSource()
        keys = src.keys_for_range("2019-06-12 00:00+00:00", "2019-06-12 03:00+00:00")
        assert keys == ["hrrr/2019/06/20190612_00-05_hrrr"]
        with pytest.raises(ValueError, match="archive begins"):
            src.keys_for_range("2019-06-11 23:00+00:00", "2019-06-12 03:00+00:00")

    def test_fetch_raises_before_downloading(self, tmp_path, monkeypatch):
        src = HRRRSource()
        monkeypatch.setitem(
            sys.modules, "fsspec", types.SimpleNamespace(open=lambda *a, **k: None)
        )

        def no_download(*args, **kwargs):
            raise AssertionError("should not download")

        monkeypatch.setattr(src, "_download", no_download)
        with pytest.raises(ValueError, match="archive begins"):
            src.fetch("2019-06-01", "2019-06-13", local_dir=tmp_path)


# ---------------------------------------------------------------------------
# fetch() import guard
# ---------------------------------------------------------------------------


def test_fetch_raises_import_error_without_fsspec(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "fsspec":
            raise ImportError("no fsspec")
        return real_import(name, *args, **kwargs)

    src = HRRRSource()
    import tempfile
    from pathlib import Path

    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="fsspec"):
            src.fetch("2024-07-18", "2024-07-18", local_dir=Path(tempfile.mkdtemp()))


# ---------------------------------------------------------------------------
# Integration tests — require live S3 access (skipped by default)
# Run with: uv run pytest -m network
# ---------------------------------------------------------------------------

s3fs = pytest.importorskip("s3fs", reason="s3fs not installed")

# A known-small file for existence checks: GDAS week 1 of a past month.
# Chosen because GDAS weekly files are the smallest in the archive (~571 MB).
_GDAS_TEST_TIME = "2024-07-01"
_HRRR_TEST_TIME = "2024-07-18 06:00"  # block 06-11


@pytest.mark.network
class TestS3Existence:
    """Verify that S3 keys resolve to real objects without downloading."""

    def _exists(self, key: str) -> bool:
        import s3fs

        fs = s3fs.S3FileSystem(anon=True)
        return fs.exists(f"noaa-oar-arl-hysplit-pds/{key}")

    def test_gdas_key_exists(self):
        src = GDASSource()
        key = src._s3_key(pd.Timestamp(_GDAS_TEST_TIME))
        assert self._exists(key), f"Expected S3 key not found: {key}"

    def test_hrrr_key_exists(self):
        src = HRRRSource()
        key = src._s3_key(pd.Timestamp(_HRRR_TEST_TIME))
        assert self._exists(key), f"Expected S3 key not found: {key}"

    def test_nam_key_exists(self):
        src = NAMSource()
        key = src._s3_key(pd.Timestamp(_GDAS_TEST_TIME))
        assert self._exists(key), f"Expected S3 key not found: {key}"

    def test_gfs_key_exists(self):
        src = GFSSource()
        key = src._s3_key(pd.Timestamp(_GDAS_TEST_TIME))
        assert self._exists(key), f"Expected S3 key not found: {key}"

    def test_nams_key_exists(self):
        src = NAMSSource()
        key = src._s3_key(pd.Timestamp(_GDAS_TEST_TIME))
        assert self._exists(key), f"Expected S3 key not found: {key}"

    def test_reanalysis_key_exists(self):
        src = ReanalysisSource()
        key = src._s3_key(pd.Timestamp(_GDAS_TEST_TIME))
        assert self._exists(key), f"Expected S3 key not found: {key}"

    def test_hrrr_v1_key_exists(self):
        src = HRRRv1Source()
        key = src._s3_key(pd.Timestamp("2017-06-01 06:00"))
        assert self._exists(key), f"Expected S3 key not found: {key}"

    def test_gdas0p5_key_exists(self):
        # Archive ends mid-2019; use a date well within range
        src = GDAS0p5Source()
        key = src._s3_key(pd.Timestamp("2018-07-01"))
        assert self._exists(key), f"Expected S3 key not found: {key}"

    def test_narr_key_exists(self):
        # Archive ends 2019; use a date well within range
        src = NARRSource()
        key = src._s3_key(pd.Timestamp("2018-07-01"))
        assert self._exists(key), f"Expected S3 key not found: {key}"


@pytest.mark.network
def test_gdas_header_is_valid_arl(tmp_path):
    """Download the first 50 bytes of a GDAS file and check the ARL header."""
    import s3fs

    from arlmet.header import Header

    src = GDASSource()
    key = src._s3_key(pd.Timestamp(_GDAS_TEST_TIME))
    fs = s3fs.S3FileSystem(anon=True)
    with fs.open(f"noaa-oar-arl-hysplit-pds/{key}", "rb") as f:
        raw = f.read(Header.N_BYTES)

    assert len(raw) == Header.N_BYTES
    header = Header.from_bytes(raw)
    assert header.variable == "INDX"


# ---------------------------------------------------------------------------
# Fetch-and-open integration tests — require live S3 access and disk space.
# Each case downloads a full source file then crops to the given bbox.
#
# Bbox choice:
#   _SLV_BBOX    : high-res regional products (3–12 km) — SLV domain gives
#                  hundreds of grid cells, easily fitting any index record.
#   _WEST_NA_BBOX: coarse-resolution products (≥32 km or global) — the SLV
#                  domain yields too few cells for their larger index records.
#   None         : skip cropping for very coarse products whose monthly index
#                  records still overflow on a regional crop.
# ---------------------------------------------------------------------------

_SLV_BBOX = (-114.0, 39.5, -110.5, 42.0)
_WEST_NA_BBOX = (-140.0, 20.0, -85.0, 60.0)

_SOURCE_OPEN_CASES = [
    pytest.param(HRRRSource(), _HRRR_TEST_TIME, _SLV_BBOX, id="hrrr"),
    pytest.param(HRRRv1Source(), "2017-06-01 06:00", _SLV_BBOX, id="hrrr-v1"),
    pytest.param(NAMSource(), _GDAS_TEST_TIME, _WEST_NA_BBOX, id="nam"),
    pytest.param(NAMSSource(), _GDAS_TEST_TIME, _WEST_NA_BBOX, id="nams"),
    pytest.param(GDASSource(), _GDAS_TEST_TIME, _WEST_NA_BBOX, id="gdas1"),
    pytest.param(GDAS0p5Source(), "2018-07-01", _WEST_NA_BBOX, id="gdas0p5"),
    pytest.param(GFSSource(), _GDAS_TEST_TIME, _WEST_NA_BBOX, id="gfs"),
    pytest.param(NARRSource(), "2018-07-01", _WEST_NA_BBOX, id="narr"),
    pytest.param(ReanalysisSource(), _GDAS_TEST_TIME, None, id="reanalysis"),
]


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize("source,time,bbox", _SOURCE_OPEN_CASES)
def test_source_fetch_and_open(tmp_path, source, time, bbox):
    """Fetch one file from each source and verify it opens as an xarray Dataset."""
    from arlmet import open_dataset

    files = source.fetch(time, time, local_dir=tmp_path, bbox=bbox)

    assert len(files) == 1
    dest = files[0]
    assert dest.exists()
    assert dest.stat().st_size > 0

    ds = open_dataset(dest)
    assert len(ds.data_vars) > 0
    assert "level" in ds.dims
