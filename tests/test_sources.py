"""Unit tests for arlmet.sources — no network access required."""

import pandas as pd
import pytest

from arlmet.sources import GdasSource, GfsSource, HrrrSource, NamSource

# ---------------------------------------------------------------------------
# HrrrSource
# ---------------------------------------------------------------------------


class TestHrrrSource:
    def setup_method(self):
        self.src = HrrrSource()

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

    def test_s3_key_uses_year_subdir(self):
        t = pd.Timestamp("2024-07-18 03:00")
        assert self.src._s3_key(t) == "hrrr/2024/20240718_00-05_hrrr"

    def test_keys_for_range_single_block(self):
        keys = self.src.keys_for_range("2024-07-18 01:00", "2024-07-18 04:00")
        assert keys == ["hrrr/2024/20240718_00-05_hrrr"]

    def test_keys_for_range_two_blocks(self):
        keys = self.src.keys_for_range("2024-07-18 04:00", "2024-07-18 08:00")
        assert keys == [
            "hrrr/2024/20240718_00-05_hrrr",
            "hrrr/2024/20240718_06-11_hrrr",
        ]

    def test_keys_for_range_all_four_blocks(self):
        keys = self.src.keys_for_range("2024-07-18 00:00", "2024-07-18 23:00")
        assert keys == [
            "hrrr/2024/20240718_00-05_hrrr",
            "hrrr/2024/20240718_06-11_hrrr",
            "hrrr/2024/20240718_12-17_hrrr",
            "hrrr/2024/20240718_18-23_hrrr",
        ]

    def test_keys_for_range_day_boundary(self):
        keys = self.src.keys_for_range("2024-07-18 22:00", "2024-07-19 02:00")
        assert keys == [
            "hrrr/2024/20240718_18-23_hrrr",
            "hrrr/2024/20240719_00-05_hrrr",
        ]

    def test_keys_for_range_backward_trajectory(self):
        # start > end: backward STILT trajectory
        keys = self.src.keys_for_range("2024-07-18 08:00", "2024-07-18 04:00")
        assert keys == [
            "hrrr/2024/20240718_00-05_hrrr",
            "hrrr/2024/20240718_06-11_hrrr",
        ]

    def test_keys_for_range_no_duplicates(self):
        # All times within the same block → single key
        keys = self.src.keys_for_range("2024-07-18 06:00", "2024-07-18 11:00")
        assert keys == ["hrrr/2024/20240718_06-11_hrrr"]

    def test_repr(self):
        assert repr(self.src) == "HrrrSource()"


# ---------------------------------------------------------------------------
# NamSource
# ---------------------------------------------------------------------------


class TestNamSource:
    def setup_method(self):
        self.src = NamSource()

    def test_filename(self):
        assert self.src._filename(pd.Timestamp("2024-07-18")) == "20240718_nam12"

    def test_s3_key(self):
        assert self.src._s3_key(pd.Timestamp("2024-07-18")) == "nam12/2024/20240718_nam12"

    def test_keys_for_range_single_day(self):
        keys = self.src.keys_for_range("2024-07-18 06:00", "2024-07-18 18:00")
        assert keys == ["nam12/2024/20240718_nam12"]

    def test_keys_for_range_two_days(self):
        keys = self.src.keys_for_range("2024-07-18 22:00", "2024-07-19 06:00")
        assert keys == [
            "nam12/2024/20240718_nam12",
            "nam12/2024/20240719_nam12",
        ]

    def test_keys_for_range_backward(self):
        keys = self.src.keys_for_range("2024-07-19 06:00", "2024-07-18 22:00")
        assert keys == [
            "nam12/2024/20240718_nam12",
            "nam12/2024/20240719_nam12",
        ]


# ---------------------------------------------------------------------------
# GdasSource
# ---------------------------------------------------------------------------


class TestGdasSource:
    def setup_method(self):
        self.src = GdasSource()

    @pytest.mark.parametrize(
        "date_str, expected_week",
        [
            ("2025-09-01", 1), ("2025-09-07", 1),
            ("2025-09-08", 2), ("2025-09-14", 2),
            ("2025-09-15", 3), ("2025-09-21", 3),
            ("2025-09-22", 4), ("2025-09-28", 4),
            ("2025-09-29", 5), ("2025-09-30", 5),
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
            self.src._s3_key(pd.Timestamp("2025-09-01"))
            == "gdas1/2025/gdas1.sep25.w1"
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
# GfsSource
# ---------------------------------------------------------------------------


class TestGfsSource:
    def setup_method(self):
        self.src = GfsSource()

    def test_filename(self):
        assert self.src._filename(pd.Timestamp("2024-07-18")) == "20240718_gfs0p25"

    def test_s3_key(self):
        assert (
            self.src._s3_key(pd.Timestamp("2024-07-18"))
            == "gfs0p25/2024/20240718_gfs0p25"
        )

    def test_keys_for_range_single_day(self):
        keys = self.src.keys_for_range("2024-07-18 06:00", "2024-07-18 18:00")
        assert keys == ["gfs0p25/2024/20240718_gfs0p25"]

    def test_keys_for_range_two_days(self):
        keys = self.src.keys_for_range("2024-07-18 20:00", "2024-07-19 06:00")
        assert keys == [
            "gfs0p25/2024/20240718_gfs0p25",
            "gfs0p25/2024/20240719_gfs0p25",
        ]


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


class TestUrls:
    def setup_method(self):
        self.src = HrrrSource()

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

    src = HrrrSource()
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
        src = GdasSource()
        key = src._s3_key(pd.Timestamp(_GDAS_TEST_TIME))
        assert self._exists(key), f"Expected S3 key not found: {key}"

    def test_hrrr_key_exists(self):
        src = HrrrSource()
        key = src._s3_key(pd.Timestamp(_HRRR_TEST_TIME))
        assert self._exists(key), f"Expected S3 key not found: {key}"

    def test_nam_key_exists(self):
        src = NamSource()
        key = src._s3_key(pd.Timestamp(_GDAS_TEST_TIME))
        assert self._exists(key), f"Expected S3 key not found: {key}"

    def test_gfs_key_exists(self):
        src = GfsSource()
        key = src._s3_key(pd.Timestamp(_GDAS_TEST_TIME))
        assert self._exists(key), f"Expected S3 key not found: {key}"


@pytest.mark.network
def test_gdas_header_is_valid_arl(tmp_path):
    """Download the first 50 bytes of a GDAS file and check the ARL header."""
    import s3fs

    from arlmet.metadata import Header

    src = GdasSource()
    key = src._s3_key(pd.Timestamp(_GDAS_TEST_TIME))
    fs = s3fs.S3FileSystem(anon=True)
    with fs.open(f"noaa-oar-arl-hysplit-pds/{key}", "rb") as f:
        raw = f.read(Header.N_BYTES)

    assert len(raw) == Header.N_BYTES
    header = Header.frombytes(raw)
    assert header.variable == "INDX"


@pytest.mark.network
@pytest.mark.slow
def test_gdas_fetch_with_bbox(tmp_path):
    """Fetch one GDAS weekly file cropped to a western North America domain.

    GDAS is 1-degree resolution with many levels/variables, so the ARL index
    record is large (~1654 bytes). The bbox must yield enough grid cells
    (nx*ny >= ~1604) to hold it — the SLV domain is far too small for GDAS.
    Use a regional domain instead.
    """
    from arlmet import File

    src = GdasSource()
    # Western North America: ~55 deg lon x 40 deg lat = ~2200 cells at 1-degree
    west_na_bbox = (-140.0, 20.0, -85.0, 60.0)
    files = src.fetch(
        _GDAS_TEST_TIME,
        _GDAS_TEST_TIME,
        local_dir=tmp_path,
        bbox=west_na_bbox,
    )

    assert len(files) == 1
    dest = files[0]
    assert dest.exists()
    assert dest.stat().st_size > 0

    with File(dest) as f:
        assert f.grid.nx > 0
        assert f.grid.ny > 0
        assert len(f.times) > 0
