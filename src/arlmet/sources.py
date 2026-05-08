"""Remote meteorological data sources for NOAA ARL archives.

Each source class encodes the filename convention, S3 path layout, and
approximate spatial extent for one ARL-formatted met product hosted on
the NOAA ARL public archives.

Storage backends
----------------
"s3"   : AWS S3 (noaa-oar-arl-hysplit-pds, anonymous) — recommended
"ftp"  : NOAA ARL FTP (ftp.arl.noaa.gov, anonymous, 2-connection limit)
"http" : NOAA READY web (www.ready.noaa.gov/data/archives)

Example
-------
>>> from arlmet.sources import HrrrSource
>>> source = HrrrSource()
>>> files = source.fetch("2024-07-18", "2024-07-19", local_dir="./met/")

>>> # Crop to domain on download (strongly recommended for GFS)
>>> files = source.fetch(
...     "2024-07-18", "2024-07-19",
...     local_dir="./met/",
...     bbox=(-114.0, 39.0, -110.0, 42.0),
... )

Requires ``fsspec`` (and ``s3fs`` for the S3 backend).
Install with: ``pip install arlmet[sources]``
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import pandas as pd

logger = logging.getLogger(__name__)

_MONTH_CODES: tuple[str, ...] = (
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
)


class MeteorologySource(ABC):
    """Abstract base for NOAA ARL meteorological data sources.

    Subclasses set class-level metadata and implement ``_s3_key()`` to
    encode the filename convention for their product.
    """

    # Subclasses must define these
    name: ClassVar[str]
    description: ClassVar[str]
    #: Approximate domain (west, south, east, north) in degrees.
    spatial_extent: ClassVar[tuple[float, float, float, float]]
    #: Earliest date available in the NOAA archive.
    start_date: ClassVar[pd.Timestamp]

    S3_BUCKET: ClassVar[str] = "noaa-oar-arl-hysplit-pds"
    FTP_HOST: ClassVar[str] = "ftp.arl.noaa.gov"
    HTTP_BASE: ClassVar[str] = "https://www.ready.noaa.gov/data/archives"

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _s3_key(self, time: pd.Timestamp) -> str:
        """S3 key (no leading slash) for the ARL file containing *time*."""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def keys_for_range(
        self,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
    ) -> list[str]:
        """Return deduplicated, sorted S3 keys covering ``[start, end]``.

        Handles backward trajectories (``start > end``) by normalizing to
        chronological order before scanning.
        """
        t0 = pd.Timestamp(start).floor("h")
        t1 = pd.Timestamp(end).floor("h")
        if t0 > t1:
            t0, t1 = t1, t0

        seen: set[str] = set()
        keys: list[str] = []
        t = t0
        while t <= t1:
            key = self._s3_key(t)
            if key not in seen:
                seen.add(key)
                keys.append(key)
            t += pd.Timedelta(hours=1)
        return keys

    def fetch(
        self,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        *,
        local_dir: Path | str,
        backend: str = "s3",
        bbox: tuple[float, float, float, float] | None = None,
        overwrite: bool = False,
    ) -> list[Path]:
        """Download ARL files covering ``[start, end]`` to *local_dir*.

        Parameters
        ----------
        start, end :
            Time range (inclusive). Backward trajectories (start > end)
            are handled automatically.
        local_dir :
            Directory to save downloaded files. Created if absent.
        backend :
            Storage backend — ``"s3"`` (default), ``"ftp"``, or ``"http"``.
        bbox :
            ``(west, south, east, north)`` in degrees. When provided, each
            file is cropped with :func:`arlmet.extract_subset` before
            caching. Strongly recommended for global products (GFS, GDAS).
        overwrite :
            Re-download even if a matching local file already exists.

        Returns
        -------
        list[Path]
            Local paths to the downloaded (and optionally cropped) files,
            in chronological order.

        Raises
        ------
        ImportError
            If ``fsspec`` is not installed.
        """
        try:
            import fsspec  # noqa: F401
        except ImportError:
            raise ImportError(
                "fsspec is required for MeteorologySource.fetch(). "
                "Install with: pip install arlmet[sources]"
            ) from None

        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        results: list[Path] = []
        for key in self.keys_for_range(start, end):
            filename = Path(key).name
            dest = self._dest_path(local_dir, filename, bbox)

            if not overwrite and dest.exists():
                logger.debug("Using cached %s", dest.name)
                results.append(dest)
                continue

            url = self._url(key, backend)
            opts = self._storage_options(backend)
            logger.info("Fetching %s → %s", url, dest.name)

            if bbox is not None:
                self._fetch_and_crop(url, dest, bbox, opts)
            else:
                self._download(url, dest, opts)

            results.append(dest)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dest_path(
        self,
        local_dir: Path,
        filename: str,
        bbox: tuple[float, float, float, float] | None,
    ) -> Path:
        if bbox is None:
            return local_dir / filename
        w, s, e, n = bbox
        tag = f".crop_{w:.2f}_{s:.2f}_{e:.2f}_{n:.2f}"
        return local_dir / f"{filename}{tag}"

    def _url(self, key: str, backend: str) -> str:
        if backend == "s3":
            return f"s3://{self.S3_BUCKET}/{key}"
        if backend == "ftp":
            # FTP path mirrors S3 key structure under /archives/
            return f"ftp://anonymous@{self.FTP_HOST}/archives/{key}"
        if backend == "http":
            return f"{self.HTTP_BASE}/{key}"
        raise ValueError(
            f"Unknown backend {backend!r}. Choose 's3', 'ftp', or 'http'."
        )

    def _storage_options(self, backend: str) -> dict:
        if backend == "s3":
            return {"anon": True}
        return {}

    def _download(self, url: str, dest: Path, opts: dict) -> None:
        import fsspec

        tmp = dest.with_suffix(dest.suffix + ".tmp")
        try:
            with fsspec.open(url, "rb", **opts) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            tmp.rename(dest)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def _fetch_and_crop(
        self,
        url: str,
        dest: Path,
        bbox: tuple[float, float, float, float],
        opts: dict,
    ) -> None:
        from arlmet.subset import extract_subset

        with tempfile.NamedTemporaryFile(suffix=".arl", delete=False) as f:
            tmp = Path(f.name)
        try:
            self._download(url, tmp, opts)
            extract_subset(tmp, dest, bbox=bbox)
        finally:
            tmp.unlink(missing_ok=True)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------------------
# Concrete source implementations
# ---------------------------------------------------------------------------


class HrrrSource(MeteorologySource):
    """HRRR 3 km analysis (CONUS, June 2019–present).

    Files cover 6-hour UTC blocks (00–05, 06–11, 12–17, 18–23),
    approximately 3.2 GB each.

    S3: ``s3://noaa-oar-arl-hysplit-pds/hrrr/{year}/{YYYYMMDD}_{HH}-{HH}_hrrr``

    Note
    ----
    The earliest June 2019 files may exist at the bucket root rather than
    under the year subdirectory. This edge case is not currently handled.
    """

    name = "hrrr"
    description = "HRRR 3 km analysis"
    spatial_extent = (-134.1, 21.1, -60.9, 52.6)
    start_date = pd.Timestamp("2019-06-12")

    _HOURS_PER_FILE: ClassVar[int] = 6

    def _filename(self, time: pd.Timestamp) -> str:
        start_h = (time.hour // self._HOURS_PER_FILE) * self._HOURS_PER_FILE
        end_h = start_h + self._HOURS_PER_FILE - 1
        return f"{time.strftime('%Y%m%d')}_{start_h:02d}-{end_h:02d}_hrrr"

    def _s3_key(self, time: pd.Timestamp) -> str:
        return f"hrrr/{time.year}/{self._filename(time)}"


class NamSource(MeteorologySource):
    """NAM 12 km analysis (North America, May 2007–present).

    One file per calendar day.

    S3: ``s3://noaa-oar-arl-hysplit-pds/nam12/{year}/{YYYYMMDD}_nam12``
    """

    name = "nam12"
    description = "NAM 12 km analysis"
    spatial_extent = (-153.0, 12.2, -49.4, 61.2)
    start_date = pd.Timestamp("2007-05-01")

    def _filename(self, time: pd.Timestamp) -> str:
        return f"{time.strftime('%Y%m%d')}_nam12"

    def _s3_key(self, time: pd.Timestamp) -> str:
        return f"nam12/{time.year}/{self._filename(time)}"


class GdasSource(MeteorologySource):
    """GDAS 1-degree global analysis (December 2004–present).

    Weekly files (~571 MB each). Week boundaries are fixed per month:
    w1 = days 1–7, w2 = days 8–14, w3 = days 15–21,
    w4 = days 22–28, w5 = days 29–end.

    S3: ``s3://noaa-oar-arl-hysplit-pds/gdas1/{year}/gdas1.{mon}{YY}.w{N}``
    """

    name = "gdas1"
    description = "GDAS 1-degree global analysis"
    spatial_extent = (-180.0, -90.0, 180.0, 90.0)
    start_date = pd.Timestamp("2004-12-01")

    def _week(self, time: pd.Timestamp) -> int:
        return (time.day - 1) // 7 + 1

    def _filename(self, time: pd.Timestamp) -> str:
        month = _MONTH_CODES[time.month - 1]
        year_2d = time.strftime("%y")
        return f"gdas1.{month}{year_2d}.w{self._week(time)}"

    def _s3_key(self, time: pd.Timestamp) -> str:
        return f"gdas1/{time.year}/{self._filename(time)}"


class GfsSource(MeteorologySource):
    """GFS 0.25-degree global analysis (June 2019–present).

    One file per calendar day, approximately 2.7 GB each.
    Cropping with ``bbox=`` on fetch is strongly recommended.

    S3: ``s3://noaa-oar-arl-hysplit-pds/gfs0p25/{year}/{YYYYMMDD}_gfs0p25``
    """

    name = "gfs0p25"
    description = "GFS 0.25-degree global analysis"
    spatial_extent = (-180.0, -90.0, 180.0, 90.0)
    start_date = pd.Timestamp("2019-06-01")

    def _filename(self, time: pd.Timestamp) -> str:
        return f"{time.strftime('%Y%m%d')}_gfs0p25"

    def _s3_key(self, time: pd.Timestamp) -> str:
        return f"gfs0p25/{time.year}/{self._filename(time)}"


__all__ = [
    "MeteorologySource",
    "HrrrSource",
    "NamSource",
    "GdasSource",
    "GfsSource",
]
