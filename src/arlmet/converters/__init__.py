"""
Convert other models' output to ARL files.

Each converter turns one product's own output, opened with xarray, into the
Dataset :func:`arlmet.write_dataset` writes: ARL's four-character variable
names, the units HYSPLIT reads, rows from south to north, the vertical axis
HYSPLIT expects for the product, and ``attrs["diff"]`` where a ``DIF*`` record
follows. The Dataset can be looked at and plotted before anything is written.
These are Python versions of HYSPLIT's ``data2arl`` programs (``era52arl``,
``api2arl``,
``arw2arl``, ...), and one for DWD's ICON, which HYSPLIT has none for.

Units follow what HYSPLIT reads, not always what its converter writes:
``TCLD`` is in percent (``metinp.f`` divides it by 100), the heat fluxes
``SHTF`` and ``LHTF`` are positive upward (``stbanl.f``), and the latent heat
flux is named ``LHTF`` (``metini.f``).
"""

from ._era5 import from_era5
from ._gfs import from_gfs
from ._icon import from_icon
from ._wrf import from_wrf

__all__ = ["from_era5", "from_gfs", "from_icon", "from_wrf"]
