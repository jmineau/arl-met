"""
ARL meteorological data reader and parser.

This module provides the main ARLMet class for reading and working with
ARL format meteorological files. ARL files are binary packed data files
used by HYSPLIT and other atmospheric transport models.
"""

from pathlib import Path
from typing import List

import pandas as pd
import xarray as xr

from arlmet.records import Header, IndexRecord, DataRecord

# TODO
# - vertical axis
# - CF compliance
# - VariableCatalog


def open_dataset(filename: Path | str, **kwargs) -> xr.Dataset:
    """
    Open an ARL meteorological data file as an xarray Dataset.

    Parameters
    ----------
    filename : Path or str
        Path to the ARL data file.
    **kwargs
        sel-like keyword arguments passed to `ARLMet.load()`.

    Returns
    -------
    xr.Dataset
        The ARL data as an xarray Dataset.
    """
    met = ARLMet.from_file(filename=filename)
    ds = met.load(**kwargs)
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    return ds


class ARLMet:
    """
    ARL (Air Resources Laboratory) packed meteorological data.

    Extracts metadata from meteorological data file headers and provides
    methods to work with ARL format files.
    """

    # _KEYS = ["grid", "time", "forecast", "level", "variable"]
    _KEYS = ["time", "forecast", "level", "variable"]  # FIXME how to identify grid?

    def __init__(self, records: List[DataRecord]):
        """
        Initialize ARLMet instance with data records.

        Parameters
        ----------
        records : List[DataRecord]
            List of data records parsed from the ARL file.
        """
        # Build index DataFrame
        index = pd.DataFrame({"record": records}).dropna(ignore_index=True)
        if index.empty:
            raise ValueError("No valid records provided")

        # Extract keys from records
        for key in self._KEYS:
            index[key] = index.record.apply(lambda r: getattr(r, key))

        # Set multi-index
        self.index = index.set_index(self._KEYS)["record"]

    @classmethod
    def from_file(cls, filename: Path | str) -> "ARLMet":
        """
        Create an ARLMet instance by reading an ARL file.

        Parameters
        ----------
        filename : Path or str
            Path to the ARL meteorological data file.

        Returns
        -------
        ARLMet
            ARLMet instance containing the parsed data records.

        Raises
        ------
        ValueError
            If the file path is invalid or if a data record is found before an index record.
        """
        path = Path(filename)

        if not path.exists():
            raise ValueError("Invalid file path")

        # Open the file
        with path.open("rb") as f:
            data = f.read()

        records = []

        index_record = None
        nxy = 0

        # Build the index
        cursor = 0
        while cursor < len(data):
            # Read the next header
            header = Header.from_bytes(data[cursor : cursor + Header.N_BYTES])
            cursor += Header.N_BYTES

            if header.variable == "INDX":
                # Parse the index record to get grid dimensions
                fixed_end = cursor + IndexRecord.N_BYTES_FIXED
                fixed = IndexRecord.parse_fixed(data=data[cursor:fixed_end])
                index_end = cursor + fixed["index_length"]
                catalog = IndexRecord.parse_extended(
                    data=data[fixed_end:index_end], nz=fixed["nz"]
                )
                index_record = IndexRecord(
                    header=header, **fixed, levels=catalog.levels
                )

                # Calculate grid size
                nxy = index_record.grid.nx * index_record.grid.ny
                cursor += nxy  # Skip any extra bytes in index record
            else:
                if index_record is None:
                    raise ValueError("Data record found before index record")

                # Build data record
                record = DataRecord(
                    index_record=index_record,
                    header=header,
                    data=data[cursor : cursor + nxy],
                )
                records.append(record)
                cursor += nxy  # Move cursor past packed data

        return cls(records=records)

    @property
    def records(self) -> List[DataRecord]:
        """
        Get list of all data records.

        Returns
        -------
        List[DataRecord]
            List of all data records in the dataset.
        """
        return self.index.tolist()

    def load(self, **kwargs) -> xr.Dataset | xr.DataArray:
        """
        Load data into an xarray Dataset or DataArray.

        Accepts xarray sel-style indexing to select specific times, variables,
        levels, forecasts, or grids.

        Parameters
        ----------
        **kwargs : dict
            Selection criteria using xarray sel-style indexing (e.g., time, variable, level, forecast).

        Returns
        -------
        xr.Dataset or xr.DataArray
            If a single record matches the criteria, returns a DataArray.
            Otherwise, returns a Dataset containing multiple variables/times.
        """
        # Select records matching criteria
        index = self.index.to_xarray()
        records = index.sel(**kwargs).values.flatten().tolist()

        # Unpack each record to DataArray
        arrays = [
            r.unpack() for r in records if isinstance(r, DataRecord)
        ]  # drop nans from xarray

        # variable is a dim of index, not data
        variables = kwargs.pop("variable", None)

        if len(arrays) == 1:
            # Single record, return DataArray
            return arrays[0].sel(**kwargs)

        # Multiple records, return Dataset
        ds = xr.merge(arrays)
        if variables is not None:
            ds = ds[variables]  # select only requested variables
        return ds.sel(**kwargs)

    def __add__(self, other: "ARLMet") -> "ARLMet":
        """
        Merge two ARLMet instances (not yet implemented).

        Parameters
        ----------
        other : ARLMet
            Another ARLMet instance to merge with this one.

        Returns
        -------
        ARLMet
            Merged ARLMet instance.

        Raises
        ------
        NotImplementedError
            This functionality is not yet implemented.
        """
        raise NotImplementedError("Merging ARLMet instances is not yet implemented.")
