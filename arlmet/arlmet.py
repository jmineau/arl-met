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
    met = ARLMet(path=filename)
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

    def __init__(self, path: Path | str):  # TODO: move to classmethod
        self.path = Path(path)

        if not self.path.exists():
            raise ValueError("Invalid file path")

        # Open the file
        with self.path.open("rb") as f:
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
                index_len = fixed["index_length"]
                index_end = cursor + index_len
                catalog = IndexRecord.parse_extended(
                    data=data[fixed_end:index_end], nz=fixed["nz"]
                )
                index_record = IndexRecord(
                    header=header, **fixed, levels=catalog.levels
                )
                cursor += index_len

                # Calculate grid size
                nxy = index_record.grid.nx * index_record.grid.ny
                cursor += nxy - index_len  # Skip any extra bytes in index record
            else:
                if index_record is None:
                    raise ValueError("Data record found before index record")
                record = DataRecord(
                    index_record=index_record,
                    header=header,
                    data=data[cursor : cursor + nxy],
                )

                records.append(
                    {
                        "grid": None,  # TODO how to identify grid?
                        "time": record.time,
                        "forecast": header.forecast,
                        "level": header.level,
                        "variable": header.variable,
                        "record": record,
                    }
                )
                cursor += nxy  # Move cursor past packed data

        index = pd.DataFrame(records)
        # index_keys = ['grid', 'time', 'forecast', 'level', 'variable']  # TODO grid
        index_keys = ["time", "forecast", "level", "variable"]
        self._index = index.set_index(index_keys)["record"]

    @property
    def records(self) -> List[DataRecord]:
        return self._index.tolist()

    def load(self, **kwargs) -> xr.Dataset | xr.DataArray:
        """
        Load data into an xarray Dataset.

        Accepts xarray sel-style indexing to select specific times, variables, levels, forecasts, or grids.

        Parameters
        ----------
        **kwargs : dict
        """
        # Select records matching criteria
        index = self._index.to_xarray()
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
        raise NotImplementedError("Merging ARLMet instances is not yet implemented.")
