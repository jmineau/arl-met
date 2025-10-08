"""
ARL meteorological data reader and parser.

This module provides the main ARLMet class for reading and working with
ARL format meteorological files. ARL files are binary packed data files
used by HYSPLIT and other atmospheric transport models.
"""

from pathlib import Path
from typing import Sequence

import pandas as pd
import xarray as xr

from arlmet.grid import Projection, Grid, VerticalAxis, Grid3D
from arlmet.records import Header, IndexRecord, DataRecord

# TODO
# - vertical coords
# - attrs
# - CF compliance


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

    def __init__(self, index_record: IndexRecord, records: Sequence[DataRecord]):
        """
        Initialize ARLMet instance with data records.

        Parameters
        ----------
        records : Sequence[DataRecord]
            List of data records parsed from the ARL file.
        """
        # Build index DataFrame
        index = pd.DataFrame({"record": records}).dropna(ignore_index=True)
        if index.empty:
            raise ValueError("No valid records provided")

        # Add metadata columns
        index["index_record"] = index_record
        index["time"] = index_record.time  # need minutes from index_record

        # Extract keys from records
        for key in ["variable", "level", "forecast"]:
            index[key] = index.record.apply(lambda r: getattr(r, key))

        # Identify DIFF variables
        # The variable preceding a DIFF is the base variable
        index["diff"] = None
        is_diff = index.variable.str.startswith("DIF")
        has_diff = is_diff.shift(-1, fill_value=False)
        shifted = index["record"].shift(-1, fill_value=None)
        index.loc[has_diff, "diff"] = shifted[has_diff]

        # Build projection
        proj = Projection(
            pole_lat=index_record.pole_lat,
            pole_lon=index_record.pole_lon,
            tangent_lat=index_record.tangent_lat,
            tangent_lon=index_record.tangent_lon,
            grid_size=index_record.grid_size,
            orientation=index_record.orientation,
            cone_angle=index_record.cone_angle,
            sync_x=index_record.sync_x,
            sync_y=index_record.sync_y,
            sync_lat=index_record.sync_lat,
            sync_lon=index_record.sync_lon,
            reserved=index_record.reserved,
        )

        # Build vertical axis
        v_axis = VerticalAxis(
            flag=index_record.vertical_flag,
            levels=[lvl.height for lvl in index_record.levels],
        )

        # Build grid
        grid = Grid3D(
            projection=proj,
            nx=index_record.total_nx,
            ny=index_record.total_ny,
            vertical_axis=v_axis,
        )
        index["grid"] = grid

        # Set multi-index
        self._index = index.set_index(["grid", "time", "forecast", "level", "variable"])

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

        mets = []  # Multiple ARLMet instances in a file

        # Parse the file
        cursor = 0
        while cursor < len(data):
            # Read the header
            header = Header.from_bytes(data[cursor : cursor + Header.N_BYTES])
            cursor += Header.N_BYTES

            if header.variable != "INDX":
                raise ValueError("Data record found before index record")

            # Parse the index record to get grid dimensions
            fixed_end = cursor + IndexRecord.N_BYTES_FIXED
            fixed = IndexRecord.parse_fixed(data=data[cursor:fixed_end])
            extended = data[fixed_end : cursor + fixed["index_length"]]
            levels = IndexRecord.parse_extended(data=extended, nz=fixed["nz"])
            index = IndexRecord(header=header, **fixed, levels=levels)

            # Calculate grid size
            nx, ny = index.total_nx, index.total_ny
            nxy = nx * ny

            # Calculate number of records
            n_recs = sum(len(lvl.variables) for lvl in levels)

            cursor += nxy  # Skip any extra bytes in index record

            # Loop over data records
            records = []
            for _ in range(n_recs):
                # Read the variable header
                header = Header.from_bytes(data[cursor : cursor + Header.N_BYTES])
                cursor += Header.N_BYTES

                # Build data record
                record = DataRecord(header=header, data=data[cursor : cursor + nxy])
                records.append(record)

                cursor += nxy  # Move cursor past packed data

            # Build ARLMet instance
            met = cls(index_record=index, records=records)
            mets.append(met)

        # Merge met instances
        if len(mets) > 1:
            met = cls.merge(mets)
        else:
            met = mets[0]

        return met

    @property
    def grids(self) -> list[Grid]:
        """
        Get list of all grids.

        Returns
        -------
        list[Grid]
            List of all grids in the dataset.
        """
        grid_pos = self._index.index.names.index("grid")
        grids = self._index.index.levels[grid_pos]
        return list(grids)

    @property
    def records(self) -> list[DataRecord]:
        """
        Get list of all data records.

        Returns
        -------
        list[DataRecord]
            List of all data records in the dataset.
        """
        return self._index["record"].tolist()

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
        index = self._index.to_xarray().sel(**kwargs).to_dataframe()
        index = index.dropna(subset=["record"]).reset_index()

        # Load each record into a DataArray
        arrays = index.apply(
            lambda row: self._load_record(
                grid=row["grid"],
                index_record=row["index_record"],
                record=row["record"],
                diff=row["diff"],
            ),
            axis=1,
        ).tolist()

        # variable is a dim of index, not data
        variables = kwargs.pop("variable", None)

        if len(arrays) == 1:
            # Single record, return DataArray
            return arrays[0].sel(**kwargs).squeeze()

        # Merge into Dataset
        ds = xr.merge(arrays)

        # Assign additional vertical coordinates
        # TODO

        # Handle attrs  TODO

        if variables is not None:
            ds = ds[variables]  # select only requested variables
        return ds.sel(**kwargs).squeeze()

    @staticmethod
    def _load_record(
        grid: Grid3D,
        index_record: IndexRecord,
        record: DataRecord,
        diff: DataRecord | None = None,
    ) -> xr.DataArray:
        """
        Load the data record into a 3D xarray DataArray.
        """
        # Get horizontal grid dimensions
        nx, ny = grid.nx, grid.ny
        dims = grid.dims
        coords = grid.coords

        # Unpack the data using the differential unpacking algorithm
        unpacked = record.unpack(nx=nx, ny=ny)

        if diff is not None:
            # Apply difference correction
            diff_data = diff.unpack(nx=nx, ny=ny)
            unpacked += diff_data

        # Construct DataArray
        coords_2d = {k: v for k, v in coords.items() if k in ("x", "y", "lon", "lat")}
        da = xr.DataArray(
            data=unpacked, dims=dims[-2:], coords=coords_2d, name=record.header.variable
        )

        # Expand dimensions for time, forecast, level, grid
        height = index_record.levels[record.level].height
        da = da.expand_dims(
            time=[index_record.time],
            forecast=[record.forecast],
            level=[height],
            grid=[grid],
        )

        # Sort on dims
        da = da.sortby(list(dims))

        # TODO: add CF attributes

        return da

    @staticmethod
    def merge(mets: Sequence["ARLMet"]) -> "ARLMet":
        """
        Merge multiple ARLMet instances into a single instance.

        Parameters
        ----------
        mets : Sequence[ARLMet]
            Sequence of ARLMet instances to merge.

        Returns
        -------
        ARLMet
            Merged ARLMet instance.
        """
        indices = [met._index for met in mets]
        index = pd.concat(indices).sort_index()

        if any(index.index.duplicated()):
            raise ValueError("Cannot merge ARLMet instances with overlapping records")

        # Create new ARLMet instance
        merged_met = ARLMet.__new__(ARLMet)  # Bypass __init__
        merged_met._index = index

        return merged_met

    def __add__(self, other: "ARLMet") -> "ARLMet":
        """
        Merge two ARLMet instances.

        Parameters
        ----------
        other : ARLMet
            Another ARLMet instance to merge with this one.

        Returns
        -------
        ARLMet
            Merged ARLMet instance.
        """
        # Merge ARLMet indices
        index = pd.concat([self._index, other._index])

        if any(index.duplicated()):
            raise ValueError("Cannot merge ARLMet instances with overlapping records")

        # Create new ARLMet instance
        merged_met = ARLMet.__new__(ARLMet)  # Bypass __init__
        merged_met._index = index.sort_index()

        return merged_met
