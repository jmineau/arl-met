import importlib.resources
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr

from arlmet.grid import Grid, Projection, wrap_lons


@dataclass
class ASCDataConfig:
    ll_lat: float
    ll_lon: float
    dlat: float
    dlon: float
    nlat: int
    nlon: int
    land_use: int
    roughness: float
    data_dir: Path

    @classmethod
    def from_file(cls, file: str) -> "ASCDataConfig":
        """Load ASC terrain configuration bundled with the package."""
        resources = importlib.resources.files("arlmet.resources")
        with importlib.resources.as_file(resources / file) as path, open(path) as handle:
            lines = handle.readlines()

        if len(lines) < 6:
            raise ValueError("Configuration file must have at least 6 lines.")

        ll = lines[0].split()[:2]
        d = lines[1].split()[:2]
        n = lines[2].split()[:2]
        land_use = lines[3].split()[0]
        roughness = lines[4].split()[0]
        data_dir = lines[5].split()[0].strip("'\"")

        return cls(
            ll_lat=float(ll[0]),
            ll_lon=float(ll[1]),
            dlat=float(d[0]),
            dlon=float(d[1]),
            nlat=int(n[0]),
            nlon=int(n[1]),
            land_use=int(land_use),
            roughness=float(roughness),
            data_dir=path.parent / data_dir,
        )


class DefaultTerrain:
    def __init__(self, filename: str = "TERRAIN.ASC", config: str = "ASCDATA.CFG"):
        self.filename = filename
        self.config = (
            config
            if isinstance(config, ASCDataConfig)
            else ASCDataConfig.from_file(config)
        )
        self._data: np.ndarray | None = None

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            terrain_file = self.config.data_dir / self.filename
            if not terrain_file.exists():
                raise FileNotFoundError(f"Terrain file not found: {terrain_file}")

            data = np.genfromtxt(terrain_file, delimiter=4, dtype="f4")
            if data.shape != (self.config.nlat, self.config.nlon):
                raise ValueError("Terrain data shape does not match configuration.")
            self._data = data
        return self._data

    @property
    def lats(self) -> np.ndarray:
        lats = self.config.ll_lat + np.arange(self.config.nlat) * self.config.dlat
        return lats[::-1]

    @property
    def lons(self) -> np.ndarray:
        lons = self.config.ll_lon + np.arange(self.config.nlon) * self.config.dlon
        return wrap_lons(lons)

    def to_xarray(self) -> xr.DataArray:
        return xr.DataArray(
            self.data,
            dims=("lat", "lon"),
            coords={"lat": self.lats, "lon": self.lons},
            name="terrain",
            attrs={"units": "m", "description": "Terrain height"},
        ).sortby("lat")


class Surface:
    DEFAULT_PRESSURE = 1013.0

    def __init__(
        self,
        terrain: npt.ArrayLike | None = None,
        pressure: npt.ArrayLike | None = None,
        default_terrain: DefaultTerrain | None = None,
    ):
        self._terrain = terrain
        self._pressure = pressure
        self._default_terrain = default_terrain

    @property
    def terrain(self) -> np.ndarray | None:
        if self._terrain is not None:
            return np.asarray(self._terrain, dtype=float)
        if self._default_terrain is not None:
            return np.asarray(self._default_terrain.data, dtype=float)
        return None

    @property
    def pressure(self) -> np.ndarray:
        if self._pressure is None:
            return np.asarray(self.DEFAULT_PRESSURE, dtype=float)
        return np.asarray(self._pressure, dtype=float)

    @property
    def mean_pressure(self) -> float:
        return float(np.asarray(self.pressure, dtype=float).mean())

    @property
    def mean_terrain(self) -> float:
        terrain = self.terrain
        if terrain is None:
            return 0.0
        return float(np.asarray(terrain, dtype=float).mean())


class VerticalAxis:
    FLAGS: dict[int, str] = {
        1: "sigma",
        2: "pressure",
        3: "terrain",
        4: "hybrid",
        5: "wrf",
    }

    Z_PHI1 = np.asarray((17.98, 14.73, 13.09, 11.98, 11.15, 10.52, 10.04, 9.75, 9.88))
    Z_PHI2 = np.asarray((31.37, 27.02, 24.59, 22.92, 21.65, 20.66, 19.83, 19.13, 18.51))

    def __init__(
        self,
        flag: int,
        heights: Sequence[float] | None = None,
        *,
        levels: Sequence[float] | None = None,
        offset: float = 0.0,
        surface: Surface | None = None,
    ):
        if heights is None:
            if levels is None:
                raise TypeError("VerticalAxis requires `heights` or `levels`.")
            heights = levels
        elif levels is not None:
            raise TypeError("Pass only one of `heights` or `levels`.")

        self.flag = flag
        self._heights = np.asarray(heights, dtype=float)
        self.offset = float(offset)
        self.surface = surface

    @property
    def coord_system(self) -> str:
        return self.FLAGS.get(self.flag, "unknown")

    @property
    def dims(self) -> tuple[str]:
        return ("level",)

    @property
    def heights(self) -> np.ndarray:
        return self._heights.copy()

    @property
    def levels(self) -> np.ndarray:
        return self.heights

    def with_surface(self, surface: Surface | None) -> "VerticalAxis":
        return VerticalAxis(
            flag=self.flag,
            heights=self._heights,
            offset=self.offset,
            surface=surface,
        )

    def calculate_coords(self) -> dict[str, np.ndarray]:
        coords = {
            "level": np.arange(len(self._heights), dtype=int),
            "height": self.height_agl(),
        }
        pressure = self.pressure()
        if pressure is not None:
            coords["pressure"] = pressure
        if self.flag == 1:
            coords["sigma"] = self.heights
        return coords

    def pressure(self) -> np.ndarray | None:
        p0 = self._surface_pressure()

        if self.flag == 1:
            return self.offset + (p0 - self.offset) * self._heights
        if self.flag == 2:
            return self.heights
        if self.flag == 3:
            return None
        if self.flag == 4:
            offsets = np.floor(self._heights)
            sigma = self._heights - offsets
            pressure = p0 * sigma + offsets
            if pressure.size:
                pressure[0] = p0
            return pressure
        if self.flag == 5:
            raise NotImplementedError("WRF hybrid vertical coordinates are not implemented.")
        return None

    def height_agl(self) -> np.ndarray:
        if self.flag == 3:
            return self.heights

        pressure = self.pressure()
        if pressure is None:
            raise ValueError(
                f"Cannot derive height for vertical coordinate system '{self.coord_system}'."
            )
        return self._p_to_z(pressure)

    def height_msl(self) -> np.ndarray:
        return self.height_agl() + self._surface_terrain()

    def _surface_pressure(self) -> float:
        if self.surface is None:
            return Surface.DEFAULT_PRESSURE
        return self.surface.mean_pressure

    def _surface_terrain(self) -> float:
        if self.surface is None:
            return 0.0
        return self.surface.mean_terrain

    def _p_to_z(self, p: npt.ArrayLike) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        delta_z = np.zeros_like(p, dtype=float)

        mask1 = p >= 100.0
        if np.any(mask1):
            idx1 = np.clip((p[mask1] // 100).astype(int), 0, len(self.Z_PHI1) - 1)
            delta_z[mask1] = self.Z_PHI1[idx1]

        mask2 = ~mask1
        if np.any(mask2):
            idx2 = np.clip((p[mask2] // 10).astype(int), 0, len(self.Z_PHI2) - 1)
            delta_z[mask2] = self.Z_PHI2[idx2]

        z = (self._surface_pressure() - p) * delta_z
        return np.where(z < 0.0, 0.0, z)

    def __eq__(self, other) -> bool:
        if not isinstance(other, VerticalAxis):
            return False
        return (
            self.flag == other.flag
            and self.offset == other.offset
            and np.array_equal(self._heights, other._heights)
        )

    def __hash__(self) -> int:
        return hash((self.flag, self.offset, tuple(self._heights)))


def _vertical_axis_from_attrs(attrs: dict[str, object]) -> VerticalAxis:
    try:
        flag = int(attrs["arl_vertical_flag"])
        levels = attrs["arl_vertical_levels"]
        offset = float(attrs.get("arl_vertical_offset", 0.0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Vertical metadata is required. Pass vertical_axis=... or include "
            "'vertical_axis' / 'arl_vertical_*' metadata."
        ) from exc

    try:
        return VerticalAxis(flag=flag, levels=levels, offset=offset)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Vertical metadata is required. Pass vertical_axis=... or include "
            "'vertical_axis' / 'arl_vertical_*' metadata."
        ) from exc


def _resolve_vertical_axis(
    obj,
    vertical_axis: VerticalAxis | None = None,
) -> VerticalAxis:
    if vertical_axis is not None:
        return vertical_axis
    if isinstance(obj, VerticalAxis):
        return obj

    attrs = getattr(obj, "attrs", {})
    axis = attrs.get("vertical_axis")
    if isinstance(axis, VerticalAxis):
        return axis
    if isinstance(attrs, dict):
        return _vertical_axis_from_attrs(attrs)
    raise ValueError(
        "Vertical metadata is required. Pass vertical_axis=... or include "
        "'vertical_axis' / 'arl_vertical_*' metadata."
    )


def _require_pressure_axis(axis: VerticalAxis, func_name: str) -> None:
    if axis.flag != 2:
        raise NotImplementedError(
            f"{func_name}() currently only supports pressure-level vertical axes (flag == 2)."
        )


def _level_coord(obj, axis: VerticalAxis) -> xr.DataArray:
    if isinstance(obj, (xr.Dataset, xr.DataArray)) and "level" in obj.coords:
        coord = obj.coords["level"]
        if coord.ndim == 1:
            return coord

    return xr.DataArray(np.arange(len(axis.heights), dtype=int), dims=("level",), name="level")


def _surface_slice(field):
    if isinstance(field, xr.DataArray) and "level" in field.dims:
        other_dims = [dim for dim in field.dims if dim != "level"]
        if not other_dims:
            return field.isel(level=0, drop=True)

        valid = ~field.isnull().all(dim=other_dims)
        valid_index = np.flatnonzero(np.asarray(valid.values))
        if valid_index.size == 0:
            return None
        return field.isel(level=int(valid_index[0]), drop=True)
    return field


def _dataset_surface_field(dataset: xr.Dataset | xr.DataArray, name: str):
    if not isinstance(dataset, xr.Dataset) or name not in dataset.data_vars:
        return None
    return _surface_slice(dataset[name])


def _surface_pressure_field(
    obj,
    surface_pressure: xr.DataArray | npt.ArrayLike | None = None,
):
    if surface_pressure is not None:
        field = _surface_slice(surface_pressure)
        return field if field is not None else xr.DataArray(Surface.DEFAULT_PRESSURE)

    dataset_pressure = _dataset_surface_field(obj, "PRSS")
    if dataset_pressure is not None:
        return dataset_pressure
    return xr.DataArray(Surface.DEFAULT_PRESSURE)


def _terrain_field(
    obj,
    terrain: xr.DataArray | npt.ArrayLike | None = None,
    default_terrain: DefaultTerrain | None = None,
):
    if terrain is not None:
        field = _surface_slice(terrain)
        if field is None:
            raise ValueError("Explicit terrain data must contain at least one valid slice.")
        return field

    dataset_terrain = _dataset_surface_field(obj, "SHGT")
    if dataset_terrain is not None:
        return dataset_terrain

    if default_terrain is not None:
        return default_terrain.to_xarray()

    raise ValueError(
        "Terrain is required to derive z_msl. Pass terrain=..., include SHGT in the "
        "dataset, or opt in with default_terrain=DefaultTerrain(...)."
    )


def _pressure_level_values(axis: VerticalAxis) -> np.ndarray:
    _require_pressure_axis(axis, "pressure")
    return axis.heights.astype(float, copy=True)


def _pressure_level_coord(obj, axis: VerticalAxis) -> xr.DataArray:
    level_coord = _level_coord(obj, axis)
    level_dim = level_coord.dims[0]
    return xr.DataArray(
        _pressure_level_values(axis),
        dims=(level_dim,),
        coords={level_dim: level_coord},
        name="pressure",
        attrs={"units": "hPa", "long_name": "pressure"},
    )


def _delta_z_for_pressure(pressure_levels: xr.DataArray) -> xr.DataArray:
    p = np.asarray(pressure_levels.values, dtype=float)
    delta_z = np.zeros_like(p, dtype=float)

    mask1 = p >= 100.0
    if np.any(mask1):
        idx1 = np.clip((p[mask1] // 100).astype(int), 0, len(VerticalAxis.Z_PHI1) - 1)
        delta_z[mask1] = VerticalAxis.Z_PHI1[idx1]

    mask2 = ~mask1
    if np.any(mask2):
        idx2 = np.clip((p[mask2] // 10).astype(int), 0, len(VerticalAxis.Z_PHI2) - 1)
        delta_z[mask2] = VerticalAxis.Z_PHI2[idx2]

    return xr.DataArray(
        delta_z,
        dims=pressure_levels.dims,
        coords=pressure_levels.coords,
        name="delta_z",
    )


def _standardize_vertical_dims(data: xr.DataArray) -> xr.DataArray:
    preferred = [
        dim for dim in ("time", "level", "lat", "lon", "y", "x") if dim in data.dims
    ]
    remaining = [dim for dim in data.dims if dim not in preferred]
    return data.transpose(*preferred, *remaining)


def pressure(
    obj,
    *,
    vertical_axis: VerticalAxis | None = None,
) -> xr.DataArray | np.ndarray:
    """
    Return the native pressure coordinate for a pressure-level archive.
    """
    axis = _resolve_vertical_axis(obj, vertical_axis=vertical_axis)
    _require_pressure_axis(axis, "pressure")

    if isinstance(obj, (xr.Dataset, xr.DataArray)):
        return _pressure_level_coord(obj, axis)
    return _pressure_level_values(axis)


def z_agl(
    obj,
    *,
    vertical_axis: VerticalAxis | None = None,
    surface_pressure: xr.DataArray | npt.ArrayLike | None = None,
) -> xr.DataArray | np.ndarray:
    """
    Derive height above ground level for a pressure-level archive.
    """
    axis = _resolve_vertical_axis(obj, vertical_axis=vertical_axis)
    _require_pressure_axis(axis, "z_agl")

    if isinstance(obj, (xr.Dataset, xr.DataArray)):
        pressure_levels = _pressure_level_coord(obj, axis)
        surface_p = _surface_pressure_field(obj, surface_pressure=surface_pressure)
        delta_z = _delta_z_for_pressure(pressure_levels)
        result = (surface_p - pressure_levels) * delta_z
        result = result.clip(min=0.0)
        result = _standardize_vertical_dims(result)
        result.name = "z_agl"
        result.attrs.update(units="m", long_name="height above ground level")
        return result

    p = _pressure_level_values(axis)
    p0 = (
        np.asarray(surface_pressure, dtype=float)
        if surface_pressure is not None
        else np.asarray(Surface.DEFAULT_PRESSURE, dtype=float)
    )
    delta_z = np.asarray(_delta_z_for_pressure(xr.DataArray(p, dims=("level",))).values)
    return np.maximum((p0 - p) * delta_z, 0.0)


def z_msl(
    obj,
    *,
    vertical_axis: VerticalAxis | None = None,
    surface_pressure: xr.DataArray | npt.ArrayLike | None = None,
    terrain: xr.DataArray | npt.ArrayLike | None = None,
    default_terrain: DefaultTerrain | None = None,
) -> xr.DataArray | np.ndarray:
    """
    Derive height above mean sea level for a pressure-level archive.
    """
    axis = _resolve_vertical_axis(obj, vertical_axis=vertical_axis)
    _require_pressure_axis(axis, "z_msl")

    if isinstance(obj, (xr.Dataset, xr.DataArray)):
        agl = z_agl(
            obj,
            vertical_axis=axis,
            surface_pressure=surface_pressure,
        )
        terrain_field = _terrain_field(
            obj,
            terrain=terrain,
            default_terrain=default_terrain,
        )
        result = agl + terrain_field
        result = _standardize_vertical_dims(result)
        result.name = "z_msl"
        result.attrs.update(units="m", long_name="height above mean sea level")
        return result

    if terrain is None:
        raise ValueError(
            "Terrain is required to derive z_msl outside xarray datasets. Pass terrain=..."
        )
    return z_agl(
        obj,
        vertical_axis=axis,
        surface_pressure=surface_pressure,
    ) + np.asarray(terrain, dtype=float)


def _interp_profile(values, coords, target) -> np.float32:
    values = np.asarray(values, dtype=float)
    coords = np.asarray(coords, dtype=float)
    target = float(np.asarray(target, dtype=float))

    mask = np.isfinite(values) & np.isfinite(coords)
    if mask.sum() < 2:
        return np.float32(np.nan)

    values = values[mask]
    coords = coords[mask]
    order = np.argsort(coords)
    coords = coords[order]
    values = values[order]

    unique_coords, unique_index = np.unique(coords, return_index=True)
    unique_values = values[unique_index]
    return np.float32(
        np.interp(target, unique_coords, unique_values, left=np.nan, right=np.nan)
    )


def interp_vertical(
    data: xr.DataArray,
    target: float | xr.DataArray,
    *,
    coord: str = "pressure",
    dataset: xr.Dataset | None = None,
    vertical_axis: VerticalAxis | None = None,
    surface_pressure: xr.DataArray | npt.ArrayLike | None = None,
    terrain: xr.DataArray | npt.ArrayLike | None = None,
    default_terrain: DefaultTerrain | None = None,
) -> xr.DataArray:
    """
    Interpolate a field along the native vertical dimension to a scalar target.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError("interp_vertical() requires an xarray.DataArray.")
    if "level" not in data.dims:
        raise ValueError("interp_vertical() requires a DataArray with a 'level' dimension.")

    context = dataset if dataset is not None else data
    if coord == "pressure":
        vertical_coord = pressure(context, vertical_axis=vertical_axis)
    elif coord == "z_agl":
        vertical_coord = z_agl(
            context,
            vertical_axis=vertical_axis,
            surface_pressure=surface_pressure,
        )
    elif coord == "z_msl":
        vertical_coord = z_msl(
            context,
            vertical_axis=vertical_axis,
            surface_pressure=surface_pressure,
            terrain=terrain,
            default_terrain=default_terrain,
        )
    else:
        raise ValueError("coord must be one of 'pressure', 'z_agl', or 'z_msl'.")

    interpolated = xr.apply_ufunc(
        _interp_profile,
        data,
        vertical_coord,
        target,
        input_core_dims=[["level"], ["level"], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="allowed",
        output_dtypes=[np.float32],
    )
    interpolated.name = data.name
    interpolated.attrs = data.attrs.copy()
    interpolated.attrs["interpolated_vertical_coord"] = coord

    if np.isscalar(target):
        interpolated = interpolated.assign_coords({coord: float(target)})

    return interpolated


class Grid3D(Grid):
    def __init__(
        self,
        projection: Projection | None = None,
        nx: int = 0,
        ny: int = 0,
        vertical_axis: VerticalAxis | None = None,
        *,
        proj: Projection | None = None,
    ):
        projection = projection or proj
        if projection is None:
            raise TypeError("Grid3D requires `projection` or `proj`.")
        if vertical_axis is None:
            raise TypeError("Grid3D requires a `vertical_axis`.")

        super().__init__(projection=projection, nx=nx, ny=ny)
        self.vertical_axis = vertical_axis

    @property
    def dims(self) -> tuple[str, ...]:
        return ("level", *super().dims)

    def calculate_coords(self) -> dict[str, object]:
        coords = super().calculate_coords()
        vcoords = self.vertical_axis.calculate_coords()
        coords["level"] = vcoords["level"]
        coords["height"] = ("level", vcoords["height"])
        if "pressure" in vcoords:
            coords["pressure"] = ("level", vcoords["pressure"])
        if "sigma" in vcoords:
            coords["sigma"] = ("level", vcoords["sigma"])
        return coords
