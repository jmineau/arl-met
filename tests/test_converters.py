"""Tests for arlmet.converters: ERA5 to an ARL Dataset."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import arlmet
from arlmet.converters import from_era5, from_gfs, from_wrf
from arlmet.converters._latlon import latlon_grid

TIMES = pd.date_range("2024-07-18 00:00", periods=2, freq="h")
# 25 x 25 cells hold the index record (50 + nx * ny bytes per record).
LATS = np.arange(46.0, 40.0 - 0.01, -0.25)  # north to south, as ERA5 stores them
LONS = np.arange(-116.0, -110.0 + 0.01, 0.25)
PRESSURES = np.array([500.0, 700.0, 850.0, 1000.0])  # as the CDS lists them


def _field(rng, base, scale, shape):
    """A smooth field from ``base`` to ``base + scale``, as packing expects."""
    *outer, ny, nx = shape
    y, x = np.meshgrid(np.linspace(0, 3, ny), np.linspace(0, 2, nx), indexing="ij")
    phase = rng.random((*outer, 1, 1)) * 6.0
    pattern = 0.5 + 0.5 * np.sin(x + phase) * np.cos(y - phase)
    return (base + scale * pattern).astype(np.float32)


def era5_pressure_levels(
    times=TIMES, *, time_dim="valid_time", level_dim="pressure_level"
):
    """Made-up ERA5 pressure-level fields, named as a CDS netCDF download has them."""
    rng = np.random.default_rng(0)
    shape = (len(times), len(PRESSURES), len(LATS), len(LONS))
    dims = (time_dim, level_dim, "latitude", "longitude")
    data = {
        "z": _field(rng, 1.0e4, 5.0e4, shape),
        "t": _field(rng, 250.0, 40.0, shape),
        "u": _field(rng, -10.0, 20.0, shape),
        "v": _field(rng, -10.0, 20.0, shape),
        "w": _field(rng, -0.5, 1.0, shape),
        "r": _field(rng, 0.0, 100.0, shape),
        "q": _field(rng, 0.0, 0.01, shape),
        "o3": _field(rng, 0.0, 1e-6, shape),  # not converted
    }
    return xr.Dataset(
        {name: (dims, values) for name, values in data.items()},
        coords={
            time_dim: times,
            level_dim: PRESSURES,
            "latitude": LATS,
            "longitude": LONS,
            "number": 0,
            "expver": (time_dim, ["0001"] * len(times)),
        },
    )


def era5_single_levels(times=TIMES, *, time_dim="valid_time"):
    """Made-up ERA5 single-level fields (instantaneous and accumulated merged)."""
    rng = np.random.default_rng(1)
    shape = (len(times), len(LATS), len(LONS))
    dims = (time_dim, "latitude", "longitude")
    data = {
        "sp": _field(rng, 8.0e4, 2.0e4, shape),
        "z": _field(rng, 1.0e4, 2.0e4, shape),
        "t2m": _field(rng, 280.0, 20.0, shape),
        "d2m": _field(rng, 270.0, 10.0, shape),
        "u10": _field(rng, -5.0, 10.0, shape),
        "v10": _field(rng, -5.0, 10.0, shape),
        "blh": _field(rng, 100.0, 2000.0, shape),
        "cape": _field(rng, 0.0, 500.0, shape),
        "tcc": _field(rng, 0.0, 1.0, shape),
        "tp": _field(rng, 0.0, 0.002, shape),
        "sshf": _field(rng, -1.0e6, 1.2e6, shape),
        "slhf": _field(rng, -1.0e6, 1.0e6, shape),
        "ssrd": _field(rng, 0.0, 3.0e6, shape),
        "zust": _field(rng, 0.0, 1.0, shape),
    }
    return xr.Dataset(
        {name: (dims, values) for name, values in data.items()},
        coords={time_dim: times, "latitude": LATS, "longitude": LONS},
    )


@pytest.fixture
def pl():
    return era5_pressure_levels()


@pytest.fixture
def sfc():
    return era5_single_levels()


class TestFromEra5:
    def test_layout_is_open_datasets(self, pl, sfc):
        ds = from_era5(pl, sfc)
        assert ds.attrs["source"] == "ERA5"
        assert ds.attrs["vertical_flag"] == 2
        assert ds["TEMP"].dims == ("time", "level", "lat", "lon")
        assert ds["PRSS"].dims == ("time", "lat", "lon")
        assert ds["level"].values.tolist() == [1, 2, 3, 4]
        # Levels from the highest pressure up; the surface is 0.
        assert ds["pressure"].values.tolist() == [1000.0, 850.0, 700.0, 500.0]
        assert ds.arl.vertical_axis.levels.tolist() == [
            0.0,
            1000.0,
            850.0,
            700.0,
            500.0,
        ]
        assert ds["forecast_hour"].values.tolist() == [0, 0]
        assert "o3" not in ds and "O3" not in ds
        assert set(ds.data_vars) == {
            "HGTS", "TEMP", "UWND", "VWND", "WWND", "RELH", "SPHU",
            "PRSS", "SHGT", "T02M", "DP2M", "U10M", "V10M", "PBLH",
            "CAPE", "TCLD", "TPP1", "SHTF", "LHTF", "DSWF", "USTR",
        }  # fmt: skip

    def test_grid_runs_south_to_north_and_east(self, pl, sfc):
        ds = from_era5(pl, sfc)
        grid = ds.arl.grid
        assert grid.is_latlon
        assert (grid.nx, grid.ny) == (25, 25)
        proj = grid.projection
        assert proj.sync_lat == 40.0
        assert proj.sync_lon == 244.0  # -116, east of Greenwich
        assert proj.pole_lat == 46.0
        assert proj.pole_lon == 250.0
        assert (proj.tangent_lat, proj.tangent_lon) == (0.25, 0.25)
        np.testing.assert_allclose(ds["lat"].values, LATS[::-1])
        np.testing.assert_allclose(ds["lon"].values, LONS)

    def test_values_are_flipped_reordered_and_converted(self, pl, sfc):
        ds = from_era5(pl, sfc)
        # Output row 0 is ERA5's last (southernmost) row; level 1 is 1000 hPa.
        np.testing.assert_allclose(
            ds["HGTS"].isel(time=1, level=0).values,
            pl["z"].isel(valid_time=1, pressure_level=3).values[::-1] / 9.80665,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            ds["TEMP"].isel(time=0, level=3).values,
            pl["t"].isel(valid_time=0, pressure_level=0).values[::-1],
        )
        np.testing.assert_allclose(
            ds["WWND"].values, pl["w"].values[:, ::-1, ::-1] * 0.01, rtol=1e-6
        )
        flip = (slice(None), slice(None, None, -1))
        np.testing.assert_allclose(
            ds["PRSS"].values, sfc["sp"].values[flip] * 0.01, rtol=1e-6
        )
        np.testing.assert_allclose(
            ds["TCLD"].values, sfc["tcc"].values[flip] * 100, rtol=1e-6
        )
        np.testing.assert_allclose(ds["TPP1"].values, sfc["tp"].values[flip])

    def test_fluxes_are_positive_upward_per_second(self, pl, sfc):
        ds = from_era5(pl, sfc)
        flip = (slice(None), slice(None, None, -1))
        np.testing.assert_allclose(
            ds["SHTF"].values, -sfc["sshf"].values[flip] / 3600.0, rtol=1e-6
        )
        np.testing.assert_allclose(
            ds["LHTF"].values, -sfc["slhf"].values[flip] / 3600.0, rtol=1e-6
        )
        np.testing.assert_allclose(
            ds["DSWF"].values, sfc["ssrd"].values[flip] / 3600.0, rtol=1e-6
        )

    def test_units_and_dif_records(self, pl, sfc):
        ds = from_era5(pl, sfc)
        assert ds["WWND"].attrs == {"units": "hPa s-1", "diff": "DIFW"}
        assert ds["TPP1"].attrs == {"units": "m", "diff": "DIFR"}
        assert ds["HGTS"].attrs == {"units": "m"}

    def test_written_file_holds_the_fields(self, pl, sfc, tmp_path):
        ds = from_era5(pl, sfc)
        dest = tmp_path / "era5.arl"
        arlmet.write_dataset(ds, dest)
        back = arlmet.open_dataset(dest)
        assert back.arl.grid == ds.arl.grid
        assert back.arl.vertical_axis.levels.tolist() == [
            0.0,
            1000.0,
            850.0,
            700.0,
            500.0,
        ]
        pd.testing.assert_index_equal(back.indexes["time"], ds.indexes["time"])
        for name in ("TEMP", "UWND", "PRSS", "SHTF", "TPP1"):
            # Packing keeps about 1/1000 of a field's range.
            span = float(ds[name].max() - ds[name].min())
            np.testing.assert_allclose(
                back[name].values, ds[name].values, rtol=0, atol=1e-3 * span
            )
        # WWND with its DIFW record reads back closer than packing alone.
        np.testing.assert_allclose(back["WWND"].values, ds["WWND"].values, atol=1e-6)
        with arlmet.File(dest) as met:
            records = met[met.times[0]]
            assert [d.variable for d in records[(1, "WWND")].diffs] == ["DIFW"]
            assert [d.variable for d in records[(0, "TPP1")].diffs] == ["DIFR"]

    def test_lazy_inputs_stay_lazy(self, pl, sfc):
        ds = from_era5(pl.chunk({"valid_time": 1}), sfc.chunk({"valid_time": 1}))
        assert ds["TEMP"].chunks is not None
        assert ds["WWND"].chunks is not None

    def test_old_cds_and_cfgrib_names(self, sfc):
        expected = from_era5(era5_pressure_levels(), sfc)
        old = era5_pressure_levels(time_dim="time", level_dim="level")
        cfgrib = era5_pressure_levels(time_dim="time", level_dim="isobaricInhPa")
        for upper in (old, cfgrib):
            ds = from_era5(upper, era5_single_levels(time_dim="time"))
            xr.testing.assert_identical(ds, expected)

    def test_levels_in_either_order(self, pl, sfc):
        expected = from_era5(pl, sfc)
        flipped = pl.isel(pressure_level=slice(None, None, -1))
        xr.testing.assert_identical(from_era5(flipped, sfc), expected)

    def test_one_time_step_as_a_scalar(self, pl, sfc):
        ds = from_era5(pl.isel(valid_time=0), sfc.isel(valid_time=0))
        assert ds.sizes["time"] == 1
        assert ds["TEMP"].dims == ("time", "level", "lat", "lon")

    def test_surface_times_follow_the_pressure_levels(self, pl):
        more = pd.date_range(TIMES[0] - pd.Timedelta("1h"), periods=4, freq="h")
        sfc = era5_single_levels(more)
        ds = from_era5(pl, sfc)
        pd.testing.assert_index_equal(
            ds.indexes["time"],
            pd.DatetimeIndex(TIMES, name="time"),
            check_freq=False,
        )
        np.testing.assert_allclose(
            ds["T02M"].isel(time=0).values, sfc["t2m"].isel(valid_time=1).values[::-1]
        )

    def test_missing_surface_time_raises(self, pl):
        with pytest.raises(ValueError, match="lacks 1 valid time"):
            from_era5(pl, era5_single_levels(TIMES[:1]))

    def test_different_grids_raise(self, pl, sfc):
        with pytest.raises(ValueError, match="different grids"):
            from_era5(pl, sfc.isel(longitude=slice(1, None)))

    def test_ensemble_dimension_raises(self, pl, sfc):
        members = xr.concat([pl, pl], dim="number")
        with pytest.raises(ValueError, match=r"\['number'\]"):
            from_era5(members, sfc)

    def test_forecast_step_dimension_raises(self, pl, sfc):
        steps = sfc.expand_dims(step=[1, 2])
        with pytest.raises(ValueError, match="'step' dimension"):
            from_era5(pl, steps)

    def test_no_known_variables_raises(self, pl, sfc):
        with pytest.raises(ValueError, match="none of the variables"):
            from_era5(pl[["o3"]], sfc)

    def test_not_a_dataset_raises(self, sfc):
        with pytest.raises(TypeError, match="must be an xarray.Dataset"):
            from_era5("era5.nc", sfc)  # pyrefly: ignore[bad-argument-type]


MODEL_LEVELS = np.array([100, 120, 130, 137])  # top down, as the CDS lists them


def era5_model_levels(levels=MODEL_LEVELS, *, level_dim="model_level"):
    """Made-up ERA5 model-level fields, named as a CDS netCDF download has them."""
    rng = np.random.default_rng(2)
    shape = (len(TIMES), len(levels), len(LATS), len(LONS))
    dims = ("valid_time", level_dim, "latitude", "longitude")
    data = {
        "t": _field(rng, 250.0, 40.0, shape),
        "u": _field(rng, -10.0, 20.0, shape),
        "v": _field(rng, -10.0, 20.0, shape),
        "w": _field(rng, -0.5, 1.0, shape),
        "q": _field(rng, 0.0, 0.01, shape),
        "clwc": _field(rng, 0.0, 1e-4, shape),  # not converted
    }
    return xr.Dataset(
        {name: (dims, values) for name, values in data.items()},
        coords={
            "valid_time": TIMES,
            level_dim: levels,
            "latitude": LATS,
            "longitude": LONS,
        },
    )


def _nint(x):
    """Fortran's NINT: the nearest whole number, halves away from zero."""
    return float(np.sign(x) * np.floor(abs(x) + 0.5))


def api2arl_levels(pv):
    """
    Return HYSPLIT's ``api2arl`` levels from the lowest model level up.

    Its ECMWF hybrid section, transcribed: ``pv`` is a GRIB message's ``a``
    then ``b`` of every half level, top down.
    """
    n = len(pv) // 2
    height = [0.0, *pv]  # Fortran's height(1:2n)
    sigmaz = [0.0] * (n + 1)
    for k in range(n, 1, -1):
        offset = _nint(0.5 * (height[k] + height[k - 1]) / 100.0)
        sigmav = 0.5 * (height[k + n] + height[k + n - 1])
        sigmaz[n - k + 1] = offset + sigmav
    return sigmaz[1:n]


class TestFromEra5ModelLevels:
    def test_l137_coefficients_are_ecmwfs(self):
        from arlmet.converters._l137 import L137_A, L137_B

        assert len(L137_A) == len(L137_B) == 138
        # ECMWF's L137 table: half levels 0, 1, 86 and the last two.
        assert (L137_A[0], L137_A[1], L137_A[86]) == (0.0, 2.000365, 20442.078125)
        assert (L137_B[0], L137_B[136], L137_B[137]) == (0.0, 0.99763, 1.0)
        assert L137_A[137] == 0.0
        assert L137_B[54] == 0.0 < L137_B[55]
        # Half-level pressures fall from the ground up at any surface pressure.
        for sp in (50000.0, 101325.0):
            p = np.asarray(L137_A) + np.asarray(L137_B) * sp
            assert np.all(np.diff(p) > 0)

    def test_levels_are_api2arls(self):
        from arlmet.converters._l137 import L137_A, L137_B
        from arlmet.converters.era5 import _hybrid_levels

        expected = api2arl_levels([*L137_A, *L137_B])
        assert len(expected) == 137
        assert _hybrid_levels(range(137, 0, -1)) == expected

    def test_layout_is_a_hybrid_open_dataset(self, sfc):
        ds = from_era5(era5_model_levels(), sfc)
        assert ds.attrs["source"] == "ERA5"
        assert ds.attrs["vertical_flag"] == 4
        assert ds["TEMP"].dims == ("time", "level", "lat", "lon")
        assert ds["level"].values.tolist() == [1, 2, 3, 4]
        axis = ds.arl.vertical_axis
        assert isinstance(axis, arlmet.HybridAxis)
        # The ground first as 1, then model levels 137, 130, 120, 100.
        from arlmet.converters.era5 import _hybrid_levels

        np.testing.assert_allclose(
            axis.levels[1:], _hybrid_levels([137, 130, 120, 100]), atol=5e-3
        )
        assert axis.levels[0] == 1.0
        assert set(ds.data_vars) == {
            "TEMP", "UWND", "VWND", "WWND", "SPHU",
            "PRSS", "SHGT", "T02M", "DP2M", "U10M", "V10M", "PBLH",
            "CAPE", "TCLD", "TPP1", "SHTF", "LHTF", "DSWF", "USTR",
        }  # fmt: skip

    def test_values_from_the_lowest_level_up(self, sfc):
        ml = era5_model_levels()
        ds = from_era5(ml, sfc)
        # Level 1 is model level 137 (ERA5's last); rows south to north.
        np.testing.assert_allclose(
            ds["TEMP"].isel(time=1, level=0).values,
            ml["t"].isel(valid_time=1, model_level=3).values[::-1],
        )
        np.testing.assert_allclose(
            ds["WWND"].values, ml["w"].values[:, ::-1, ::-1] * 0.01, rtol=1e-6
        )
        assert ds["WWND"].attrs == {"units": "hPa s-1", "diff": "DIFW"}
        np.testing.assert_allclose(ds["SPHU"].values, ml["q"].values[:, ::-1, ::-1])

    def test_cfgrib_names(self, sfc):
        expected = from_era5(era5_model_levels(), sfc)
        cfgrib = era5_model_levels(level_dim="hybrid")
        xr.testing.assert_identical(from_era5(cfgrib, sfc), expected)

    def test_written_file_is_hybrid_and_reads_back(self, sfc, tmp_path):
        ds = from_era5(era5_model_levels(), sfc)
        dest = tmp_path / "era5_ml.arl"
        arlmet.write_dataset(ds, dest)
        with arlmet.File(dest) as met:
            axis = met.vertical_axis
            assert isinstance(axis, arlmet.HybridAxis)
            # Stored as api2arl's six characters: model level 137 is .99882.
            assert axis.levels.tolist()[:2] == [1.0, 0.99882]
            # HYSPLIT's pressure of level 137 at 1013.25 hPa, as ERA5's within
            # the rounding of the stored sigma.
            p = axis.to_pressure(surface_pressure=1013.25)
            assert abs(p[1] - (0.5 * (1.0 + 0.99763) * 1013.25)) < 0.01
        back = arlmet.open_dataset(dest)
        np.testing.assert_allclose(back["WWND"].values, ds["WWND"].values, atol=1e-6)

    def test_top_levels_that_round_together_raise(self, sfc):
        ml = era5_model_levels(np.arange(1, 138))
        with pytest.raises(ValueError, match="model level 23 is stored as the same"):
            from_era5(ml, sfc)

    def test_lowest_usable_top_converts(self):
        from arlmet.converters.era5 import _hybrid_axis

        axis = _hybrid_axis(list(range(137, 23, -1)))
        assert len(axis.levels) == 115

    def test_level_outside_l137_raises(self, sfc):
        with pytest.raises(ValueError, match="1 to 137, got 138"):
            from_era5(era5_model_levels(np.array([130, 138])), sfc)

    def test_no_surface_pressure_raises(self, sfc):
        with pytest.raises(ValueError, match="no surface pressure 'sp'"):
            from_era5(era5_model_levels(), sfc.drop_vars("sp"))


class TestLatlonGrid:
    def test_wraps_across_greenwich(self):
        grid = latlon_grid([0.0, 1.0], [350.0, 355.0, 0.0, 5.0])
        assert grid.projection.sync_lon == 350.0
        assert grid.projection.pole_lon == 365.0
        assert grid.projection.tangent_lon == 5.0
        np.testing.assert_allclose(grid.calculate_coords()["lon"][1], [-10, -5, 0, 5])

    def test_global_grid(self):
        lons = np.arange(0.0, 360.0, 0.25)
        grid = latlon_grid(np.arange(-90.0, 90.01, 0.25), lons)
        assert grid.wraps_lon
        assert (grid.nx, grid.ny) == (1440, 721)

    @pytest.mark.parametrize(
        ("lat", "lon", "match"),
        [
            ([0.0], [0.0, 1.0], "at least two latitude"),
            ([1.0, 0.0], [0.0, 1.0], "Latitude values must increase"),
            ([0.0, 1.0, 3.0], [0.0, 1.0], "Latitude values must increase"),
            ([0.0, 1.0], [0.0, 1.0, 2.5], "Longitude values must increase"),
            ([0.0, 1.0], np.arange(0.0, 361.0, 1.0), "more than 360"),
        ],
    )
    def test_irregular_values_raise(self, lat, lon, match):
        with pytest.raises(ValueError, match=match):
            latlon_grid(lat, lon)


# ---- GFS / GDAS GRIB2, as cfgrib.open_datasets() returns them ----------------

GFS_LATS = np.arange(45.0, 39.0 - 0.01, -0.25)  # north to south, as GRIB stores them
GFS_LONS = np.arange(250.0, 256.0 + 0.01, 0.25)
GFS_PRESSURES = np.array([1000.0, 850.0, 500.0])


def _cube(variables, valid_time, step, level_coord, *, step_type="instant"):
    """One cfgrib Dataset: scalar time, step and valid_time, rows north to south."""
    valid_time = pd.Timestamp(valid_time)
    step = pd.Timedelta(hours=step)
    type_of_level, level = level_coord
    coords = {
        "time": valid_time - step,
        "step": step,
        "valid_time": valid_time,
        "latitude": GFS_LATS,
        "longitude": GFS_LONS,
    }
    if np.ndim(level):
        coords[type_of_level] = level
    else:
        coords[type_of_level] = float(level)
    attrs = {"GRIB_typeOfLevel": type_of_level, "GRIB_stepType": step_type}
    return xr.Dataset(
        {name: (dims, values, attrs) for name, (dims, values) in variables.items()},
        coords=coords,
    )


def gfs_file(valid_time, step, *, seed=0, levels=GFS_PRESSURES, old_names=False):
    """The cfgrib Datasets of one made-up GFS file; accumulations unless an analysis."""
    rng = np.random.default_rng(seed)
    ny, nx = len(GFS_LATS), len(GFS_LONS)
    up = (len(levels), ny, nx)
    upper_dims = ("isobaricInhPa", "latitude", "longitude")
    flat = ("latitude", "longitude")

    def f(base, scale, shape=(ny, nx)):
        return _field(rng, base, scale, shape)

    cubes = [
        _cube(
            {
                "gh": (upper_dims, f(100.0, 5000.0, up)),
                "t": (upper_dims, f(250.0, 40.0, up)),
                "u": (upper_dims, f(-10.0, 20.0, up)),
                "v": (upper_dims, f(-10.0, 20.0, up)),
                "w": (upper_dims, f(-0.5, 1.0, up)),
                "r": (upper_dims, f(0.0, 100.0, up)),
                "absv": (upper_dims, f(0.0, 1e-4, up)),  # not converted
            },
            valid_time,
            step,
            ("isobaricInhPa", levels),
        ),
        # cfgrib's second pressure-level cube (cloud fields): not converted.
        _cube({"tcc": (upper_dims, f(0.0, 100.0, up))}, valid_time, step,
              ("isobaricInhPa", levels)),
        _cube(
            {
                "sp": (flat, f(8.0e4, 2.0e4)),
                "orog": (flat, f(1000.0, 2000.0)),
                "t": (flat, f(270.0, 30.0)),
                "cape": (flat, f(0.0, 500.0)),
            },
            valid_time,
            step,
            ("surface", 0.0),
        ),
        _cube({"t2m": (flat, f(280.0, 20.0)), "r2": (flat, f(10.0, 80.0))},
              valid_time, step, ("heightAboveGround", 2.0)),
        _cube({"u10": (flat, f(-5.0, 10.0)), "v10": (flat, f(-5.0, 10.0))},
              valid_time, step, ("heightAboveGround", 10.0)),
        # Temperature at 80 and 100 m: a level dimension, not converted.
        _cube({"t": (("heightAboveGround", *flat), f(270.0, 30.0, (2, ny, nx)))},
              valid_time, step, ("heightAboveGround", np.array([80.0, 100.0]))),
        _cube({"tcc": (flat, f(0.0, 100.0))}, valid_time, step, ("atmosphere", 0.0)),
        _cube({"prmsl": (flat, f(1.0e5, 3.0e3))}, valid_time, step, ("meanSea", 0.0)),
    ]  # fmt: skip
    if step:
        shtf, dswf = ("shtfl", "dswrf") if old_names else ("avg_ishf", "sdswrf")
        cubes += [
            _cube({"tp": (flat, f(0.0, 5.0))}, valid_time, step, ("surface", 0.0),
                  step_type="accum"),
            _cube({shtf: (flat, f(-50.0, 300.0)), dswf: (flat, f(0.0, 800.0))},
                  valid_time, step, ("surface", 0.0), step_type="avg"),
            # The averaged cloud cover is not TCLD (the instantaneous one is).
            _cube({"tcc": (flat, f(0.0, 100.0))}, valid_time, step,
                  ("atmosphere", 0.0), step_type="avg"),
        ]  # fmt: skip
    return cubes


def _by_name(cubes, name, type_of_level, *, step_type="instant"):
    for cube in cubes:
        attrs = cube[name].attrs if name in cube else {}
        if (attrs.get("GRIB_typeOfLevel"), attrs.get("GRIB_stepType")) == (
            type_of_level,
            step_type,
        ):
            return cube[name]
    raise KeyError(name)


class TestFromGfs:
    def test_analysis(self):
        cubes = gfs_file("2025-09-01 00:00", 0)
        ds = from_gfs(cubes)
        assert ds.attrs["source"] == "GFS"
        assert ds.attrs["vertical_flag"] == 2
        assert ds.sizes["time"] == 1
        assert ds["forecast_hour"].values.tolist() == [0]
        assert ds["pressure"].values.tolist() == [1000.0, 850.0, 500.0]
        assert set(ds.data_vars) == {
            "HGTS", "TEMP", "UWND", "VWND", "WWND", "RELH",
            "PRSS", "SHGT", "TMPS", "CAPE", "T02M", "RH2M", "U10M", "V10M",
            "TCLD", "MSLP",
        }  # fmt: skip
        flip = slice(None, None, -1)
        np.testing.assert_allclose(
            ds["PRSS"].isel(time=0).values,
            _by_name(cubes, "sp", "surface").values[flip] * 0.01,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            ds["WWND"].isel(time=0).values,
            _by_name(cubes, "w", "isobaricInhPa").values[:, flip] * 0.01,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            ds["TMPS"].isel(time=0).values,
            _by_name(cubes, "t", "surface").values[flip],
        )
        grid = ds.arl.grid.projection
        assert (grid.sync_lat, grid.sync_lon, grid.pole_lon) == (39.0, 250.0, 256.0)
        assert all("diff" not in ds[name].attrs for name in ds.data_vars)

    def test_analysis_takes_fluxes_from_the_previous_forecast(self):
        analysis = gfs_file("2025-09-01 00:00", 0, seed=0)
        forecast = gfs_file("2025-09-01 00:00", 6, seed=1)
        ds = from_gfs([*forecast, *analysis])
        flip = slice(None, None, -1)
        assert ds["forecast_hour"].values.tolist() == [0]
        # Instantaneous fields from the analysis, the bucket's from the forecast.
        np.testing.assert_allclose(
            ds["TEMP"].isel(time=0).values,
            _by_name(analysis, "t", "isobaricInhPa").values[:, flip],
        )
        np.testing.assert_allclose(
            ds["TCLD"].isel(time=0).values,
            _by_name(analysis, "tcc", "atmosphere").values[flip],
        )
        np.testing.assert_allclose(
            ds["TPP6"].isel(time=0).values,
            _by_name(forecast, "tp", "surface", step_type="accum").values[flip] * 0.001,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            ds["SHTF"].isel(time=0).values,
            _by_name(forecast, "avg_ishf", "surface", step_type="avg").values[flip],
        )
        assert ds["TPP6"].attrs == {"units": "m"}

    def test_forecast_run(self):
        cubes = [
            *gfs_file("2025-09-01 06:00", 6, seed=2),
            *gfs_file("2025-09-01 03:00", 3, seed=1),
        ]
        ds = from_gfs(cubes)
        assert ds.indexes["time"].tolist() == [
            pd.Timestamp("2025-09-01 03:00"),
            pd.Timestamp("2025-09-01 06:00"),
        ]
        assert ds["forecast_hour"].values.tolist() == [3, 6]
        assert np.isfinite(ds["TPP6"].values).all()

    def test_time_step_without_a_field_is_written_missing(self, tmp_path):
        cubes = [
            *gfs_file("2025-09-01 00:00", 0),
            *gfs_file("2025-09-01 03:00", 3, seed=1),
        ]
        ds = from_gfs(cubes)
        assert np.isnan(ds["TPP6"].isel(time=0).values).all()
        assert np.isfinite(ds["TPP6"].isel(time=1).values).all()
        dest = tmp_path / "gfs.arl"
        arlmet.write_dataset(ds, dest)
        back = arlmet.open_dataset(dest)
        assert np.isnan(back["TPP6"].isel(time=0).values).all()
        np.testing.assert_allclose(
            back["TPP6"].isel(time=1).values,
            ds["TPP6"].isel(time=1).values,
            atol=1e-3 * float(ds["TPP6"].max()),
        )
        assert back["forecast_hour"].values.tolist() == [0, 3]

    def test_older_eccodes_names(self):
        ds = from_gfs(gfs_file("2025-09-01 06:00", 6, old_names=True))
        assert {"SHTF", "DSWF"} <= set(ds.data_vars)

    def test_different_levels_raise(self):
        cubes = [
            *gfs_file("2025-09-01 00:00", 0),
            *gfs_file("2025-09-01 03:00", 3, levels=np.array([1000.0, 700.0, 500.0])),
        ]
        with pytest.raises(ValueError, match="different grids or pressure levels"):
            from_gfs(cubes)

    def test_no_pressure_levels_raises(self):
        surface_only = [
            c
            for c in gfs_file("2025-09-01 00:00", 0)
            if "isobaricInhPa" not in c.coords
        ]
        with pytest.raises(ValueError, match="pressure-level fields"):
            from_gfs(surface_only)

    def test_no_scalar_valid_time_raises(self):
        cubes = gfs_file("2025-09-01 00:00", 0)
        stacked = xr.concat([cubes[0], cubes[0]], dim="valid_time")
        with pytest.raises(ValueError, match="scalar 'valid_time'"):
            from_gfs([stacked])

    def test_not_a_dataset_raises(self):
        with pytest.raises(TypeError, match="takes xarray.Datasets"):
            from_gfs(["gdas.t00z.pgrb2.1p00.f000"])  # pyrefly: ignore[bad-argument-type]


# ---- WRF-ARW output, as xarray opens wrfout files ----------------------------

WRF_ATTRS = {
    "MAP_PROJ": 1,
    "CEN_LAT": 40.0,
    "CEN_LON": -111.0,
    "STAND_LON": -111.0,
    "TRUELAT1": 30.0,
    "TRUELAT2": 60.0,
    "DX": 12000.0,
    "DY": 12000.0,
    "HYBRID_OPT": 0,
    "START_DATE": "2013-06-29_18:00:00",
}


def wrf_output(minutes=(360.0, 420.0, 480.0), *, nx=25, ny=20, nz=3, attrs=None):
    """Made-up wrfout fields, joined along Time, on WRF's staggered dimensions."""
    rng = np.random.default_rng(3)
    nt = len(minutes)
    start = pd.Timestamp("2013-06-29 18:00")
    times = [
        (start + pd.Timedelta(minutes=m)).strftime("%Y-%m-%d_%H:%M:%S").encode()
        for m in minutes
    ]
    mass = ("Time", "bottom_top", "south_north", "west_east")
    flat = ("Time", "south_north", "west_east")

    def f(base, scale, shape):
        return _field(rng, base, scale, shape)

    m3, m2 = (nt, nz, ny, nx), (nt, ny, nx)
    rain = np.cumsum(f(0.0, 2.0, m2), axis=0)  # accumulated, mm
    data = {
        "Times": (("Time",), np.array(times, dtype="S19")),
        "XTIME": (("Time",), np.asarray(minutes, dtype=np.float32)),
        "ZNU": (("Time", "bottom_top"), np.tile([0.99, 0.95, 0.85][:nz], (nt, 1))),
        "P_TOP": (("Time",), np.full(nt, 5000.0, dtype=np.float32)),
        "P": (mass, f(-500.0, 1000.0, m3)),
        "PB": (mass, np.broadcast_to(np.array([95000.0, 90000.0, 80000.0][:nz])[:, None, None], m3).astype(np.float32)),
        "T": (mass, f(-10.0, 20.0, m3)),
        "QVAPOR": (mass, f(0.0, 0.01, m3)),
        "U": (("Time", "bottom_top", "south_north", "west_east_stag"), f(-10.0, 20.0, (nt, nz, ny, nx + 1))),
        "V": (("Time", "bottom_top", "south_north_stag", "west_east"), f(-10.0, 20.0, (nt, nz, ny + 1, nx))),
        "W": (("Time", "bottom_top_stag", "south_north", "west_east"), f(-1.0, 2.0, (nt, nz + 1, ny, nx))),
        "HGT": (flat, f(1000.0, 2000.0, m2)),
        "PSFC": (flat, f(80000.0, 20000.0, m2)),
        "RAINC": (flat, 0.3 * rain),
        "RAINNC": (flat, 0.7 * rain),
        "PBLH": (flat, f(100.0, 2000.0, m2)),
        "HFX": (flat, f(-50.0, 300.0, m2)),
        "T2": (flat, f(280.0, 20.0, m2)),
    }  # fmt: skip
    return xr.Dataset(
        {name: (dims, np.asarray(values)) for name, (dims, values) in data.items()},
        attrs={**WRF_ATTRS, **(attrs or {})},
    )


class TestFromWrf:
    def test_layout(self):
        wrf = wrf_output()
        ds = from_wrf(wrf)
        assert ds.attrs["source"] == "AWRF"
        assert ds.attrs["vertical_flag"] == 1
        assert ds["TEMP"].dims == ("time", "level", "y", "x")
        assert ds["PRSS"].dims == ("time", "y", "x")
        axis = ds.arl.vertical_axis
        assert isinstance(axis, arlmet.SigmaAxis)
        assert axis.levels.tolist() == [1.0, 0.99, 0.95, 0.85]
        assert axis.offset == 50.0  # P_TOP in hPa
        assert ds["forecast_hour"].values.tolist() == [6, 7, 8]
        assert ds.indexes["time"][0] == pd.Timestamp("2013-06-30 00:00")
        assert set(ds.data_vars) == {
            "SHGT", "PRSS", "TPP1", "PBLH", "SHTF", "T02M",
            "PRES", "TEMP", "UWND", "VWND", "WWND", "SPHU",
        }  # fmt: skip
        assert ds["WWND"].attrs == {"units": "hPa s-1", "diff": "DIFW"}
        assert ds["TPP1"].attrs == {"units": "m", "diff": "DIFR"}

    def test_grid_as_setmap(self):
        grid = from_wrf(wrf_output()).arl.grid
        proj = grid.projection
        assert (grid.nx, grid.ny) == (25, 20)
        assert proj.cone_angle == pytest.approx(45.6897, abs=1e-4)
        assert proj.pole_lat == proj.cone_angle
        assert (proj.pole_lon, proj.tangent_lat, proj.tangent_lon) == (
            -111.0,
            30.0,
            -111.0,
        )
        assert proj.grid_size == 12.0
        assert (proj.sync_x, proj.sync_y) == (13.0, 10.5)  # the domain's center
        assert (proj.sync_lat, proj.sync_lon) == (40.0, -111.0)

    def test_fields_as_arw2arl(self):
        wrf = wrf_output()
        ds = from_wrf(wrf)
        pressure = wrf["P"].values + wrf["PB"].values
        temperature = (wrf["T"].values + 300.0) * (pressure / 1e5) ** (287.04 / 1004.0)
        np.testing.assert_allclose(ds["PRES"].values, pressure / 100, rtol=1e-6)
        np.testing.assert_allclose(ds["TEMP"].values, temperature, rtol=1e-6)
        # Staggered winds keep their west and south faces; W the bottom of each layer.
        np.testing.assert_array_equal(ds["UWND"].values, wrf["U"].values[..., :-1])
        np.testing.assert_array_equal(ds["VWND"].values, wrf["V"].values[:, :, :-1])
        omega = (
            -wrf["W"].values[:, :-1] * 9.81 * pressure * 0.01
            / (287.04 * temperature * (1 + 0.6077 * wrf["QVAPOR"].values))
        )  # fmt: skip
        np.testing.assert_allclose(ds["WWND"].values, omega, rtol=1e-5, atol=1e-9)
        np.testing.assert_allclose(
            ds["PRSS"].values, wrf["PSFC"].values / 100, rtol=1e-6
        )

    def test_precipitation_per_hour_since_the_step_before(self):
        wrf = wrf_output(minutes=(360.0, 390.0, 420.0))  # every 30 minutes
        ds = from_wrf(wrf)
        rain = (wrf["RAINC"] + wrf["RAINNC"]).values * 0.001
        assert (ds["TPP1"].isel(time=0).values == 0).all()
        np.testing.assert_allclose(
            ds["TPP1"].isel(time=slice(1, None)).values,
            np.diff(rain, axis=0) * 2.0,  # scaled to one hour
            rtol=1e-5,
            atol=1e-9,
        )

    def test_written_file_reads_back(self, tmp_path):
        ds = from_wrf(wrf_output())
        dest = tmp_path / "wrf.arl"
        arlmet.write_dataset(ds, dest)
        with arlmet.File(dest) as met:
            assert met.source == "AWRF"
            assert met.grid == ds.arl.grid
            assert met.vertical_axis == ds.arl.vertical_axis
            records = met[met.times[1]]
            assert [d.variable for d in records[(1, "WWND")].diffs] == ["DIFW"]
            assert [d.variable for d in records[(0, "TPP1")].diffs] == ["DIFR"]
        back = arlmet.open_dataset(dest)
        span = float(ds["TEMP"].max() - ds["TEMP"].min())
        np.testing.assert_allclose(
            back["TEMP"].values, ds["TEMP"].values, atol=1e-3 * span
        )

    def test_hybrid_coordinate_is_flag_5(self):
        ds = from_wrf(wrf_output(attrs={"HYBRID_OPT": 2}))
        assert ds.attrs["vertical_flag"] == 5
        assert isinstance(ds.arl.vertical_axis, arlmet.WrfAxis)

    def test_one_time_step_and_decoded_xtime(self):
        wrf = wrf_output().isel(Time=1)
        start = pd.Timestamp("2013-06-29 18:00")
        wrf["XTIME"] = start + pd.Timedelta(minutes=420)  # as xarray decodes it
        ds = from_wrf(wrf)
        assert ds.sizes["time"] == 1
        assert ds["forecast_hour"].values.tolist() == [7]

    def test_polar_stereographic_and_mercator(self):
        polar = from_wrf(wrf_output(attrs={"MAP_PROJ": 2, "TRUELAT2": 30.0}))
        assert polar.arl.grid.projection.cone_angle == 90.0
        mercator = from_wrf(wrf_output(attrs={"MAP_PROJ": 3}))
        assert mercator.arl.grid.projection.cone_angle == 0.0

    @pytest.mark.parametrize(
        ("attrs", "match"),
        [
            ({"MAP_PROJ": 6}, "MAP_PROJ 6 is not converted"),
            ({"MAP_PROJ": 2}, "TRUELAT1 equal to TRUELAT2"),
        ],
    )
    def test_unsupported_grids_raise(self, attrs, match):
        with pytest.raises(ValueError, match=match):
            from_wrf(wrf_output(attrs=attrs))

    def test_missing_variables_raise(self):
        with pytest.raises(ValueError, match="lacks PB, W"):
            from_wrf(wrf_output().drop_vars(["PB", "W"]))

    def test_not_a_dataset_raises(self):
        with pytest.raises(TypeError, match="takes an xarray.Dataset"):
            from_wrf("wrfout_d01")  # pyrefly: ignore[bad-argument-type]
