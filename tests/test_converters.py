"""Tests for arlmet.converters: ERA5 to an ARL Dataset."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import arlmet
from arlmet.converters import from_era5
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
        with pytest.raises(ValueError, match="none of the ERA5 variables"):
            from_era5(pl[["o3"]], sfc)

    def test_not_a_dataset_raises(self, sfc):
        with pytest.raises(TypeError, match="must be an xarray.Dataset"):
            from_era5("era5.nc", sfc)  # pyrefly: ignore[bad-argument-type]


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
