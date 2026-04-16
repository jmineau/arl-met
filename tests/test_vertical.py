"""Tests for explicit vertical helper APIs."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arlmet.vertical import VerticalAxis, interp_vertical, pressure, z_agl, z_msl


def make_pressure_dataset() -> xr.Dataset:
    time = pd.to_datetime(["2024-07-18 00:00"])
    level = [0, 1]
    lat = [40.0]
    lon = [-111.0, -110.0]
    vertical_axis = VerticalAxis(flag=2, levels=[1000.0, 900.0])

    temp = xr.DataArray(
        np.asarray([[[[280.0, 280.0]], [[290.0, 290.0]]]], dtype=np.float32),
        dims=("time", "level", "lat", "lon"),
        coords={"time": time, "level": level, "lat": lat, "lon": lon},
        name="TEMP",
    )
    prss = xr.DataArray(
        np.asarray([[[[1000.0, 950.0]], [[np.nan, np.nan]]]], dtype=np.float32),
        dims=("time", "level", "lat", "lon"),
        coords={"time": time, "level": level, "lat": lat, "lon": lon},
        name="PRSS",
    )
    shgt = xr.DataArray(
        np.asarray([[[[100.0, 200.0]], [[np.nan, np.nan]]]], dtype=np.float32),
        dims=("time", "level", "lat", "lon"),
        coords={"time": time, "level": level, "lat": lat, "lon": lon},
        name="SHGT",
    )

    return xr.Dataset(
        data_vars={"TEMP": temp, "PRSS": prss, "SHGT": shgt},
        attrs={
            "vertical_axis": vertical_axis,
            "arl_vertical_flag": 2,
            "arl_vertical_levels": [1000.0, 900.0],
            "arl_vertical_offset": 0.0,
        },
    )


class TestVerticalHelpers:
    def test_pressure_uses_dataset_vertical_metadata(self):
        ds = make_pressure_dataset()

        result = pressure(ds)

        assert result.dims == ("level",)
        np.testing.assert_allclose(result.values, [1000.0, 900.0])
        assert result.attrs["units"] == "hPa"

    def test_pressure_falls_back_to_serialized_vertical_metadata(self):
        ds = make_pressure_dataset()
        del ds.attrs["vertical_axis"]

        result = pressure(ds)

        np.testing.assert_allclose(result.values, [1000.0, 900.0])

    def test_z_agl_uses_surface_pressure_field(self):
        ds = make_pressure_dataset()

        result = z_agl(ds)

        assert result.name == "z_agl"
        assert result.dims == ("time", "level", "lat", "lon")
        np.testing.assert_allclose(result.isel(level=0), 0.0)
        assert (
            result.isel(level=1, lon=0).item() > result.isel(level=1, lon=1).item()
        )

    def test_z_msl_adds_dataset_terrain(self):
        ds = make_pressure_dataset()

        agl = z_agl(ds)
        msl = z_msl(ds)

        expected = ds["SHGT"].isel(level=0, drop=True).expand_dims(
            level=msl.coords["level"]
        )
        expected = expected.transpose("time", "level", "lat", "lon")
        np.testing.assert_allclose(msl - agl, expected)

    def test_z_msl_requires_terrain_source(self):
        ds = make_pressure_dataset().drop_vars("SHGT")

        with pytest.raises(ValueError, match="Terrain is required"):
            z_msl(ds)

    def test_interp_vertical_interpolates_scalar_pressure_target(self):
        ds = make_pressure_dataset()

        result = interp_vertical(ds["TEMP"], 950.0, coord="pressure", dataset=ds)

        assert result.dims == ("time", "lat", "lon")
        np.testing.assert_allclose(result.values, 285.0)
        assert result.coords["pressure"].item() == 950.0

    def test_helpers_fail_clearly_for_unsupported_vertical_flags(self):
        ds = make_pressure_dataset()
        ds.attrs["vertical_axis"] = VerticalAxis(flag=1, levels=[1.0, 0.9])
        ds.attrs["arl_vertical_flag"] = 1
        ds.attrs["arl_vertical_levels"] = [1.0, 0.9]

        with pytest.raises(NotImplementedError, match="pressure-level"):
            pressure(ds)
