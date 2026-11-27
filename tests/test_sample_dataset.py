"""
sample_points on a Dataset in open_dataset's layout (#239).

A Dataset is sampled as the ARL file write_dataset would make from it, bit for
bit. The Datasets here are open_dataset's of the files test_sample.py writes;
the comparison is with the file write_dataset writes from each, not with the
original file, which may hold what a Dataset cannot (an upper-air variable at
the ground too).
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import arlmet
from arlmet import File, sample_points
from arlmet.vertical import SigmaAxis

from .test_sample import (
    TIME0,
    TIME1,
    make_lambert_grid,
    write_coupled_wrf_file,
    write_deep_sampling_file,
    write_global_file,
    write_hilly_file,
    write_hybrid_sampling_file,
    write_pressure_column_file,
    write_sampling_file,
    write_sigma_sampling_file,
    write_terrain_sampling_file,
    write_two_time_file,
    write_varying_wrf_file,
    write_wind_file,
)
from .test_vertical import _write_column_file

T = pd.Timestamp("2024-07-18 00:00")

#: name -> writer(path); each writes one of test_sample.py's files.
WRITERS = {
    "pressure": lambda p: write_sampling_file(p, time=T),
    "deep-pressure": lambda p: write_deep_sampling_file(p, time=T),
    "sigma": lambda p: write_sigma_sampling_file(p, time=T),
    "terrain": lambda p: write_terrain_sampling_file(p, time=T),
    "hybrid": lambda p: write_hybrid_sampling_file(p, time=T),
    "pressure-column": lambda p: write_pressure_column_file(p),
    "hilly-sigma": lambda p: write_hilly_file(p, flag=1),
    "hilly-pressure": lambda p: write_hilly_file(p, flag=2),
    "wrf": lambda p: write_varying_wrf_file(p),
    "wrf-coupled": lambda p: write_coupled_wrf_file(p),
    "global": lambda p: write_global_file(p, T),
    "lambert": lambda p: write_wind_file(p, make_lambert_grid(), time=T, u=5.0, v=2.0),
    "two-times": lambda p: write_two_time_file(p),
}


def _points(path, n=40, seed=0):
    """Points between grid nodes all over the file's grid, at each of its times."""
    rng = np.random.default_rng(seed)
    with File(path) as met:
        grid = met.grid
        times = list(met.times)
    coords = grid.calculate_coords()
    lat = np.asarray(coords["lat"][1], dtype=float)
    lon = np.asarray(coords["lon"][1], dtype=float)
    if lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    j = rng.integers(0, grid.ny - 1, n)
    i = rng.integers(0, grid.nx - 1, n)
    f = rng.uniform(0.1, 0.9, n)
    g = rng.uniform(0.1, 0.9, n)
    # Inside the cell (j, i): its corners' lon/lat, weighted.
    lon_p = (1 - f) * (1 - g) * lon[j, i] + f * (1 - g) * lon[j, i + 1]
    lon_p += (1 - f) * g * lon[j + 1, i] + f * g * lon[j + 1, i + 1]
    lat_p = (1 - f) * (1 - g) * lat[j, i] + f * (1 - g) * lat[j, i + 1]
    lat_p += (1 - f) * g * lat[j + 1, i] + f * g * lat[j + 1, i + 1]
    return pd.DataFrame(
        {
            "lon": lon_p,
            "lat": lat_p,
            "time": [times[k % len(times)] for k in range(n)],
        }
    )


def _variables(path):
    with File(path) as met:
        names = {r.variable for r in met.records}
    return sorted(n for n in names if not n.startswith("DIF")) + ["pressure"]


#: (z_kind, model_top, z values) to sample at; z for heights in m, pressures in hPa.
CASES = [
    ("agl", "default", [0.0, 5.0, 40.0, 300.0, 1500.0, 9000.0, 30000.0]),
    ("agl", None, [0.0, 40.0, 300.0, 1500.0]),
    ("agl", 15000.0, [10.0, 700.0, 5000.0]),
    ("msl", "default", [200.0, 1200.0, 4000.0]),
    ("pressure", "default", [1005.0, 950.0, 850.0, 500.0, 120.0]),
    ("native", "default", [0.0, 1.0, 2.0]),
]


def _sample(met, points, variables, z_kind, model_top, earth_relative=False):
    """Return the sampled frame, or the exception raised, for comparison."""
    kwargs = {} if model_top == "default" else {"model_top": model_top}
    try:
        out = sample_points(
            met,
            points,
            variables,
            z_kind=z_kind,
            earth_relative=earth_relative,
            **kwargs,
        )
    except (ValueError, KeyError) as error:
        return error
    return out


def _assert_same(from_file, from_dataset):
    if isinstance(from_file, Exception):
        assert type(from_dataset) is type(from_file), from_dataset
        assert str(from_dataset) == str(from_file)
        return
    assert not isinstance(from_dataset, Exception), from_dataset
    pd.testing.assert_frame_equal(from_dataset, from_file, check_exact=True)
    assert from_dataset.attrs == from_file.attrs


def _dataset_and_file(tmp_path, name):
    """Return open_dataset's Dataset of a test file, and the file write_dataset writes from it."""
    original = tmp_path / f"{name}.arl"
    WRITERS[name](original)
    ds = arlmet.open_dataset(original)
    path = tmp_path / f"{name}-written.arl"
    arlmet.write_dataset(ds, path)
    return ds, path


@pytest.mark.parametrize("name", list(WRITERS))
@pytest.mark.parametrize(("z_kind", "model_top", "z"), CASES)
def test_dataset_samples_as_its_file(tmp_path, name, z_kind, model_top, z):
    ds, path = _dataset_and_file(tmp_path, name)
    base = _points(path)
    points = pd.concat([base.assign(z=value) for value in z], ignore_index=True)
    variables = _variables(path)

    for variable in variables:
        from_file = _sample(path, points, [variable], z_kind, model_top)
        from_dataset = _sample(ds, points, [variable], z_kind, model_top)
        _assert_same(from_file, from_dataset)


@pytest.mark.parametrize("name", ["lambert", "wrf"])
def test_dataset_rotates_winds_as_its_file(tmp_path, name):
    ds, path = _dataset_and_file(tmp_path, name)
    points = _points(path).assign(z=100.0)
    winds = ["UWND", "VWND"]
    _assert_same(
        _sample(path, points, winds, "agl", "default", earth_relative=True),
        _sample(ds, points, winds, "agl", "default", earth_relative=True),
    )


def test_dataset_before_writing_agrees_to_packing_precision(tmp_path):
    # Values off the packing grid, as a converter's output would be: the
    # Dataset samples its own values, the file its packed ones.
    path = tmp_path / "hilly.arl"
    write_hilly_file(path, flag=1)
    unpacked = arlmet.open_dataset(path).load()
    rng = np.random.default_rng(1)
    for name in unpacked.data_vars:
        values = unpacked[name].values
        unpacked[name].values = values + rng.uniform(-1e-4, 1e-4, values.shape) * (
            np.abs(values) + 1.0
        )
    written = tmp_path / "written.arl"
    arlmet.write_dataset(unpacked, written)
    points = _points(written).assign(z=250.0)
    variables = ["TEMP", "UWND", "PRSS"]
    from_file = sample_points(written, points, variables, z_kind="agl")
    from_dataset = sample_points(unpacked, points, variables, z_kind="agl")
    with File(written) as met:
        precision = {
            name: max(r.header.precision for r in met.records if r.variable == name)
            for name in variables
        }
    for name in variables:
        a = from_file[name].to_numpy()
        b = from_dataset[name].to_numpy()
        assert not np.array_equal(a, b), name  # the packing shows
        # Each stored value is within half a packing step; the level heights
        # come from packed fields too, so a value can move by a little more.
        np.testing.assert_allclose(b, a, rtol=0, atol=2 * precision[name], err_msg=name)


def test_dataset_levels_are_numbered_in_coordinate_order(tmp_path):
    # A Dataset holding some of a file's levels samples as write_dataset's
    # compact file of those levels.
    path = tmp_path / "deep.arl"
    write_deep_sampling_file(path, time=T)
    ds = arlmet.open_dataset(path)
    subset = ds.isel(level=[0, 1, 2, 4, 6, 8, 10])
    written = tmp_path / "subset.arl"
    arlmet.write_dataset(subset, written)
    points = _points(written).assign(z=900.0)
    for z_kind in ("agl", "pressure"):
        z = 900.0 if z_kind == "agl" else 650.0
        _assert_same(
            _sample(written, points.assign(z=z), ["TEMP"], z_kind, "default"),
            _sample(subset, points.assign(z=z), ["TEMP"], z_kind, "default"),
        )


def test_a_level_nan_everywhere_is_not_stored(tmp_path):
    # TEMP removed from the top levels samples as the file without those records.
    path = tmp_path / "deep.arl"
    write_deep_sampling_file(path, time=T)
    ds = arlmet.open_dataset(path).load()
    top = ds["level"].values[-3:]
    ds["TEMP"].loc[{"level": top}] = np.nan
    written = tmp_path / "fewer.arl"
    arlmet.write_dataset(ds, written)
    with File(written) as met:
        assert (ds.sizes["level"], "TEMP") not in met[T]  # no record there
    points = _points(written).assign(z=12000.0)
    _assert_same(
        _sample(written, points, ["TEMP"], "agl", "default"),
        _sample(ds, points, ["TEMP"], "agl", "default"),
    )


def test_a_lazy_dataset_reads_only_the_time_step_sampled(tmp_path, monkeypatch):
    """Which levels hold data is settled from the first slice with a value."""
    from arlmet.record import DataRecord

    path = write_two_time_file(tmp_path / "two.arl")
    read_times = []
    read = DataRecord.read

    def recording(self, *args, **kwargs):
        read_times.append(self.time)
        return read(self, *args, **kwargs)

    monkeypatch.setattr(DataRecord, "read", recording)
    points = pd.DataFrame({"lon": [20.0], "lat": [-5.0], "z": [950.0], "time": [TIME0]})
    sample_points(arlmet.open_dataset(path), points, ["TEMP"], z_kind="pressure")

    assert read_times
    assert set(read_times) == {TIME0}


def _two_time_dataset(tmp_path):
    """The deep pressure file (SHGT, PRSS, TEMP on 10 levels) at two times, as one Dataset."""
    parts = []
    for time, offset in ((TIME0, 0.0), (TIME1, 3.0)):
        path = tmp_path / f"deep-{time:%H}.arl"
        write_deep_sampling_file(path, time=time)
        part = arlmet.open_dataset(path).load()
        part["TEMP"] = part["TEMP"] + np.float32(offset)  # differs between times
        parts.append(part)
    return xr.concat(
        parts, dim="time", data_vars="minimal", coords="minimal", compat="override"
    )


@pytest.mark.parametrize(
    ("emptied", "level", "sampled"),
    [
        ("TEMP", None, "TEMP"),
        ("TEMP", 0, "TEMP"),
        ("TEMP", 4, "TEMP"),
        ("TEMP", -1, "TEMP"),
        ("PRSS", None, "PRSS"),
        ("PRSS", None, "TEMP"),
        ("SHGT", None, "TEMP"),
    ],
)
@pytest.mark.parametrize("z_kind", ["agl", "pressure", "native"])
def test_a_field_nan_at_one_time_reads_as_nan(
    tmp_path, emptied, level, sampled, z_kind
):
    # Stored is decided across time steps, as write_dataset writes a record
    # (flagged missing) at a time its field is all NaN: the Dataset samples
    # as the file, and points at the other time keep their values.
    ds = _two_time_dataset(tmp_path)
    where = {"time": TIME1}
    if level is not None:
        where["level"] = ds["level"].values[level]
    ds[emptied].loc[where] = np.nan
    written = tmp_path / "written.arl"
    arlmet.write_dataset(ds, written)
    z = {"agl": 1500.0, "pressure": 700.0, "native": 4.5}[z_kind]
    points = _points(written).assign(z=z)
    from_file = _sample(written, points, [sampled], z_kind, "default")
    assert not isinstance(from_file, Exception), from_file
    assert from_file.loc[points["time"] == TIME0, sampled].notna().all()
    _assert_same(from_file, _sample(ds, points, [sampled], z_kind, "default"))


def test_datasets_and_files_mix(tmp_path):
    first, second = tmp_path / "a.arl", tmp_path / "b.arl"
    write_sampling_file(first, time=TIME0)
    write_sampling_file(second, time=TIME1, temp_offset=3.0)
    ds = arlmet.open_dataset(second)
    arlmet.write_dataset(ds, second_written := tmp_path / "b-written.arl")
    points = pd.concat(
        [_points(first).assign(time=TIME0), _points(first).assign(time=TIME1)],
        ignore_index=True,
    ).assign(z=100.0)
    from_files = sample_points([first, second_written], points, ["TEMP"], z_kind="agl")
    mixed = sample_points([first, ds], points, ["TEMP"], z_kind="agl")
    pd.testing.assert_frame_equal(mixed, from_files, check_exact=True)


def test_a_time_in_two_inputs_is_refused(tmp_path):
    path = tmp_path / "a.arl"
    write_sampling_file(path, time=T)
    with pytest.raises(ValueError, match="Multiple input files"):
        sample_points(
            [path, arlmet.open_dataset(path)],
            _points(path).assign(z=100.0),
            ["TEMP"],
            z_kind="agl",
        )


def test_a_point_time_not_in_the_dataset_is_refused(tmp_path):
    path = tmp_path / "a.arl"
    write_sampling_file(path, time=T)
    points = _points(path).assign(z=100.0, time=TIME1)
    with pytest.raises(ValueError, match="No input file contains"):
        sample_points(arlmet.open_dataset(path), points, ["TEMP"], z_kind="agl")


def _without_attrs(ds):
    # Dataset.drop_attrs() is newer than the xarray floor (2023.10).
    ds = ds.copy()
    ds.attrs = {}
    return ds


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_without_attrs, "attrs\\['source'\\]"),
        (lambda ds: ds.drop_vars("arl_grid"), "arl_grid"),
        (lambda ds: ds.isel(time=0), "scalar time"),
        (lambda ds: ds.isel(level=0, drop=True)[["SHGT", "PRSS"]], "surface-only"),
        (lambda ds: ds.isel(lat=slice(None, None, -1)), "must increase"),
        (lambda ds: ds.isel(lat=slice(2, 12)), "keeps the whole grid"),
        (lambda ds: ds.assign(BAD=ds["SHGT"].isel(time=0)), "lacks the dimension"),
    ],
)
def test_a_dataset_not_in_the_layout_is_refused(tmp_path, edit, message):
    path = tmp_path / "a.arl"
    write_sampling_file(path, time=T)
    ds = edit(arlmet.open_dataset(path))
    with pytest.raises(ValueError, match=message):
        sample_points(ds, _points(path).assign(z=100.0), ["TEMP"], z_kind="agl")


class TestOneUpperLevel:
    """A variable stored at one level above the ground still has a column (#254)."""

    def test_one_level_is_not_the_same_at_every_height(self, tmp_path):
        path = tmp_path / "column.arl"
        _write_column_file(path, axis=SigmaAxis(levels=[1.0, 0.99, 0.95, 0.9]))
        full = arlmet.open_dataset(path)
        one = arlmet.open_dataset(path, levels=[0, 1])

        def sample(ds, z, z_kind, **kwargs):
            points = pd.DataFrame({"lon": -100.0, "lat": 50.0, "z": z})
            return arlmet.sample_points(ds, points, ["TEMP"], z_kind=z_kind, **kwargs)

        # Below level 1 the column is the same, from the ground (T02M) up.
        heights = [50.0, 5000.0, 20000.0]
        got = sample(one, heights, "agl", model_top=None)["TEMP"].to_numpy()
        want = sample(full, heights, "agl", model_top=None)["TEMP"].to_numpy()
        np.testing.assert_allclose(got[0], want[0], rtol=1e-6)
        assert got[0] != 283.0  # not level 1's value
        assert np.isnan(got[1:]).all()  # above the Dataset's top

        # 900 hPa is below the 850 hPa ground, 500 hPa above the top.
        pressures = [900.0, 845.0, 500.0]
        got = sample(one, pressures, "pressure")["TEMP"].to_numpy()
        want = sample(full, pressures, "pressure")["TEMP"].to_numpy()
        np.testing.assert_allclose(got[1], want[1], rtol=1e-6)
        assert np.isnan(got[[0, 2]]).all()


#: Files with three or more levels, and the ways to keep some of them (#270).
DEEP = ["deep-pressure", "pressure-column", "hilly-sigma", "hilly-pressure", "wrf"]
SUBSETS = {
    "isel": lambda path: arlmet.open_dataset(path).isel(level=[1, 2]),
    "skip": lambda path: arlmet.open_dataset(path).isel(level=[0, 2]),
    "levels": lambda path: arlmet.open_dataset(path, levels=[0, 1]),
}


@pytest.mark.parametrize("name", DEEP)
@pytest.mark.parametrize("subset", SUBSETS)
@pytest.mark.parametrize(("z_kind", "model_top", "z"), CASES)
def test_dataset_of_some_levels_samples_as_its_file(
    tmp_path, name, subset, z_kind, model_top, z
):
    """A Dataset holding some of a file's levels is the file of those levels alone."""
    original = tmp_path / f"{name}.arl"
    WRITERS[name](original)
    ds = SUBSETS[subset](original)
    path = tmp_path / "some.arl"
    arlmet.write_dataset(ds, path)
    base = _points(path)
    points = pd.concat([base.assign(z=value) for value in z], ignore_index=True)

    for variable in _variables(path):
        from_file = _sample(path, points, [variable], z_kind, model_top)
        from_dataset = _sample(ds, points, [variable], z_kind, model_top)
        _assert_same(from_file, from_dataset)


@pytest.mark.parametrize(
    ("name", "drop"),
    [
        ("hilly-sigma", []),
        ("hilly-sigma", ["T02M"]),  # the column starts at the first level's sigma
        ("hilly-sigma", ["PRES", "T02M"]),  # pressure from sigma and PRSS
        ("wrf", []),
        ("wrf", ["T02M"]),
    ],
)
@pytest.mark.parametrize("subset", SUBSETS)
def test_dataset_of_some_levels_reads_each_level_where_it_is(
    tmp_path, name, drop, subset
):
    """
    At a level's own height or pressure the Dataset gives that level's value (#253, #254).

    Checked against the Dataset's own values and the full file's; the column
    ends at the Dataset's top and at the ground, not at the file's.
    """
    path = tmp_path / f"{name}.arl"
    WRITERS[name](path)
    some = SUBSETS[subset](path).drop_vars(drop)
    node = {"time": 0, "lat": 5, "lon": 7}
    where = some.isel(**node)
    heights = arlmet.z_agl(some).isel(**node).values.astype(float)
    pressures = arlmet.pressure(some).isel(**node).values.astype(float)

    def sample(z, z_kind):
        point = {"lon": float(where["lon"]), "lat": float(where["lat"])}
        points = pd.DataFrame(
            {**point, "z": z, "time": pd.Timestamp(where["time"].values)}
        )
        kwargs = {"model_top": None} if z_kind == "agl" else {}
        out = sample_points(some, points, ["TEMP"], z_kind=z_kind, **kwargs)
        return out["TEMP"].to_numpy()

    # A hair inside the top level: the heights and pressures come from float32 fields.
    want = where["TEMP"].values
    np.testing.assert_allclose(sample(heights * (1 - 1e-4), "agl"), want, rtol=1e-4)
    np.testing.assert_allclose(
        sample(pressures * (1 + 1e-5), "pressure"), want, rtol=1e-4
    )
    full = arlmet.open_dataset(path).isel(**node)["TEMP"]
    np.testing.assert_allclose(
        want, full.sel(level=some["level"].values).values, rtol=1e-5
    )

    # Above the Dataset's top, and below the ground, there is no value.
    assert np.isnan(sample([heights.max() + 500.0], "agl")).all()
    assert np.isnan(sample([0.5 * pressures.min()], "pressure")).all()
    assert np.isnan(sample([float(where["PRSS"]) + 20.0], "pressure")).all()
