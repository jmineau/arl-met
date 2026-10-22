"""Tests for ARL packing and header serialization."""

import numpy as np
import pytest

from arlmet._pack import pack_core as _pack_core
from arlmet.grid import GridWindow
from arlmet.header import Header, format_fortran_float
from arlmet.packing import calculate_checksum, pack, unpack


def _pack_core_python(
    unpacked: np.ndarray,
    scale: float,
    inv_scale: float,
    initial_value: float,
) -> np.ndarray:
    """Pure-Python reference implementation of the feedback-loop encoder."""
    ny, nx = unpacked.shape
    packed = np.empty((ny, nx), dtype=np.uint8)
    packed[0, 0] = 127
    previous_row0 = initial_value
    for y in range(ny):
        if y > 0:
            icval = int((unpacked[y, 0] - previous_row0) * scale + 127.5)
            packed[y, 0] = icval
            previous_row0 += (icval - 127) * inv_scale
        previous_value = previous_row0
        for x in range(1, nx):
            icval = int((unpacked[y, x] - previous_value) * scale + 127.5)
            packed[y, x] = icval
            previous_value += (icval - 127) * inv_scale
    return packed


def _pakinp_python(
    packed: bytes,
    nx: int,
    ny: int,
    precision: float,
    exponent: int,
    initial_value: float,
    window: GridWindow | None = None,
) -> np.ndarray:
    """
    Line-by-line transcription of HYSPLIT's ``PAKINP`` in single precision.

    Checked against the compiled Fortran on the cases below: it gives the
    same bits. Indices are zero-based; ``window`` is the sub-grid.
    """
    if window is None:
        window = GridWindow(x_start=0, x_stop=nx, y_start=0, y_stop=ny)
    f32 = np.float32
    cvar = np.frombuffer(packed, dtype=np.uint8)
    prec = f32(precision)
    scexp = f32(1.0) / f32(2.0) ** (7 - exponent)
    lx = window.x_stop - window.x_start
    ly = window.y_stop - window.y_start
    rvar = np.zeros((ly, lx), dtype=np.float32)

    # Column 1 of every row, from the row before, with no precision test.
    rold = f32(initial_value)
    for j in range(ny):
        rnew = f32(int(cvar[j * nx]) - 127) * scexp + rold
        rold = rnew
        if window.y_start <= j < window.y_stop:
            rvar[j - window.y_start, 0] = rnew

    # The rest of each row of the sub-grid, from its first column.
    for j in range(window.y_start, window.y_stop):
        jj = j - window.y_start
        rold = rvar[jj, 0]
        for i in range(1, window.x_stop):
            rnew = f32(int(cvar[j * nx + i]) - 127) * scexp + rold
            rold = rnew
            if abs(rnew) < prec:
                rnew = f32(0.0)
            if window.x_start <= i < window.x_stop:
                rvar[jj, i - window.x_start] = rnew
    return rvar


def _edge_records() -> dict[str, tuple[bytes, int, int, float, int, float]]:
    """Packed records that hit the edges of ``PAKINP``: (bytes, nx, ny, prec, exp, var1)."""
    rng = np.random.default_rng(60)
    cases = {}
    # No change anywhere and 0.25 at the corner, below the precision (#60).
    cases["corner_below_precision"] = (bytes([127]) * 12, 4, 3, 1.0, 7, 0.25)
    # The first column walks across zero in steps smaller than the precision.
    walk = np.full((9, 5), 127, dtype=np.uint8)
    walk[1:, 0] = 126
    cases["first_column_crosses_zero"] = (
        walk.tobytes(),
        5,
        9,
        0.001968504,
        -1,
        0.005905512,
    )
    # Steps far below the float32 spacing at the initial value, so the order of
    # the single-precision additions shows.
    steps = rng.integers(0, 256, (40, 6), dtype=np.uint8)
    cases["steps_below_spacing"] = (steps.tobytes(), 6, 40, 0.00006151575, -6, 101325.4)
    # Random differences around zero at a few precisions, each written with the
    # seven digits a header holds.
    for exponent, var1 in [(0, 0.001), (3, -2.7), (-4, 0.0), (-12, 1e-5)]:
        data = rng.integers(100, 155, (13, 17), dtype=np.uint8)
        precision = float(f"{2.0**exponent / 254:.6e}")
        cases[f"random_exp{exponent}"] = (
            data.tobytes(),
            17,
            13,
            precision,
            exponent,
            var1,
        )
    return cases


_EDGE_RECORDS = _edge_records()


class TestChecksum:
    def test_checksum_wraps_at_255(self):
        packed = bytes([255, 1, 255])
        assert calculate_checksum(packed) == 1


class TestHeaderSerialization:
    @pytest.mark.parametrize(
        ("value", "text"),
        [
            # What gfortran's WRITE(...,'(E14.7)') gives for these float32
            # values; the first three are exact ties at the seventh digit, which
            # Fortran rounds to even.
            (9763.0625, " 0.9763062E+04"),
            (5301842.5, " 0.5301842E+07"),
            (-83888.125, "-0.8388812E+05"),
            # float32(1e-4) is just under 1e-4; its mantissa rounds up to 1.
            (9.999999747378752e-05, " 0.1000000E-03"),
            (0.003937007859349251, " 0.3937008E-02"),
            (101325.3671875, " 0.1013254E+06"),
            (-2.700000047683716, "-0.2700000E+01"),
            (1.0000000031710769e-30, " 0.1000000E-29"),
            (3.3999999521443642e38, " 0.3400000E+39"),
            (0.5, " 0.5000000E+00"),
            (1.0, " 0.1000000E+01"),
            (0.0, " 0.0000000E+00"),
        ],
    )
    def test_floats_are_written_as_fortran_e14_7(self, value, text):
        assert format_fortran_float(value) == text

    def test_header_roundtrip(self):
        header = Header(
            year=2024,
            month=7,
            day=18,
            hour=0,
            forecast=6,
            level=3,
            grid=(0, 0),
            variable="TEMP",
            exponent=7,
            precision=0.5039370,
            initial_value=1011.406,
        )

        raw = header.tobytes()

        assert len(raw) == Header.N_BYTES
        assert raw == b"24 718 0 6 399TEMP   7 0.5039370E+00 0.1011406E+04"
        assert Header.from_bytes(raw) == header

    def test_header_roundtrip_with_large_grid_letters(self):
        header = Header(
            year=2025,
            month=9,
            day=1,
            hour=12,
            forecast=0,
            level=0,
            grid=(1000, 2000),
            variable="PRSS",
            exponent=8,
            precision=1.007874,
            initial_value=680.5,
        )

        raw = header.tobytes()

        assert raw[12:14] == b"AB"
        assert Header.from_bytes(raw) == header


class TestPack:
    def test_constant_field_roundtrip(self):
        unpacked = np.full((3, 4), 42.0, dtype=np.float32)

        packed, precision, exponent, initial_value = pack(unpacked)
        roundtripped = unpack(
            packed.tobytes(),
            nx=4,
            ny=3,
            precision=precision,
            exponent=exponent,
            initial_value=initial_value,
        )

        np.testing.assert_array_equal(packed, 127)
        np.testing.assert_allclose(roundtripped, unpacked, atol=precision)

    def test_gradient_roundtrip(self):
        unpacked = np.array(
            [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]],
            dtype=np.float32,
        )

        packed, precision, exponent, initial_value = pack(unpacked)
        roundtripped = unpack(
            packed.tobytes(),
            nx=3,
            ny=3,
            precision=precision,
            exponent=exponent,
            initial_value=initial_value,
        )

        np.testing.assert_allclose(roundtripped, unpacked, atol=precision)

    def test_signed_field_roundtrip(self):
        unpacked = np.array(
            [[-3.0, -1.0, 2.0], [4.0, 0.0, -4.0]],
            dtype=np.float32,
        )

        packed, precision, exponent, initial_value = pack(unpacked)
        roundtripped = unpack(
            packed.tobytes(),
            nx=3,
            ny=2,
            precision=precision,
            exponent=exponent,
            initial_value=initial_value,
        )

        np.testing.assert_allclose(roundtripped, unpacked, atol=precision)

    def test_row_reset_uses_previous_row_first_column(self):
        unpacked = np.array(
            [[10.0, 11.0, 12.0], [12.0, 13.0, 14.0]],
            dtype=np.float32,
        )

        packed, _, _, _ = pack(unpacked)

        assert packed[0, 0] == 127
        assert packed[1, 0] > 127

    def test_windowed_unpack_matches_full_subset(self):
        unpacked = np.arange(1, 31, dtype=np.float32).reshape(5, 6)

        packed, precision, exponent, initial_value = pack(unpacked)
        window = GridWindow(x_start=2, x_stop=5, y_start=1, y_stop=4)

        subset = unpack(
            packed.tobytes(),
            nx=6,
            ny=5,
            precision=precision,
            exponent=exponent,
            initial_value=initial_value,
            window=window,
        )

        np.testing.assert_allclose(subset, unpacked[1:4, 2:5], atol=precision)

    def test_roundtrip_uses_running_reconstructed_value(self):
        unpacked = np.array(
            [
                [200.0, 208.25, 208.25, 208.25, 208.25, 208.25, 208.25, 208.25],
                [
                    231.7,
                    232.0125,
                    232.325,
                    232.6375,
                    232.95,
                    233.2625,
                    233.575,
                    233.8875,
                ],
            ],
            dtype=np.float32,
        )

        packed, precision, exponent, initial_value = pack(unpacked)
        roundtripped = unpack(
            packed.tobytes(),
            nx=unpacked.shape[1],
            ny=unpacked.shape[0],
            precision=precision,
            exponent=exponent,
            initial_value=initial_value,
        )

        np.testing.assert_allclose(roundtripped, unpacked, atol=precision)


class TestUnpackMatchesPakinp:
    """``unpack`` gives HYSPLIT's ``PAKINP`` values, bit for bit."""

    def test_first_column_keeps_values_below_precision(self):
        # The values the compiled PAKINP gives for these bytes (#60).
        unpacked = unpack(
            bytes([127]) * 12, 4, 3, precision=1.0, exponent=7, initial_value=0.25
        )
        expected = np.zeros((3, 4), dtype=np.float32)
        expected[:, 0] = 0.25
        np.testing.assert_array_equal(unpacked, expected)

    @pytest.mark.parametrize("name", sorted(_EDGE_RECORDS))
    def test_full_grid(self, name):
        packed, nx, ny, precision, exponent, var1 = _EDGE_RECORDS[name]
        got = unpack(packed, nx, ny, precision, exponent, var1)
        want = _pakinp_python(packed, nx, ny, precision, exponent, var1)
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))

    @pytest.mark.parametrize("name", sorted(_EDGE_RECORDS))
    @pytest.mark.parametrize(
        "corner", [(0, 0, 3, 3), (1, 1, 3, 2), (0, 1, 4, 2), (2, 0, 2, 3)]
    )
    def test_window(self, name, corner):
        packed, nx, ny, precision, exponent, var1 = _EDGE_RECORDS[name]
        x0, y0, lx, ly = corner
        window = GridWindow(x_start=x0, x_stop=x0 + lx, y_start=y0, y_stop=y0 + ly)
        got = unpack(packed, nx, ny, precision, exponent, var1, window=window)
        want = _pakinp_python(packed, nx, ny, precision, exponent, var1, window)
        np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))

    def test_window_is_the_full_grid_cut(self):
        packed, nx, ny, precision, exponent, var1 = _EDGE_RECORDS["random_exp0"]
        full = unpack(packed, nx, ny, precision, exponent, var1)
        window = GridWindow(x_start=3, x_stop=11, y_start=2, y_stop=9)
        cut = unpack(packed, nx, ny, precision, exponent, var1, window=window)
        np.testing.assert_array_equal(cut, full[2:9, 3:11])

    def test_roundtrip_keeps_small_first_column(self):
        unpacked = np.zeros((4, 5), dtype=np.float32)
        unpacked[:, 1:] = 10.0
        unpacked[2, 0] = 0.02

        packed, precision, exponent, initial_value = pack(unpacked)
        roundtripped = unpack(
            packed.tobytes(), 5, 4, precision, exponent, initial_value
        )

        assert abs(unpacked[2, 0]) < precision
        np.testing.assert_allclose(roundtripped, unpacked, atol=precision)


class TestPackCore:
    """Validate the C extension byte-for-byte against the Python reference."""

    def _args(self, unpacked: np.ndarray):
        import math

        diffs = np.empty_like(unpacked)
        diffs[:, 1:] = unpacked[:, 1:] - unpacked[:, :-1]
        diffs[1:, 0] = unpacked[1:, 0] - unpacked[:-1, 0]
        diffs[0, 0] = 0.0
        rmax = float(np.max(np.abs(diffs)))
        if rmax > 0.0:
            sexp = math.log(rmax) / math.log(2.0)
            exponent = int(sexp)
            if sexp >= 0.0 or sexp == int(sexp):
                exponent += 1
        else:
            exponent = 0
        scale = 2.0 ** (7 - exponent)
        inv_scale = 1.0 / scale
        return scale, inv_scale, float(unpacked[0, 0])

    def test_c_ext_matches_python_reference_gradient(self):
        unpacked = np.array(
            [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]],
            dtype=np.float32,
        )
        scale, inv_scale, iv = self._args(unpacked)
        np.testing.assert_array_equal(
            _pack_core(unpacked, scale, inv_scale, iv),
            _pack_core_python(unpacked, scale, inv_scale, iv),
        )

    def test_c_ext_matches_python_reference_signed(self):
        unpacked = np.array([[-3.0, -1.0, 2.0], [4.0, 0.0, -4.0]], dtype=np.float32)
        scale, inv_scale, iv = self._args(unpacked)
        np.testing.assert_array_equal(
            _pack_core(unpacked, scale, inv_scale, iv),
            _pack_core_python(unpacked, scale, inv_scale, iv),
        )

    def test_c_ext_matches_python_reference_running_reconstructed(self):
        """Field where simple diff diverges — validates feedback-loop correctness."""
        unpacked = np.array(
            [
                [200.0, 208.25, 208.25, 208.25, 208.25, 208.25, 208.25, 208.25],
                [
                    231.7,
                    232.0125,
                    232.325,
                    232.6375,
                    232.95,
                    233.2625,
                    233.575,
                    233.8875,
                ],
            ],
            dtype=np.float32,
        )
        scale, inv_scale, iv = self._args(unpacked)
        np.testing.assert_array_equal(
            _pack_core(unpacked, scale, inv_scale, iv),
            _pack_core_python(unpacked, scale, inv_scale, iv),
        )
