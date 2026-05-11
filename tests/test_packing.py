"""Tests for ARL packing and header serialization."""

import numpy as np

from arlmet.grid import GridWindow
from arlmet.header import Header
from arlmet.packing import calculate_checksum, pack, unpack


class TestChecksum:
    def test_checksum_wraps_at_255(self):
        packed = bytes([255, 1, 255])
        assert calculate_checksum(packed) == 1


class TestHeaderSerialization:
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
            driver=np,
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
            driver=np,
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
            driver=np,
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
            driver=np,
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
            driver=np,
        )

        np.testing.assert_allclose(roundtripped, unpacked, atol=precision)
