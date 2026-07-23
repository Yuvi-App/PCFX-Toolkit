"""port of Mednafen's fixed-point IDCT (`pcfx/idct.cpp`).

The PC-FX decoder uses an integer Loeffler-Ligtenberg-Moschytz IDCT with
specific shift/rounding behaviour. Reproducing it approximately is not good
enough: a one-LSB difference shows up as a different pixel, and the whole point
of this port is that its output can be diffed against the emulator's.

All arithmetic is done in int64 and wrapped back to int32 wherever the C code
stores into an `int32`, so overflow behaves the same way.
"""

from __future__ import annotations

import numpy as np

IDCT_PRESHIFT = 9
EFF_RSHIFT_1D_COEFF = 2
EFF_RSHIFT_1D_POST = 6

_INT32_MIN = -(1 << 31)
_UINT32_MASK = (1 << 32) - 1


def _c_coeff(value: float) -> int:
    """`C_COEFF` from idct.cpp: round-half-up then truncate toward zero."""
    return int((1 << (32 - EFF_RSHIFT_1D_COEFF)) * value + 0.5)


COEFFS = (
    1779033704,
    _c_coeff(0.5411961001461970),
    _c_coeff(-1.8477590650225736),
    _c_coeff(0.7653668647301796),
    _c_coeff(-0.5555702330196022),
    _c_coeff(1.3870398453221474),
    _c_coeff(0.2758993792829430),
    _c_coeff(0.1950903220161282),
    _c_coeff(0.7856949583871022),
    _c_coeff(-1.1758756024193586),
)


def _w32(value: np.ndarray) -> np.ndarray:
    """Wrap an int64 array back into int32 range, as a C `int32` store would."""
    return ((value + (1 << 31)) & _UINT32_MASK) + _INT32_MIN


def _mulh(coeff: int, value: np.ndarray) -> np.ndarray:
    """`MUL_32x32_H32`: the high 32 bits of a signed 32x32 product."""
    return (value * coeff) >> 32


def _idct_1d(rows: np.ndarray, psh: int) -> np.ndarray:
    """One-dimensional IDCT over the last axis, returning it transposed.

    `IDCT_1D` reads eight contiguous inputs and writes eight outputs strided by
    8, so in matrix terms it transposes as it goes. Callers get shape
    `(..., 8, 8)` back with that transpose already applied.
    """
    c_in = [rows[..., i] for i in range(8)]
    c = [None] * 8

    if psh == 0:
        shift = IDCT_PRESHIFT - EFF_RSHIFT_1D_COEFF
        c[0] = _w32(c_in[0] << shift)
        c[4] = _w32(c_in[4] << shift)
        c[7] = _w32((c_in[7] + c_in[1]) << IDCT_PRESHIFT)
        c[1] = _w32((c_in[7] - c_in[1]) << IDCT_PRESHIFT)
        c[3] = _w32(_w32(46341 * c_in[5]) >> (15 - IDCT_PRESHIFT))
        c[5] = _w32(_w32(46341 * c_in[3]) >> (15 - IDCT_PRESHIFT))
        m = _w32(35468 * (c_in[2] + c_in[6]))
        down = 16 - IDCT_PRESHIFT + EFF_RSHIFT_1D_COEFF
        c[2] = _w32(_w32(-121095 * c_in[6] + m) >> down)
        c[6] = _w32(_w32(50159 * c_in[2] + m) >> down)
    else:
        c[0] = _w32((c_in[0] >> EFF_RSHIFT_1D_COEFF) + ((1 << psh) >> 1))
        c[4] = c_in[4] >> EFF_RSHIFT_1D_COEFF
        c[7] = _w32(c_in[7] + c_in[1])
        c[1] = _w32(c_in[7] - c_in[1])
        c[3] = _w32(_w32(c_in[5] * 181) >> 7)
        c[5] = _w32(_w32(c_in[3] * 181) >> 7)
        m = _mulh(COEFFS[1], _w32(c_in[2] + c_in[6]))
        c[2] = _w32(_mulh(COEFFS[2], c_in[6]) + m)
        c[6] = _w32(_mulh(COEFFS[3], c_in[2]) + m)

    def snorp(a0: int, a1: int) -> None:
        total = _w32(c[a0] + c[a1])
        c[a1] = _w32(c[a0] - c[a1])
        c[a0] = total

    snorp(0, 4)
    snorp(7, 5)
    snorp(3, 1)

    m = _mulh(COEFFS[4], _w32(c[7] + c[1]))
    r = _w32(_mulh(COEFFS[5], c[1]) + m)
    c[1] = _w32(_mulh(COEFFS[6], c[7]) - m)
    c[7] = r

    m = _mulh(COEFFS[7], _w32(c[3] + c[5]))
    r = _w32(_mulh(COEFFS[8], c[5]) + m)
    c[5] = _w32(_mulh(COEFFS[9], c[3]) + m)
    c[3] = r

    snorp(0, 6)
    snorp(4, 2)

    out = np.empty(rows.shape[:-1] + (8,), dtype=np.int64)
    out[..., 0] = _w32(c[0] + c[1]) >> psh
    out[..., 1] = _w32(c[4] + c[5]) >> psh
    out[..., 2] = _w32(c[2] + c[3]) >> psh
    out[..., 3] = _w32(c[6] + c[7]) >> psh
    out[..., 4] = _w32(c[6] - c[7]) >> psh
    out[..., 5] = _w32(c[2] - c[3]) >> psh
    out[..., 6] = _w32(c[4] - c[5]) >> psh
    out[..., 7] = _w32(c[0] - c[1]) >> psh
    return np.swapaxes(out, -1, -2)


def idct(blocks: np.ndarray) -> np.ndarray:
    """Run the 2-D IDCT over a batch of 8x8 coefficient blocks.

    `blocks` is `(n, 64)` or `(n, 8, 8)` of int64 coefficients; the result has
    the same shape as the input, in raster order.
    """
    flat = blocks.ndim == 2 and blocks.shape[-1] == 64
    data = blocks.reshape(-1, 8, 8).astype(np.int64, copy=False)
    buf = _idct_1d(data, 0)
    out = _idct_1d(buf, EFF_RSHIFT_1D_POST)
    return out.reshape(-1, 64) if flat else out
