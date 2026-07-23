"""Bit-exact port of Mednafen's RAINBOW decoder (`pcfx/rainbow.cpp`).

Decodes one RAINBOW frame packet into a 256-wide YUV picture. Chroma
interpolation is off, matching `RAINBOW_Accurate_Init(false)` in the reference
`rainbow_avi.c` harness, which means blocks carry no state across frames other
than the quant tables -- so frames parallelise cleanly.

Two things here look odd but are deliberate:

* `_to_int16` truncation on `quant * coefficient`. The C code stores through an
  `(int16)` cast and the product routinely exceeds 16 bits, so the wraparound is
  visible in the output.
* Integer division in the YUV->BGR matrix truncates toward zero, as C does, not
  toward negative infinity as Python's `//` would.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import PcfxError
from .blocks import (
    DEFAULT_MAX_INTERBLOCK_PAD,
    QUANT_TABLE_BYTES,
    STRIP_HEIGHT,
    FRAME_WIDTH,
    parse_block,
    skip_interblock_pad,
)
from .idct import idct
from .tables import AC_UV_LUT, AC_Y_LUT, DC_UV_LUT, DC_Y_LUT, ZIGZAG

MACROBLOCK_COLUMNS = 16
BLOCKS_PER_MACROBLOCK = 6

#: Default RAINBOW null-run colour, matching the `--null-yuv` default of the
#: reference harness. Y=0 U=0x80 V=0x80 renders as black.
DEFAULT_NULL_YUV = 0x008080


def _to_int16(value: int) -> int:
    return ((value + 0x8000) & 0xFFFF) - 0x8000


def unescape(payload: bytes) -> bytes:
    """Undo RAINBOW's 0xFF byte stuffing.

    The stream escapes every 0xFF by following it with 0x00, and the decoder
    drops the byte after any 0xFF it reads. Inside a block payload that makes
    `replace` exactly equivalent to the sequential rule and far faster; if a
    stream ever breaks the convention we fall back to the literal walk.
    """
    if b"\xff" not in payload:
        return payload
    if payload.count(b"\xff") == payload.count(b"\xff\x00"):
        return payload.replace(b"\xff\x00", b"\xff")

    out = bytearray()
    index = 0
    length = len(payload)
    while index < length:
        byte = payload[index]
        out.append(byte)
        index += 2 if byte == 0xFF else 1
    return bytes(out)


@dataclass(slots=True)
class DecodedFrame:
    """One frame as packed 0x00YYUUVV values, shape `(height, 256)`."""

    yuv: np.ndarray
    rle_blocks: int = 0
    truncated_blocks: int = 0

    @property
    def height(self) -> int:
        return int(self.yuv.shape[0])

    @property
    def width(self) -> int:
        return int(self.yuv.shape[1])


def _decode_strip(
    buf: bytes,
    quant: list[list[int]],
    quant_base: list[list[int]],
    blocks_out: list[list[int]],
    decoded_columns: list[int],
    null_columns: list[int],
) -> None:
    """Entropy-decode one 256x16 DCT strip.

    Everything lives in locals: this runs about 1,400 times per frame and tens
    of millions of times per disc, so attribute and global lookups matter.
    """
    zigzag = ZIGZAG
    ac_y_lut = AC_Y_LUT
    ac_uv_lut = AC_UV_LUT
    dc_y_lut = DC_Y_LUT
    dc_uv_lut = DC_UV_LUT

    data = buf
    size = len(data)
    pos = 0
    bit_buffer = 0
    bit_count = 0

    quant_y = quant[0]
    quant_uv = quant[1]
    dc_y = 0
    dc_u = 0
    dc_v = 0
    column = 0

    while column < MACROBLOCK_COLUMNS:
        # ---- get_dc_y_coeff -------------------------------------------------
        zeroes = 0
        while True:
            while bit_count < 9:
                bit_buffer = ((bit_buffer << 8) | (data[pos] if pos < size else 0)) & 0xFFFFFFFF
                pos += 1
                bit_count += 8
            code, used = dc_y_lut[(bit_buffer >> (bit_count - 9)) & 0x1FF]
            bit_count -= used

            if code < 0xF:
                if code:
                    while bit_count < code:
                        bit_buffer = (
                            (bit_buffer << 8) | (data[pos] if pos < size else 0)
                        ) & 0xFFFFFFFF
                        pos += 1
                        bit_count += 8
                    bit_count -= code
                    value = (bit_buffer >> bit_count) & ((1 << code) - 1)
                    if value < (1 << (code - 1)):
                        value += 1 - (1 << code)
                else:
                    value = 0
                dc_y += value
                break

            if code == 0xF:
                while bit_count < 12:
                    bit_buffer = (
                        (bit_buffer << 8) | (data[pos] if pos < size else 0)
                    ) & 0xFFFFFFFF
                    pos += 1
                    bit_count += 8
                ac_code, ac_used = ac_y_lut[(bit_buffer >> (bit_count - 12)) & 0xFFF]
                bit_count -= ac_used
                numbits = ac_code & 0xF
                zeroes = (ac_code >> 4) + 1
                if numbits:
                    while bit_count < numbits:
                        bit_buffer = (
                            (bit_buffer << 8) | (data[pos] if pos < size else 0)
                        ) & 0xFFFFFFFF
                        pos += 1
                        bit_count += 8
                    bit_count -= numbits
                break

            # code >= 0x10: rewrite the active quant tables from the base pair.
            scale = code - 0x10
            base_y = quant_base[0]
            base_uv = quant_base[1]
            for index in range(64):
                value = (base_y[index] * scale) >> 2
                quant_y[index] = 1 if value < 1 else (0xFE if value > 0xFE else value)
                if index:
                    value = (base_uv[index] * scale) >> 2
                else:
                    value = base_uv[index] >> 2
                quant_uv[index] = 1 if value < 1 else (0xFE if value > 0xFE else value)

        if zeroes:
            # A null run blanks whole macroblock columns and resets prediction.
            while zeroes:
                if column < MACROBLOCK_COLUMNS:
                    null_columns.append(column)
                column += 1
                zeroes -= 1
            dc_y = dc_u = dc_v = 0
            continue

        decoded_columns.append(column)
        column += 1

        # ---- six 8x8 blocks: Y a/b/c/d then U then V ------------------------
        for slot in range(BLOCKS_PER_MACROBLOCK):
            if slot < 4:
                table = ac_y_lut
                quant_table = quant_y
                if slot == 0:
                    predictor = dc_y
                else:
                    # Each further luma block carries its own DC delta.
                    while bit_count < 9:
                        bit_buffer = (
                            (bit_buffer << 8) | (data[pos] if pos < size else 0)
                        ) & 0xFFFFFFFF
                        pos += 1
                        bit_count += 8
                    code, used = dc_y_lut[(bit_buffer >> (bit_count - 9)) & 0x1FF]
                    bit_count -= used
                    if code < 0xF:
                        if code:
                            while bit_count < code:
                                bit_buffer = (
                                    (bit_buffer << 8) | (data[pos] if pos < size else 0)
                                ) & 0xFFFFFFFF
                                pos += 1
                                bit_count += 8
                            bit_count -= code
                            value = (bit_buffer >> bit_count) & ((1 << code) - 1)
                            if value < (1 << (code - 1)):
                                value += 1 - (1 << code)
                        else:
                            value = 0
                        dc_y += value
                    predictor = dc_y
            else:
                table = ac_uv_lut
                quant_table = quant_uv
                while bit_count < 8:
                    bit_buffer = (
                        (bit_buffer << 8) | (data[pos] if pos < size else 0)
                    ) & 0xFFFFFFFF
                    pos += 1
                    bit_count += 8
                code, used = dc_uv_lut[(bit_buffer >> (bit_count - 8)) & 0xFF]
                bit_count -= used
                if code:
                    while bit_count < code:
                        bit_buffer = (
                            (bit_buffer << 8) | (data[pos] if pos < size else 0)
                        ) & 0xFFFFFFFF
                        pos += 1
                        bit_count += 8
                    bit_count -= code
                    value = (bit_buffer >> bit_count) & ((1 << code) - 1)
                    if value < (1 << (code - 1)):
                        value += 1 - (1 << code)
                else:
                    value = 0
                if slot == 4:
                    dc_u += value
                    predictor = dc_u
                else:
                    dc_v += value
                    predictor = dc_v

            block = [0] * 64
            product = quant_table[0] * predictor
            block[0] = ((product + 0x8000) & 0xFFFF) - 0x8000
            index = 0
            while index < 63:
                while bit_count < 12:
                    bit_buffer = (
                        (bit_buffer << 8) | (data[pos] if pos < size else 0)
                    ) & 0xFFFFFFFF
                    pos += 1
                    bit_count += 8
                ac_code, ac_used = table[(bit_buffer >> (bit_count - 12)) & 0xFFF]
                bit_count -= ac_used
                numbits = ac_code & 0xF
                run = ac_code >> 4
                if numbits:
                    while bit_count < numbits:
                        bit_buffer = (
                            (bit_buffer << 8) | (data[pos] if pos < size else 0)
                        ) & 0xFFFFFFFF
                        pos += 1
                        bit_count += 8
                    bit_count -= numbits
                    coeff = (bit_buffer >> bit_count) & ((1 << numbits) - 1)
                    if coeff < (1 << (numbits - 1)):
                        coeff += 1 - (1 << numbits)
                else:
                    coeff = 0

                if not coeff:
                    if not run:
                        break  # end of block; the rest is already zero
                    if run == 1:
                        run = 0xF

                while run and index < 63:
                    index += 1
                    run -= 1
                if index < 63:
                    target = zigzag[index]
                    index += 1
                    product = quant_table[target] * coeff
                    block[target] = ((product + 0x8000) & 0xFFFF) - 0x8000

            blocks_out.append(block)


def _decode_rle(payload: bytes, block_type: int) -> np.ndarray:
    """Palettized RLE block -> 8-bit indices (`rainbow_accurate.c:667`)."""
    plt_shift = 4 - (block_type & 0x3)
    crl_mask = (1 << plt_shift) - 1
    out = np.zeros(0x2000, dtype=np.uint8)

    remaining = len(payload)
    pos = 0
    x = 0
    while remaining > 0 and pos < len(payload):
        boot = payload[pos]
        pos += 1
        remaining -= 1
        if boot == 0xFF:
            pos += 1
            remaining -= 1

        if not (boot & crl_mask):
            if pos >= len(payload):
                break
            count = payload[pos]
            pos += 1
            remaining -= 1
            if count == 0xFF:
                pos += 1
                remaining -= 1
            count += 1
        else:
            count = boot & crl_mask

        value = boot >> plt_shift
        end = min(x + count, 0x2000)
        if end > x:
            out[x:end] = value
        x = end
        if x >= 0x2000:
            break
    return out


class RainbowDecoder:
    """Decodes RAINBOW frame packets.

    Quant tables persist between frames, matching the hardware: a frame whose
    first block is 0xF8 reuses whatever the previous 0xFF block loaded.
    """

    __slots__ = ("quant", "quant_base", "happy_color", "null_yuv", "rle_as_grey")

    def __init__(self, null_yuv: int = DEFAULT_NULL_YUV, rle_as_grey: bool = False) -> None:
        self.quant = [[0] * 64, [0] * 64]
        self.quant_base = [[0] * 64, [0] * 64]
        self.null_yuv = null_yuv
        self.happy_color = null_yuv & 0xFFFFFF
        self.rle_as_grey = rle_as_grey

    def reset(self) -> None:
        self.quant = [[0] * 64, [0] * 64]
        self.quant_base = [[0] * 64, [0] * 64]

    def decode(self, packet: bytes, strips: int | None = None) -> DecodedFrame:
        """Decode one frame packet into a `(strips * 16, 256)` YUV picture."""
        parsed: list[tuple[int, int, int]] = []  # (block_type, payload_start, payload_end)
        cursor = 0
        while True:
            probe = skip_interblock_pad(packet, cursor, max_pad=DEFAULT_MAX_INTERBLOCK_PAD)
            block = parse_block(packet, probe)
            if block is None:
                break
            if parsed and block.carries_quant_tables:
                break
            parsed.append((block.block_type, block.start + 4, block.end))
            cursor = block.end
            if strips is not None and len(parsed) >= strips:
                break

        if not parsed:
            raise PcfxError("Packet does not begin with a RAINBOW block")
        if strips is None:
            strips = len(parsed)

        height = strips * STRIP_HEIGHT
        picture = np.zeros((height, FRAME_WIDTH), dtype=np.uint32)
        # A (strip, y, column, x) view lets whole macroblocks be scattered in
        # one vectorised assignment instead of 240 slice writes per frame.
        tiles = picture.reshape(strips, STRIP_HEIGHT, MACROBLOCK_COLUMNS, 16)
        rle_blocks = 0
        truncated = 0

        macroblock_rows: list[int] = []
        macroblock_columns: list[int] = []
        dct_blocks: list[list[int]] = []
        null_rows: list[int] = []
        null_columns_all: list[int] = []

        for row, (block_type, start, end) in enumerate(parsed):
            payload_start = start
            if block_type == 0xFF:
                table_bytes = packet[start : start + QUANT_TABLE_BYTES]
                if len(table_bytes) == QUANT_TABLE_BYTES:
                    for index in range(64):
                        value = table_bytes[index]
                        self.quant[0][index] = value
                        self.quant_base[0][index] = value
                        value = table_bytes[64 + index]
                        self.quant[1][index] = value
                        self.quant_base[1][index] = value
                payload_start = start + QUANT_TABLE_BYTES

            if block_type in (0xF8, 0xFF):
                if payload_start >= end:
                    truncated += 1
                    continue
                before = len(dct_blocks)
                columns: list[int] = []
                nulls: list[int] = []
                _decode_strip(
                    unescape(packet[payload_start:end]),
                    self.quant,
                    self.quant_base,
                    dct_blocks,
                    columns,
                    nulls,
                )
                if (len(dct_blocks) - before) != len(columns) * BLOCKS_PER_MACROBLOCK:
                    raise PcfxError("Strip produced an inconsistent block count")
                macroblock_rows.extend([row] * len(columns))
                macroblock_columns.extend(columns)
                null_rows.extend([row] * len(nulls))
                null_columns_all.extend(nulls)
            else:
                rle_blocks += 1
                indices = _decode_rle(packet[payload_start:end], block_type)
                if self.rle_as_grey:
                    grey = indices[: STRIP_HEIGHT * FRAME_WIDTH].reshape(
                        STRIP_HEIGHT, FRAME_WIDTH
                    )
                    top = row * STRIP_HEIGHT
                    picture[top : top + STRIP_HEIGHT] = (grey.astype(np.uint32) << 16) | 0x8080

        if dct_blocks:
            samples = idct(np.array(dct_blocks, dtype=np.int64))
            tiles[macroblock_rows, :, macroblock_columns, :] = _pack_macroblocks(samples)

        if null_rows:
            tiles[null_rows, :, null_columns_all, :] = self.happy_color

        return DecodedFrame(yuv=picture, rle_blocks=rle_blocks, truncated_blocks=truncated)


def _pack_macroblocks(samples: np.ndarray) -> np.ndarray:
    """Six 8x8 blocks -> one 16x16 macroblock of packed 0x00YYUUVV values.

    Luma quadrant order is A top-left, B bottom-left, C top-right, D
    bottom-right (`rainbow_accurate.c:556`), and chroma is 4:2:0, replicated
    2x2 because chroma interpolation is disabled.
    """
    macroblocks = samples.reshape(-1, BLOCKS_PER_MACROBLOCK, 8, 8)
    count = macroblocks.shape[0]

    luma = np.empty((count, 16, 16), dtype=np.int64)
    luma[:, 0:8, 0:8] = macroblocks[:, 0]
    luma[:, 8:16, 0:8] = macroblocks[:, 1]
    luma[:, 0:8, 8:16] = macroblocks[:, 2]
    luma[:, 8:16, 8:16] = macroblocks[:, 3]

    chroma_u = np.repeat(np.repeat(macroblocks[:, 4], 2, axis=1), 2, axis=2)
    chroma_v = np.repeat(np.repeat(macroblocks[:, 5], 2, axis=1), 2, axis=2)

    y = np.clip(luma + 0x80, 0, 255).astype(np.uint32)
    u = np.clip(chroma_u + 0x80, 0, 255).astype(np.uint32)
    v = np.clip(chroma_v + 0x80, 0, 255).astype(np.uint32)
    return (y << 16) | (u << 8) | v


def yuv_to_bgra(picture: np.ndarray) -> np.ndarray:
    """Convert packed YUV to BGRA bytes, exactly as `rainbow_avi.c` does."""
    yuv = picture.astype(np.int64) & 0xFFFFFF
    y = (yuv >> 16) & 0xFF
    u = ((yuv >> 8) & 0xFF) - 128
    v = (yuv & 0xFF) - 128

    red = y + _trunc_div_65536(-3 * u + 74700 * v)
    green = y + _trunc_div_65536(-25861 * u - 38044 * v)
    blue = y + _trunc_div_65536(133169 * u - 32 * v)

    out = np.zeros(picture.shape + (4,), dtype=np.uint8)
    out[..., 0] = np.clip(blue, 0, 255)
    out[..., 1] = np.clip(green, 0, 255)
    out[..., 2] = np.clip(red, 0, 255)
    # A fully zero YUV word is the decoder's "nothing here" marker and renders
    # as transparent black rather than going through the matrix.
    out[yuv == 0] = 0
    return out


def _trunc_div_65536(value: np.ndarray) -> np.ndarray:
    """C integer division by 65536: truncates toward zero, not down."""
    return np.where(value >= 0, value >> 16, -((-value) >> 16))
