"""RAINBOW block ("strip") layout primitives.

A RAINBOW packet is a chain of blocks. Each block is:

    FF <type> <BE16 size> <payload...>

`size` counts itself (2 bytes) plus the payload, so the payload length is
`size - 2` and the block ends at `start + 4 + (size - 2)`.

Type meanings, from `rainbow_accurate.c`:

    0xFF  DCT block that carries a fresh 128-byte quant table pair
    0xF8  DCT block that reuses the previous quant tables
    0xF0..0xF3  palettized RLE block (palette lives in KING KRAM, not on disc)

One block covers 16 rasters. A frame therefore starts at a 0xFF block and runs
until the next one; the frame height is `strips * 16` and the width is always
256, because the decoder always walks 16 macroblock columns.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

DCT_BLOCK_TYPES = frozenset({0xF8, 0xFF})
RLE_BLOCK_TYPES = frozenset({0xF0, 0xF1, 0xF2, 0xF3})
BLOCK_TYPES = DCT_BLOCK_TYPES | RLE_BLOCK_TYPES

QUANT_TABLE_BYTES = 128
STRIP_HEIGHT = 16
FRAME_WIDTH = 256
MACROBLOCK_COLUMNS = FRAME_WIDTH // 16

#: Zero bytes the hardware wants between blocks. `C6272_2.WRI` asks for three
#: 0000h dummy words; Team Innocent's original stream advances 8 bytes.
DEFAULT_MAX_INTERBLOCK_PAD = 64

_BE16 = struct.Struct(">h")


@dataclass(slots=True)
class Block:
    start: int
    block_type: int
    size_field: int
    end: int

    @property
    def is_dct(self) -> bool:
        return self.block_type in DCT_BLOCK_TYPES

    @property
    def carries_quant_tables(self) -> bool:
        return self.block_type == 0xFF

    @property
    def payload_size(self) -> int:
        return self.size_field - 2


def parse_block(data: bytes, pos: int, limit: int | None = None) -> Block | None:
    """Parse one block at `pos`, or return None if there is not one there."""
    end_limit = len(data) if limit is None else min(limit, len(data))
    if pos < 0 or pos + 4 > end_limit:
        return None
    if data[pos] != 0xFF:
        return None
    block_type = data[pos + 1]
    if block_type not in BLOCK_TYPES:
        return None

    (size_field,) = _BE16.unpack_from(data, pos + 2)
    if block_type == 0xFF and size_field <= 2:
        # `rainbow_accurate.c:479` still consumes a quant table pair here, then
        # keeps looking for a block with real payload.
        end = pos + 4 + QUANT_TABLE_BYTES
        if end > end_limit:
            return None
        return Block(start=pos, block_type=block_type, size_field=size_field, end=end)

    if size_field < 2:
        return None
    end = pos + 4 + (size_field - 2)
    if end > end_limit:
        return None
    return Block(start=pos, block_type=block_type, size_field=size_field, end=end)


def skip_interblock_pad(
    data: bytes,
    pos: int,
    limit: int | None = None,
    max_pad: int = DEFAULT_MAX_INTERBLOCK_PAD,
) -> int:
    """Advance past the zero-fill that separates blocks."""
    end_limit = len(data) if limit is None else min(limit, len(data))
    stop = min(end_limit, pos + max_pad)
    while pos < stop and data[pos] == 0x00:
        pos += 1
    return pos
