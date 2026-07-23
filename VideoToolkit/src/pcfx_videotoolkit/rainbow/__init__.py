"""RAINBOW (HuC6271) bitstream handling: block layout, discovery, decoding."""

from __future__ import annotations

from .blocks import (
    BLOCK_TYPES,
    DCT_BLOCK_TYPES,
    FRAME_WIDTH,
    RLE_BLOCK_TYPES,
    STRIP_HEIGHT,
    Block,
    parse_block,
)
from .discover import DiscoveredStream, FrameRef, discover_track
from .scan import Frame, Stream, scan_window

__all__ = [
    "BLOCK_TYPES",
    "Block",
    "DCT_BLOCK_TYPES",
    "DiscoveredStream",
    "FRAME_WIDTH",
    "Frame",
    "FrameRef",
    "RLE_BLOCK_TYPES",
    "STRIP_HEIGHT",
    "Stream",
    "discover_track",
    "parse_block",
    "scan_window",
]
