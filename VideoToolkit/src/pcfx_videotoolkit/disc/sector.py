"""Reading user data out of CUE/BIN data tracks.

Everything downstream works in "cooked" space: a flat byte stream of 2048-byte
user areas concatenated in LSN order. `cooked_offset = lsn * 2048 + offset`.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

from .. import PcfxError
from .cue import CueTrack

DEFAULT_CHUNK_SECTORS = 4096
DEFAULT_WINDOW_SECTORS = 32768  # 64 MiB of cooked data
DEFAULT_OVERLAP_SECTORS = 2048  # 4 MiB, far larger than any single RAINBOW frame


def cook(raw: bytes, track: CueTrack) -> bytes:
    """Strip sync/header/ECC from raw sectors, returning only user bytes."""
    sector_size = track.sector_size
    user_size = track.user_sector_size
    if sector_size == user_size:
        return raw

    count = len(raw) // sector_size
    if count * sector_size != len(raw):
        raise PcfxError("Raw sector buffer is not a whole number of sectors")

    offset = track.user_offset_in_raw_sector
    view = np.frombuffer(raw, dtype=np.uint8, count=count * sector_size)
    view = view.reshape(count, sector_size)[:, offset : offset + user_size]
    return view.tobytes()


def _read_raw(handle: BinaryIO, track: CueTrack, start_lsn: int, count: int) -> bytes:
    total = track.logical_sector_count
    if start_lsn < 0 or count < 0 or start_lsn + count > total:
        raise PcfxError(
            f"Requested LSN range {start_lsn}..{start_lsn + count} is outside "
            f"track {track.number:02d} ({total} sectors)"
        )
    handle.seek((track.data_start_sector + start_lsn) * track.sector_size)
    expected = count * track.sector_size
    raw = handle.read(expected)
    if len(raw) != expected:
        raise PcfxError(
            f"Short read at LSN {start_lsn}: expected {expected}, got {len(raw)}"
        )
    return raw


def read_user_sector(handle: BinaryIO, track: CueTrack, lsn: int) -> bytes:
    return cook(_read_raw(handle, track, lsn, 1), track)


def read_user_range(
    handle: BinaryIO,
    track: CueTrack,
    start_lsn: int,
    count: int,
    chunk_sectors: int = DEFAULT_CHUNK_SECTORS,
) -> bytes:
    if count <= chunk_sectors:
        return cook(_read_raw(handle, track, start_lsn, count), track)

    parts: list[bytes] = []
    remaining = count
    lsn = start_lsn
    while remaining > 0:
        take = min(chunk_sectors, remaining)
        parts.append(cook(_read_raw(handle, track, lsn, take), track))
        lsn += take
        remaining -= take
    return b"".join(parts)


def read_user_range_from_path(
    track: CueTrack,
    start_lsn: int,
    count: int,
) -> bytes:
    """Convenience wrapper that opens the track file itself.

    Used by worker processes, which cannot inherit an open handle.
    """
    with track.file_path.open("rb") as handle:
        return read_user_range(handle, track, start_lsn, count)


def iter_user_sectors(
    track: CueTrack,
    start_lsn: int = 0,
    count: int | None = None,
    chunk_sectors: int = DEFAULT_CHUNK_SECTORS,
) -> Iterator[tuple[int, bytes]]:
    total = track.logical_sector_count
    if count is None:
        count = total - start_lsn

    user_size = track.user_sector_size
    with track.file_path.open("rb") as handle:
        for chunk_start in range(start_lsn, start_lsn + count, chunk_sectors):
            chunk_count = min(chunk_sectors, (start_lsn + count) - chunk_start)
            cooked = cook(_read_raw(handle, track, chunk_start, chunk_count), track)
            for index in range(chunk_count):
                offset = index * user_size
                yield chunk_start + index, cooked[offset : offset + user_size]


def iter_user_windows(
    track: CueTrack,
    start_lsn: int = 0,
    count: int | None = None,
    window_sectors: int = DEFAULT_WINDOW_SECTORS,
    overlap_sectors: int = DEFAULT_OVERLAP_SECTORS,
) -> Iterator[tuple[int, bytes]]:
    """Yield `(window_start_lsn, cooked_bytes)` windows that overlap.

    The overlap lets a sequential scanner resynchronise across window seams
    without ever holding a whole track in memory. Callers de-duplicate by
    absolute cooked offset.
    """
    total = track.logical_sector_count
    if count is None:
        count = total - start_lsn
    if count <= 0:
        return
    if overlap_sectors >= window_sectors:
        raise PcfxError("overlap_sectors must be smaller than window_sectors")

    end_lsn = start_lsn + count
    stride = window_sectors - overlap_sectors

    with track.file_path.open("rb") as handle:
        window_start = start_lsn
        while window_start < end_lsn:
            window_count = min(window_sectors, end_lsn - window_start)
            yield window_start, read_user_range(handle, track, window_start, window_count)
            if window_start + window_count >= end_lsn:
                break
            window_start += stride


def track_from_cue_path(cue_path: Path, track_number: int) -> CueTrack:
    """Re-resolve a track in a worker process from picklable inputs."""
    from .cue import parse_cue

    return parse_cue(cue_path).track(track_number)
