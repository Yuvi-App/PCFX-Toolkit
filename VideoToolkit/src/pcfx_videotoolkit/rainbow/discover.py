"""Track-level discovery: find every RAINBOW stream on a data track.

Containers are located first, then headerless RAINBOW runs.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

from ..disc.cue import CueTrack, containing_index, data_index_ranges
from ..disc.sector import iter_user_windows, read_user_range
from . import containers as container_mod
from .blocks import FRAME_WIDTH, STRIP_HEIGHT, parse_block, skip_interblock_pad
from .scan import (
    DEFAULT_MAX_FRAME_GAP,
    DEFAULT_MIN_FRAMES,
    DEFAULT_MIN_STRIPS,
    dedupe_frames,
    group_frames_into_streams,
    scan_window,
)

USER_SECTOR_SIZE = 2048
DEFAULT_CHUNK_TABLE_SCAN_BACK = 0x4000
MAX_CONTAINER_BYTES = 96 * 1024 * 1024


@dataclass(slots=True)
class FrameRef:
    """One frame, in absolute cooked-track offsets."""

    index: int
    video_start: int
    video_end: int
    slot_end: int
    strips: int
    audio_start: int | None = None
    audio_end: int | None = None
    record_offset: int | None = None
    #: True when this frame's first block carries its own quant tables, so
    #: decoding can start here without replaying earlier frames.
    quant_start: bool = True

    @property
    def video_size(self) -> int:
        return self.video_end - self.video_start

    @property
    def slot_size(self) -> int:
        return self.slot_end - self.video_start

    @property
    def audio_size(self) -> int:
        if self.audio_start is None or self.audio_end is None:
            return 0
        return self.audio_end - self.audio_start


@dataclass(slots=True)
class DiscoveredStream:
    track: int
    track_mode: str
    kind: str
    detector: str
    source_start: int
    source_end: int
    frames: list[FrameRef]
    strips: int
    header: dict[str, object] | None = None
    chunk_table: dict[str, object] | None = None
    cue_index: object = ""
    index: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def first_frame_offset(self) -> int:
        return self.frames[0].video_start

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def width(self) -> int:
        return FRAME_WIDTH

    @property
    def height(self) -> int:
        return self.strips * STRIP_HEIGHT

    @property
    def has_audio(self) -> bool:
        return any(frame.audio_size for frame in self.frames)

    @property
    def start_lsn(self) -> int:
        return self.source_start // USER_SECTOR_SIZE

    @property
    def sector_offset(self) -> int:
        return self.source_start % USER_SECTOR_SIZE

    @property
    def first_frame_lsn(self) -> int:
        return self.first_frame_offset // USER_SECTOR_SIZE

    @property
    def end_lsn_exclusive(self) -> int:
        return -(-self.source_end // USER_SECTOR_SIZE)

    @property
    def byte_size(self) -> int:
        return self.source_end - self.source_start


def walk_chain(data: bytes, pos: int, limit: int | None = None) -> tuple[int, int]:
    """Walk the block chain at `pos`; return `(block count, end offset)`."""
    head = parse_block(data, pos, limit)
    if head is None:
        return 0, pos
    strips = 1
    cursor = head.end
    while True:
        probe = skip_interblock_pad(data, cursor, limit)
        block = parse_block(data, probe, limit)
        if block is None or block.carries_quant_tables:
            break
        strips += 1
        cursor = block.end
    return strips, cursor


def count_strips(data: bytes, pos: int) -> int:
    """Count blocks in the frame starting at `pos` within `data`."""
    return walk_chain(data, pos)[0]


def _find_container_offsets(track: CueTrack, start_lsn: int, count: int) -> list[int]:
    """Cheap sweep for movie header signatures, in cooked-track offsets."""
    offsets: set[int] = set()
    for window_start_lsn, data in iter_user_windows(track, start_lsn, count):
        base = window_start_lsn * USER_SECTOR_SIZE
        for _kind, signature in container_mod.MOVIE_SIGNATURES:
            search_from = 0
            while True:
                found = data.find(signature, search_from)
                if found < 0:
                    break
                offsets.add(base + found)
                search_from = found + 1
    return sorted(offsets)


def _read_container(
    track: CueTrack,
    offset: int,
    limit: int,
) -> tuple[bytes, int]:
    """Read enough cooked bytes to parse a container that begins at `offset`."""
    start_lsn = offset // USER_SECTOR_SIZE
    inner = offset % USER_SECTOR_SIZE
    available = (track.logical_sector_count - start_lsn) * USER_SECTOR_SIZE - inner
    want = min(limit, available, MAX_CONTAINER_BYTES)
    sectors = min(
        track.logical_sector_count - start_lsn,
        -(-(inner + want) // USER_SECTOR_SIZE),
    )
    with track.file_path.open("rb") as handle:
        data = read_user_range(handle, track, start_lsn, sectors)
    return data[inner:], start_lsn * USER_SECTOR_SIZE + inner


def discover_containers(
    track: CueTrack,
    start_lsn: int,
    count: int,
    min_frames: int,
) -> list[DiscoveredStream]:
    offsets = _find_container_offsets(track, start_lsn, count)
    track_end = track.logical_sector_count * USER_SECTOR_SIZE
    streams: list[DiscoveredStream] = []
    consumed_to = -1

    for position, offset in enumerate(offsets):
        if offset < consumed_to:
            continue
        next_offset = offsets[position + 1] if position + 1 < len(offsets) else track_end
        limit = max(next_offset - offset, USER_SECTOR_SIZE)
        data, base = _read_container(track, offset, limit)

        try:
            container = container_mod.parse(data)
        except Exception:
            continue
        if len(container.frames) < min_frames:
            continue
        if container.parsed_length_rounded > len(data):
            continue

        strips = count_strips(data, container.frames[0].video_start)
        # A container's record gives the size of the *slot* the video packet
        # occupies, and MpConv2 pads the compressed data out to fill it. Walking
        # the block chain finds where the real payload stops, so `video_end` is
        # the payload end and `slot_end` is the slot end -- the same meaning
        # those fields carry for headerless streams.
        frames = []
        for frame in container.frames:
            _strips, chain_end = walk_chain(data, frame.video_start, frame.video_end)
            if chain_end <= frame.video_start:
                chain_end = frame.video_end
            frames.append(
                FrameRef(
                    index=frame.index,
                    video_start=base + frame.video_start,
                    video_end=base + chain_end,
                    slot_end=base + frame.video_end,
                    strips=strips,
                    audio_start=None if frame.audio_start is None else base + frame.audio_start,
                    audio_end=None if frame.audio_end is None else base + frame.audio_end,
                    record_offset=base + frame.record_offset,
                    quant_start=data[frame.video_start + 1] == 0xFF,
                )
            )

        streams.append(
            DiscoveredStream(
                track=track.number,
                track_mode=track.mode,
                kind=container.kind,
                detector=f"movie_header:{container.kind}",
                source_start=base,
                source_end=base + container.parsed_length_rounded,
                frames=frames,
                strips=strips,
                header=container.header.to_dict(),
            )
        )
        consumed_to = base + container.parsed_length_rounded

    return streams


def discover_raw(
    track: CueTrack,
    start_lsn: int,
    count: int,
    covered: list[tuple[int, int]],
    min_frames: int,
    min_strips: int,
    max_frame_gap: int,
) -> list[DiscoveredStream]:
    collected = []
    for window_start_lsn, data in iter_user_windows(track, start_lsn, count):
        collected.extend(
            scan_window(
                data,
                base=window_start_lsn * USER_SECTOR_SIZE,
                min_frames=min_frames,
                min_strips=min_strips,
                max_frame_gap=max_frame_gap,
            )
        )

    frames = dedupe_frames(collected)
    frames = [
        frame
        for frame in frames
        if not any(start <= frame.video_start < end for start, end in covered)
    ]

    streams: list[DiscoveredStream] = []
    for group in group_frames_into_streams(
        frames, max_frame_gap=max_frame_gap, min_frames=min_frames
    ):
        refs = [
            FrameRef(
                index=frame.index,
                video_start=frame.video_start,
                video_end=frame.video_end,
                slot_end=frame.slot_end,
                strips=frame.strips,
            )
            for frame in group.frames
        ]
        streams.append(
            DiscoveredStream(
                track=track.number,
                track_mode=track.mode,
                kind=group.kind,
                detector="raw_dct_strip_run",
                source_start=group.start,
                source_end=group.end,
                frames=refs,
                strips=group.strips,
            )
        )
    return streams


def detect_chunk_table(
    track: CueTrack,
    stream: DiscoveredStream,
    scan_back: int = DEFAULT_CHUNK_TABLE_SCAN_BACK,
    min_entries: int = 8,
) -> dict[str, object] | None:
    """Look for an offset/size index sitting just before a raw stream."""
    if len(stream.frames) < min_entries:
        return None

    first_offset = stream.first_frame_offset
    window_start = max(0, first_offset - scan_back)
    start_lsn = window_start // USER_SECTOR_SIZE
    end_lsn = -(-first_offset // USER_SECTOR_SIZE)
    if end_lsn <= start_lsn:
        return None
    with track.file_path.open("rb") as handle:
        data = read_user_range(handle, track, start_lsn, end_lsn - start_lsn)
    base = start_lsn * USER_SECTOR_SIZE

    frame_offsets = [frame.video_start for frame in stream.frames]
    frame_sizes = [frame.video_size for frame in stream.frames]
    slot_sizes = [frame.slot_size for frame in stream.frames]
    first_local = first_offset - base
    best: dict[str, object] | None = None
    best_score = -1

    for row_size in (8, 4):
        last_start = first_local - (row_size * min_entries)
        for row_start in range(0, max(0, last_start) + 1, 4):
            max_entries = (first_local - row_start) // row_size
            entry_count = min(max_entries, len(frame_offsets))
            if entry_count < min_entries:
                continue
            compare_count = min(entry_count, 64)

            for endian, endian_label in (("<", "le"), (">", "be")):
                for base_label, bias in (
                    ("track_start", 0),
                    ("first_frame", first_offset),
                    ("table_start", base + row_start),
                ):
                    matches = 0
                    size_matches = 0
                    previous = -1
                    valid = True

                    for index in range(compare_count):
                        entry_offset = row_start + (index * row_size)
                        if entry_offset + row_size > first_local:
                            valid = False
                            break
                        (value,) = struct.unpack_from(f"{endian}I", data, entry_offset)
                        if value != frame_offsets[index] - bias or value < previous:
                            valid = False
                            break
                        previous = value
                        matches += 1
                        if row_size == 8:
                            (size,) = struct.unpack_from(f"{endian}I", data, entry_offset + 4)
                            if size in (frame_sizes[index], slot_sizes[index]):
                                size_matches += 1
                            else:
                                valid = False
                                break

                    if not valid or (row_size == 8 and size_matches < min_entries):
                        continue

                    score = (matches * 2) + size_matches
                    if entry_count == len(frame_offsets):
                        score += len(frame_offsets)
                    if score > best_score:
                        best_score = score
                        best = {
                            "table_start": base + row_start,
                            "table_end": first_offset,
                            "table_size": first_offset - (base + row_start),
                            "entry_size": row_size,
                            "entries_observed": entry_count,
                            "entries_compared": compare_count,
                            "endian": endian_label,
                            "offset_base": base_label,
                            "confidence": (
                                "high" if entry_count == len(frame_offsets) else "medium"
                            ),
                        }
    return best


def discover_track(
    cue_path: Path,
    track_number: int,
    start_lsn: int = 0,
    sector_count: int | None = None,
    min_frames: int = DEFAULT_MIN_FRAMES,
    min_strips: int = DEFAULT_MIN_STRIPS,
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
    find_chunk_tables: bool = True,
    min_container_frames: int = 1,
) -> list[DiscoveredStream]:
    """Find every RAINBOW stream on one data track. Safe to run in a worker."""
    from ..disc.cue import parse_cue

    track = parse_cue(cue_path).track(track_number)
    if not track.is_data:
        return []
    total = track.logical_sector_count
    count = total - start_lsn if sector_count is None else sector_count
    if count <= 0:
        return []

    # A container has an explicit header and record chain, so it does not need
    # the run-length evidence that a headerless scan does. TRC-uty packs are
    # often only a handful of frames.
    streams = discover_containers(track, start_lsn, count, min_container_frames)
    covered = [(item.source_start, item.source_end) for item in streams]
    streams.extend(
        discover_raw(
            track,
            start_lsn,
            count,
            covered,
            min_frames=min_frames,
            min_strips=min_strips,
            max_frame_gap=max_frame_gap,
        )
    )
    streams.sort(key=lambda item: item.first_frame_offset)

    ranges = data_index_ranges(track)
    for stream in streams:
        stream.cue_index = containing_index(ranges, stream.first_frame_lsn)
        if find_chunk_tables and stream.detector == "raw_dct_strip_run":
            stream.chunk_table = detect_chunk_table(track, stream)
        if stream.kind == "rle_rainbow":
            stream.notes.append(
                "palettized RLE stream; the palette lives in KING KRAM and is not on the disc"
            )
    return streams
