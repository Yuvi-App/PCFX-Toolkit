"""Discovery of RAINBOW frames and streams inside a data track.

The rule that makes this title-agnostic: a 0xFF block carries quant tables and
therefore begins a frame; every following non-0xFF block continues it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .blocks import (
    DEFAULT_MAX_INTERBLOCK_PAD,
    STRIP_HEIGHT,
    Block,
    FRAME_WIDTH,
    parse_block,
    skip_interblock_pad,
)

DEFAULT_MIN_FRAMES = 8
DEFAULT_MIN_STRIPS = 4
DEFAULT_MAX_FRAME_GAP = 0x20000


@dataclass(slots=True)
class Frame:
    """One decoded-picture worth of blocks, in absolute cooked offsets."""

    index: int
    video_start: int
    video_end: int
    slot_end: int
    strips: int
    first_size_field: int
    block_types: bytes
    blocks: list[Block] | None = None

    @property
    def video_size(self) -> int:
        return self.video_end - self.video_start

    @property
    def slot_size(self) -> int:
        return self.slot_end - self.video_start

    @property
    def post_video_pad(self) -> int:
        return self.slot_end - self.video_end

    @property
    def has_rle(self) -> bool:
        return any(b in (0xF0, 0xF1, 0xF2, 0xF3) for b in self.block_types)


@dataclass(slots=True)
class Stream:
    """A contiguous run of frames with a consistent geometry."""

    frames: list[Frame] = field(default_factory=list)

    @property
    def start(self) -> int:
        return self.frames[0].video_start

    @property
    def end(self) -> int:
        return self.frames[-1].slot_end

    @property
    def strips(self) -> int:
        return self.frames[0].strips

    @property
    def width(self) -> int:
        return FRAME_WIDTH

    @property
    def height(self) -> int:
        return self.strips * STRIP_HEIGHT

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def kind(self) -> str:
        return "rle_rainbow" if self.frames[0].has_rle else "raw_rainbow"


def parse_frame(
    data: bytes,
    pos: int,
    base: int = 0,
    max_pad: int = DEFAULT_MAX_INTERBLOCK_PAD,
    keep_blocks: bool = False,
) -> tuple[Frame, int] | None:
    """Parse a frame starting at `pos`; return it plus the offset after it.

    `base` is added to every recorded offset so callers can work in absolute
    cooked-space coordinates while scanning a window.
    """
    head = parse_block(data, pos)
    if head is None or not head.carries_quant_tables:
        return None

    blocks = [head]
    cursor = head.end
    while True:
        probe = skip_interblock_pad(data, cursor, max_pad=max_pad)
        block = parse_block(data, probe)
        if block is None or block.carries_quant_tables:
            break
        blocks.append(block)
        cursor = block.end

    frame = Frame(
        index=0,
        video_start=base + head.start,
        video_end=base + cursor,
        slot_end=base + cursor,
        strips=len(blocks),
        first_size_field=head.size_field,
        block_types=bytes(b.block_type for b in blocks),
        blocks=(
            [Block(b.start + base, b.block_type, b.size_field, b.end + base) for b in blocks]
            if keep_blocks
            else None
        ),
    )
    return frame, cursor


def _run_length(
    data: bytes,
    pos: int,
    strips: int,
    max_pad: int,
    limit: int = 2,
) -> int:
    """How many back-to-back frames of `strips` strips start at `pos`."""
    count = 0
    cursor = pos
    while count < limit:
        parsed = parse_frame(data, cursor, max_pad=max_pad)
        if parsed is None or parsed[0].strips != strips:
            break
        count += 1
        cursor = parsed[1]
        probe = data.find(b"\xff\xff", cursor, cursor + 64)
        if probe < 0:
            break
        cursor = probe
    return count


def _next_frame_start(
    data: bytes,
    pos: int,
    stop: int,
    expected_strips: int,
    max_pad: int,
    min_strips: int,
) -> int:
    """Find where the next real frame begins, skipping false FF FF hits.

    Compressed payloads are 0xFF-escaped, so a literal FF FF pair inside one is
    impossible; the false hits live in inter-frame padding. A candidate is
    accepted when it has the stream's strip count, or when it starts a run of
    its own geometry (a genuine geometry change, which ends this stream).
    """
    limit = min(stop + 2, len(data))
    probe = data.find(b"\xff\xff", pos, limit)
    while probe >= 0:
        parsed = parse_frame(data, probe, max_pad=max_pad)
        if parsed is not None and parsed[0].strips >= min_strips:
            strips = parsed[0].strips
            if strips == expected_strips or _run_length(data, probe, strips, max_pad) >= 2:
                return probe
        probe = data.find(b"\xff\xff", probe + 1, limit)
    return -1


def walk_frames(
    data: bytes,
    pos: int,
    base: int = 0,
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
    max_pad: int = DEFAULT_MAX_INTERBLOCK_PAD,
    min_strips: int = DEFAULT_MIN_STRIPS,
    keep_blocks: bool = False,
) -> list[Frame]:
    """Collect consecutive frames from `pos` until the geometry changes."""
    frames: list[Frame] = []
    strips: int | None = None
    cursor = pos

    while cursor + 4 <= len(data):
        parsed = parse_frame(data, cursor, base=base, max_pad=max_pad, keep_blocks=keep_blocks)
        if parsed is None:
            break
        frame, after = parsed
        if frame.strips < min_strips:
            break
        if strips is None:
            strips = frame.strips
        elif frame.strips != strips:
            break

        next_start = _next_frame_start(
            data,
            after,
            min(len(data), after + max_frame_gap),
            expected_strips=strips,
            max_pad=max_pad,
            min_strips=min_strips,
        )
        if next_start >= 0:
            frame.slot_end = base + next_start
        frame.index = len(frames)
        frames.append(frame)

        if next_start < 0:
            break
        cursor = next_start

    return frames


def scan_window(
    data: bytes,
    base: int = 0,
    min_frames: int = DEFAULT_MIN_FRAMES,
    min_strips: int = DEFAULT_MIN_STRIPS,
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
    max_pad: int = DEFAULT_MAX_INTERBLOCK_PAD,
    keep_blocks: bool = False,
) -> list[Frame]:
    """Find every frame in one cooked window, in absolute offsets."""
    found: list[Frame] = []
    cursor = 0
    length = len(data)

    while cursor + 4 <= length:
        start = data.find(b"\xff\xff", cursor)
        if start < 0:
            break
        frames = walk_frames(
            data,
            start,
            base=base,
            max_frame_gap=max_frame_gap,
            max_pad=max_pad,
            min_strips=min_strips,
            keep_blocks=keep_blocks,
        )
        if len(frames) >= min_frames:
            found.extend(frames)
            cursor = max(frames[-1].slot_end - base, start + 2)
        else:
            cursor = start + 2

    return found


def group_frames_into_streams(
    frames: list[Frame],
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
    min_frames: int = DEFAULT_MIN_FRAMES,
) -> list[Stream]:
    """Group globally de-duplicated frames into contiguous streams.

    A stream breaks when the strip count changes or the distance to the next
    frame exceeds `max_frame_gap`.
    """
    ordered = sorted(frames, key=lambda item: item.video_start)
    streams: list[Stream] = []
    current: list[Frame] = []

    for frame in ordered:
        if current:
            previous = current[-1]
            contiguous = (
                frame.strips == previous.strips
                and previous.video_end <= frame.video_start
                and frame.video_start - previous.video_end <= max_frame_gap
            )
            if not contiguous:
                if len(current) >= min_frames:
                    streams.append(_finish(current))
                current = []
        current.append(frame)

    if len(current) >= min_frames:
        streams.append(_finish(current))
    return streams


def _finish(frames: list[Frame]) -> Stream:
    for index, frame in enumerate(frames):
        frame.index = index
        following = frames[index + 1] if index + 1 < len(frames) else None
        frame.slot_end = following.video_start if following else frame.video_end
    return Stream(frames=frames)


def dedupe_frames(frames: list[Frame]) -> list[Frame]:
    """Drop frames re-found in an overlapping window."""
    seen: dict[int, Frame] = {}
    for frame in frames:
        existing = seen.get(frame.video_start)
        if existing is None or frame.video_end > existing.video_end:
            seen[frame.video_start] = frame
    return [seen[key] for key in sorted(seen)]
