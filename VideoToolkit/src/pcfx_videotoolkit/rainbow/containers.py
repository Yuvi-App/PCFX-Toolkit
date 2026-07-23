"""Text-headered PC-FX movie containers.

0x30-byte text preamble followed by a binary field block introduced by 0x1A:

    ARS Ver. 1.01v / IDCT / 32k Monoral     interleaved video + KING ADPCM
    TYY2TRC        / IDCT / TYY -> TRC      video only, pointer-chained records
    TRC uty 1.14a  / IDCT / Merged 10file   video only, same record chain

Blue Breaker uses all three. Everything else so far stores headerless
RAINBOW streams.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field

from .. import PcfxError
from .blocks import BLOCK_TYPES

MOVIE_SIGNATURES: tuple[tuple[str, bytes], ...] = (
    ("tyy2trc", b"TYY2TRC"),
    ("ars", b"ARS Ver."),
    ("trc_uty", b"TRC uty"),
)

USER_SECTOR_SIZE = 2048

_RATE_RE = re.compile(r"(\d+)\s*k", re.IGNORECASE)
_AUDIO_RATE_CODES = {0x20: 32000, 0x10: 16000, 0x08: 8000, 0x04: 4000}


@dataclass(slots=True)
class MovieHeader:
    kind: str
    lines: list[str]
    marker_offset: int
    raw: bytes
    count: int
    line_count: int
    audio_flag: int
    audio_rate_code: int

    @property
    def audio_sample_rate(self) -> int:
        """Sample rate in Hz, 0 when the container carries no audio."""
        from_code = _AUDIO_RATE_CODES.get(self.audio_rate_code, 0)
        if from_code:
            return from_code
        match = _RATE_RE.search(self.lines[2] if len(self.lines) > 2 else "")
        if match:
            return int(match.group(1)) * 1000
        return 0

    @property
    def audio_channels(self) -> int:
        text = (self.lines[2] if len(self.lines) > 2 else "").lower()
        if not self.audio_sample_rate:
            return 0
        return 1 if "mono" in text else 2

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "line0": self.lines[0],
            "line1": self.lines[1],
            "line2": self.lines[2],
            "count": self.count,
            "line_count": self.line_count,
            "audio_flag": self.audio_flag,
            "audio_rate_code": self.audio_rate_code,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_channels": self.audio_channels,
            "binary_header_hex": self.raw.hex(" "),
        }


@dataclass(slots=True)
class ContainerFrame:
    index: int
    record_offset: int
    video_start: int
    video_end: int
    audio_start: int | None
    audio_end: int | None
    record_words: tuple[int, ...] = ()

    @property
    def video_size(self) -> int:
        return self.video_end - self.video_start

    @property
    def audio_size(self) -> int:
        if self.audio_start is None or self.audio_end is None:
            return 0
        return self.audio_end - self.audio_start


@dataclass(slots=True)
class Container:
    kind: str
    header: MovieHeader
    frames: list[ContainerFrame] = field(default_factory=list)
    parsed_length: int = 0
    parsed_length_rounded: int = 0

    @property
    def has_audio(self) -> bool:
        return any(frame.audio_size for frame in self.frames)


def identify(data: bytes) -> str | None:
    for kind, signature in MOVIE_SIGNATURES:
        if data.startswith(signature):
            return kind
    return None


def round_up_to_sector(value: int, sector_size: int = USER_SECTOR_SIZE) -> int:
    return ((value + sector_size - 1) // sector_size) * sector_size


def parse_header(data: bytes) -> MovieHeader:
    kind = identify(data)
    if kind is None:
        raise PcfxError("Data does not start with a known movie signature")

    marker_offset = data.find(b"\x1a", 0x20, 0x50)
    if marker_offset < 0:
        marker_offset = 0x2F
    if marker_offset + 17 > len(data):
        raise PcfxError("Movie header is too short")

    lines = [
        item.decode("ascii", errors="replace").rstrip(" ")
        for item in data[:marker_offset].split(b"\r\n")
        if item
    ]
    while len(lines) < 3:
        lines.append("")

    raw = data[marker_offset : marker_offset + 17]
    # Field layout confirmed against Blue Breaker's ARS/TYY2TRC/TRC-uty headers:
    # the record count is a little-endian 16-bit value at +3, +9 holds the line
    # count (0xF0 = 240) and +13 encodes the ADPCM sample rate.
    (count,) = struct.unpack_from("<H", data, marker_offset + 3)
    return MovieHeader(
        kind=kind,
        lines=lines[:3],
        marker_offset=marker_offset,
        raw=raw,
        count=count,
        line_count=raw[9],
        audio_flag=raw[11],
        audio_rate_code=raw[13],
    )


def parse_ars(data: bytes) -> Container:
    """Interleaved video/audio records, 0x20-byte header per frame."""
    header = parse_header(data)
    record_offset = 0x100
    frames: list[ContainerFrame] = []

    while record_offset + 0x20 <= len(data):
        video_size, word1, word2, audio_size = struct.unpack_from("<IIII", data, record_offset)
        if video_size == 0 and audio_size == 0:
            break

        video_start = record_offset + 0x20
        video_end = video_start + video_size
        audio_start = video_end
        audio_end = audio_start + audio_size
        if audio_end > len(data) or audio_end <= record_offset:
            raise PcfxError(f"ARS frame {len(frames)} has invalid packet sizes")
        if not (data[video_start] == 0xFF and data[video_start + 1] in BLOCK_TYPES):
            raise PcfxError(f"ARS frame {len(frames)} does not start with a RAINBOW marker")

        frames.append(
            ContainerFrame(
                index=len(frames),
                record_offset=record_offset,
                video_start=video_start,
                video_end=video_end,
                audio_start=audio_start if audio_size else None,
                audio_end=audio_end if audio_size else None,
                record_words=(video_size, word1, word2, audio_size),
            )
        )
        record_offset = audio_end

    if not frames:
        raise PcfxError("ARS container had no frame records")

    parsed_length = frames[-1].audio_end or frames[-1].video_end
    rounded = round_up_to_sector(parsed_length)
    if rounded > len(data):
        raise PcfxError("ARS parsed length extends beyond loaded data")

    return Container(
        kind="ars",
        header=header,
        frames=frames,
        parsed_length=parsed_length,
        parsed_length_rounded=rounded,
    )


RECORD_STRIDE = 0x20
RECORD_GAP = 8


def parse_record_chain(data: bytes) -> Container:
    """TYY2TRC and TRC-uty: an 8-byte record 0x20 bytes ahead of each frame."""
    header = parse_header(data)
    record_offset = 0x100
    if len(data) < record_offset + RECORD_STRIDE:
        raise PcfxError(f"{header.kind} data is too short")

    size_plus_8, pointer = struct.unpack_from("<II", data, record_offset)
    if size_plus_8 < 8 or pointer == 0:
        raise PcfxError(f"{header.kind} first frame record is invalid")

    frames: list[ContainerFrame] = []

    while record_offset + RECORD_STRIDE <= len(data):
        size_plus_8, pointer = struct.unpack_from("<II", data, record_offset)
        if size_plus_8 == 0 and pointer == 0:
            break
        if size_plus_8 < 8:
            raise PcfxError(
                f"Invalid {header.kind} size field 0x{size_plus_8:x} at 0x{record_offset:x}"
            )

        frame_start = record_offset + RECORD_STRIDE
        frame_end = frame_start + (size_plus_8 - 8)
        if frame_end > len(data):
            raise PcfxError(f"{header.kind} frame at 0x{frame_start:x} runs past loaded data")
        if data[frame_start] != 0xFF:
            raise PcfxError(f"{header.kind} frame {len(frames)} has no RAINBOW marker")

        frames.append(
            ContainerFrame(
                index=len(frames),
                record_offset=record_offset,
                video_start=frame_start,
                video_end=frame_end,
                audio_start=None,
                audio_end=None,
                record_words=(size_plus_8, pointer),
            )
        )
        record_offset = frame_end + RECORD_GAP

    if not frames:
        raise PcfxError(f"{header.kind} container had no frame records")

    parsed_length = frames[-1].video_end
    return Container(
        kind=header.kind,
        header=header,
        frames=frames,
        parsed_length=parsed_length,
        parsed_length_rounded=round_up_to_sector(parsed_length),
    )


def parse(data: bytes) -> Container:
    kind = identify(data)
    if kind == "ars":
        return parse_ars(data)
    if kind in ("tyy2trc", "trc_uty"):
        return parse_record_chain(data)
    raise PcfxError("Unsupported or unknown movie container")


def find_containers(data: bytes, base: int = 0, min_frames: int = 1) -> list[dict[str, object]]:
    """Locate and parse every movie container inside a buffer."""
    results: list[dict[str, object]] = []
    seen: set[int] = set()

    for _kind, signature in MOVIE_SIGNATURES:
        search_from = 0
        while True:
            offset = data.find(signature, search_from)
            if offset < 0:
                break
            search_from = offset + 1
            if offset in seen:
                continue
            seen.add(offset)

            try:
                container = parse(data[offset:])
            except PcfxError:
                continue
            if len(container.frames) < min_frames:
                continue
            if offset + container.parsed_length_rounded > len(data):
                continue

            results.append(
                {
                    "container": container,
                    "source_start": base + offset,
                    "source_end": base + offset + container.parsed_length_rounded,
                }
            )

    results.sort(key=lambda item: int(item["source_start"]))
    return results
