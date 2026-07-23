"""Per-stream manifests.

A manifest has to carry everything a later encoder needs to put a replacement
video back where the original came from, without disturbing the surrounding
bytes: exact frame offsets and slot sizes, the container's record layout, the
chunk index table if there is one, the original quant tables, and the transfer
budget the replacement must stay inside.

Frame tables are stored as parallel integer arrays rather than a list of
objects for space saving and easier JSON diffing.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from ..rainbow.discover import DiscoveredStream

MANIFEST_VERSION = 1
USER_SECTOR_SIZE = 2048


def parse_fps(fps: str) -> Fraction:
    text = fps.strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return Fraction(int(numerator), int(denominator))
    return Fraction(text)


def _window_peak(sizes: list[int], window: int) -> int:
    """Largest total over any `window` consecutive frames."""
    if not sizes:
        return 0
    window = max(1, min(window, len(sizes)))
    running = sum(sizes[:window])
    peak = running
    for index in range(window, len(sizes)):
        running += sizes[index] - sizes[index - window]
        if running > peak:
            peak = running
    return peak


def _median(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def build_manifest(
    stream: DiscoveredStream,
    cue_path: Path,
    prefix: str,
    fps: str,
    outputs: dict[str, str],
    quant_tables: bytes | None = None,
    audio: dict[str, object] | None = None,
    fps_source: str = "default",
) -> dict[str, object]:
    frames = stream.frames
    video_sizes = [frame.video_size for frame in frames]
    slot_sizes = [frame.slot_size for frame in frames]
    rate = parse_fps(fps)
    window = max(1, round(float(rate)))

    frame_table: dict[str, object] = {
        "video_start": [frame.video_start for frame in frames],
        "video_size": video_sizes,
    }
    if any(frame.slot_size != frame.video_size for frame in frames):
        # Raw streams pad between frames; the encoder must preserve those gaps.
        frame_table["slot_size"] = slot_sizes
    if stream.has_audio:
        frame_table["audio_size"] = [frame.audio_size for frame in frames]
    if frames[0].record_offset is not None:
        frame_table["record_stride"] = frames[0].video_start - frames[0].record_offset
    if not all(frame.quant_start for frame in frames):
        frame_table["quant_start"] = [int(frame.quant_start) for frame in frames]

    return {
        "manifest_version": MANIFEST_VERSION,
        "prefix": prefix,
        "cue": str(cue_path),
        "track": stream.track,
        "track_mode": stream.track_mode,
        "cue_index": stream.cue_index,
        "kind": stream.kind,
        "detector": stream.detector,
        "geometry": {
            "width": stream.width,
            "height": stream.height,
            "strips_per_frame": stream.strips,
            "frames": stream.frame_count,
            "fps": fps,
            # "audio_packet_rate" is exact; "default" is a guess, because a
            # headerless RAINBOW stream does not record its frame rate.
            "fps_source": fps_source,
        },
        "location": {
            # All offsets are in cooked space: lsn * 2048 + offset-in-sector.
            "source_start": stream.source_start,
            "source_end": stream.source_end,
            "byte_size": stream.byte_size,
            "start_lsn": stream.start_lsn,
            "sector_offset": stream.sector_offset,
            "first_frame_lsn": stream.first_frame_lsn,
            "first_frame_offset": stream.first_frame_offset,
            "end_lsn_exclusive": stream.end_lsn_exclusive,
            "user_sector_size": USER_SECTOR_SIZE,
        },
        "budget": {
            # What a replacement stream has to fit inside. The slot figures are
            # the hard limit for layout-preserving patching; the per-second
            # peaks are the KING->RAINBOW transfer rate the original sustained.
            "min_video_bytes": min(video_sizes),
            "median_video_bytes": _median(video_sizes),
            "max_video_bytes": max(video_sizes),
            "min_slot_bytes": min(slot_sizes),
            "median_slot_bytes": _median(slot_sizes),
            "max_slot_bytes": max(slot_sizes),
            "total_video_bytes": sum(video_sizes),
            "peak_video_bytes_per_second": _window_peak(video_sizes, window),
            "peak_slot_bytes_per_second": _window_peak(slot_sizes, window),
        },
        "container_header": stream.header,
        "chunk_table": stream.chunk_table,
        "first_frame_quant_tables": quant_tables.hex() if quant_tables else None,
        "audio": audio,
        "frame_table": frame_table,
        "notes": stream.notes,
        "outputs": outputs,
    }


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
