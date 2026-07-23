"""Orchestration: discover streams on a disc, then decode and export them.

Frames within a stream are decoded in a process pool and handed to a single
ffmpeg process in order. Work is bounded so memory stays flat no
matter how long the stream is
"""

from __future__ import annotations

import os
import re
import sys
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import PcfxError
from .audio.king_adpcm import decode_adpcm, write_wav
from .disc.cue import CueTrack, parse_cue
from .disc.sector import read_user_range
from .export.manifest import build_manifest, write_json
from .export.video import DEFAULT_CODEC, AviWriter, find_ffmpeg
from .rainbow.blocks import QUANT_TABLE_BYTES
from .rainbow.decoder import RainbowDecoder, yuv_to_bgra
from .rainbow.discover import (
    DEFAULT_MAX_FRAME_GAP,
    DEFAULT_MIN_FRAMES,
    DEFAULT_MIN_STRIPS,
    DiscoveredStream,
    discover_track,
)

DEFAULT_FPS = "15/1"
DEFAULT_CHUNK_FRAMES = 16
USER_SECTOR_SIZE = 2048

_TRACK_CACHE: dict[tuple[str, int], CueTrack] = {}


@dataclass
class ExtractOptions:
    cue: Path
    out_dir: Path
    track: int | None = None
    #: None means "work it out per stream"; see `resolve_fps`.
    fps: str | None = None
    fps_overrides: dict[int, str] = field(default_factory=dict)
    codec: str = DEFAULT_CODEC
    pixel_format: str = "bgr24"
    jobs: int | None = None
    chunk_frames: int = DEFAULT_CHUNK_FRAMES
    min_frames: int = DEFAULT_MIN_FRAMES
    min_strips: int = DEFAULT_MIN_STRIPS
    max_frame_gap: int = DEFAULT_MAX_FRAME_GAP
    mux_audio: bool = False
    no_audio: bool = False
    palettized: str = "black"
    keep_intermediate: bool = False
    dry_run: bool = False
    ffmpeg: str | None = None
    quiet: bool = False

    @property
    def worker_count(self) -> int:
        return self.jobs or max(1, (os.cpu_count() or 4))


def resolve_fps(stream: DiscoveredStream, options: ExtractOptions) -> tuple[str, str]:
    """Work out the frame rate to tag a stream with, and say where it came from.

    RAINBOW streams carry no frame rate. But an ARS container interleaves one
    ADPCM packet per frame, so the audio length pins the video length exactly:
    `fps = frames * sample_rate / (2 * audio_bytes)`, since each byte holds two
    4-bit samples. Even if the hardware's real ADPCM clock is not precisely the
    "32k" the header claims, deriving the rate this way keeps the AVI and the
    WAV the same length, which is what an editor cares about.
    """
    override = options.fps_overrides.get(stream.index)
    if override:
        return override, "override"
    if options.fps:
        return options.fps, "requested"

    if stream.has_audio and stream.header:
        sample_rate = int(stream.header.get("audio_sample_rate") or 0)
        channels = max(1, int(stream.header.get("audio_channels") or 1))
        audio_bytes = sum(frame.audio_size for frame in stream.frames)
        if sample_rate and audio_bytes:
            rate = Fraction(
                stream.frame_count * sample_rate * channels, 2 * audio_bytes
            ).limit_denominator(1000000)
            return f"{rate.numerator}/{rate.denominator}", "audio_packet_rate"

    return DEFAULT_FPS, "default"


def _track_for(cue_path: str, track_number: int) -> CueTrack:
    key = (cue_path, track_number)
    track = _TRACK_CACHE.get(key)
    if track is None:
        track = parse_cue(Path(cue_path)).track(track_number)
        _TRACK_CACHE[key] = track
    return track


def read_cooked(track: CueTrack, start: int, end: int) -> tuple[bytes, int]:
    """Read the cooked byte range `[start, end)`, returning it plus its base."""
    start_lsn = start // USER_SECTOR_SIZE
    end_lsn = -(-end // USER_SECTOR_SIZE)
    with track.file_path.open("rb") as handle:
        data = read_user_range(handle, track, start_lsn, end_lsn - start_lsn)
    return data, start_lsn * USER_SECTOR_SIZE


def decode_chunk(job: tuple) -> bytes:
    """Worker entry point: decode a run of frames into packed pixel bytes."""
    cue_path, track_number, strips, spans, pixel_format, rle_as_grey = job
    track = _track_for(cue_path, track_number)
    data, base = read_cooked(track, spans[0][0], spans[-1][1])

    decoder = RainbowDecoder(rle_as_grey=rle_as_grey)
    parts: list[bytes] = []
    for video_start, video_end in spans:
        packet = data[video_start - base : video_end - base]
        picture = decoder.decode(packet, strips=strips)
        pixels = yuv_to_bgra(picture.yuv)
        if pixel_format == "bgr24":
            pixels = np.ascontiguousarray(pixels[..., :3])
        parts.append(pixels.tobytes())
    return b"".join(parts)


def _chunk_spans(stream: DiscoveredStream, chunk_frames: int) -> list[list[tuple[int, int]]]:
    """Split a stream into decodable chunks.

    A chunk may only start on a frame that carries its own quant tables,
    otherwise it would inherit state from the frame before it.
    """
    chunks: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for frame in stream.frames:
        if current and len(current) >= chunk_frames and frame.quant_start:
            chunks.append(current)
            current = []
        current.append((frame.video_start, frame.video_end))
    if current:
        chunks.append(current)
    return chunks


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "stream"


def stream_prefix(stream: DiscoveredStream, index: int) -> str:
    return _safe_name(
        f"stream_{index:03d}_track{stream.track:02d}_"
        f"lsn_{stream.first_frame_lsn:06d}_{stream.kind}"
    )


def extract_audio(
    track: CueTrack,
    stream: DiscoveredStream,
    out_path: Path,
) -> dict[str, object] | None:
    """Decode a container's interleaved ADPCM packets into one WAV."""
    if not stream.has_audio or stream.header is None:
        return None
    sample_rate = int(stream.header.get("audio_sample_rate") or 0)
    channels = int(stream.header.get("audio_channels") or 0)
    if not sample_rate:
        return None

    data, base = read_cooked(track, stream.source_start, stream.source_end)
    packets = bytearray()
    for frame in stream.frames:
        if frame.audio_start is None or not frame.audio_size:
            continue
        packets += data[frame.audio_start - base : frame.audio_end - base]
    if not packets:
        return None

    samples = decode_adpcm(bytes(packets))
    write_wav(out_path, samples, sample_rate, channels or 1)
    return {
        "codec": "king_adpcm",
        "sample_rate": sample_rate,
        "channels": channels or 1,
        "packet_bytes": len(packets),
        "samples": int(samples.size),
        "duration_seconds": round(samples.size / (sample_rate * max(channels, 1)), 4),
        "path": out_path.name,
    }


def first_frame_quant_tables(track: CueTrack, stream: DiscoveredStream) -> bytes | None:
    frame = stream.frames[0]
    if not frame.quant_start:
        return None
    data, base = read_cooked(track, frame.video_start, frame.video_start + 4 + QUANT_TABLE_BYTES)
    start = frame.video_start - base + 4
    tables = data[start : start + QUANT_TABLE_BYTES]
    return tables if len(tables) == QUANT_TABLE_BYTES else None


def discover(options: ExtractOptions) -> list[DiscoveredStream]:
    """Scan every requested data track, one worker per track."""
    cue = parse_cue(options.cue)
    if options.track is not None:
        tracks = [cue.track(options.track)]
        if not tracks[0].is_data:
            raise PcfxError(f"Track {options.track:02d} is not a data track")
    else:
        tracks = cue.data_tracks()
    if not tracks:
        raise PcfxError("No MODE data tracks found in this CUE")

    numbers = [track.number for track in tracks]
    workers = min(options.worker_count, len(numbers))
    results: list[list[DiscoveredStream]]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(
                pool.map(
                    _discover_one,
                    [
                        (
                            options.cue,
                            number,
                            options.min_frames,
                            options.min_strips,
                            options.max_frame_gap,
                        )
                        for number in numbers
                    ],
                )
            )
    else:
        results = [
            _discover_one(
                (
                    options.cue,
                    number,
                    options.min_frames,
                    options.min_strips,
                    options.max_frame_gap,
                )
            )
            for number in numbers
        ]

    streams = [item for group in results for item in group]
    streams.sort(key=lambda item: (item.track, item.first_frame_offset))
    for index, stream in enumerate(streams):
        stream.index = index
    return streams


def _discover_one(job: tuple) -> list[DiscoveredStream]:
    cue_path, number, min_frames, min_strips, max_frame_gap = job
    return discover_track(
        cue_path,
        number,
        min_frames=min_frames,
        min_strips=min_strips,
        max_frame_gap=max_frame_gap,
    )


def extract(options: ExtractOptions) -> dict[str, object]:
    cue = parse_cue(options.cue)
    streams = discover(options)
    options.out_dir.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        if not options.quiet:
            print(message, file=sys.stderr, flush=True)

    log(f"{options.cue.name}: {len(streams)} stream(s) found")
    manifests: list[dict[str, object]] = []

    if options.dry_run:
        for stream in streams:
            fps, fps_source = resolve_fps(stream, options)
            manifests.append(
                build_manifest(
                    stream,
                    options.cue,
                    stream_prefix(stream, stream.index),
                    fps,
                    outputs={},
                    fps_source=fps_source,
                )
            )
        return _finish(options, cue, streams, manifests, log)

    ffmpeg = find_ffmpeg(options.ffmpeg)
    workers = options.worker_count
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for stream in streams:
            manifests.append(_export_stream(options, stream, pool, ffmpeg, log))

    return _finish(options, cue, streams, manifests, log)


def _export_stream(
    options: ExtractOptions,
    stream: DiscoveredStream,
    pool: ProcessPoolExecutor,
    ffmpeg: str,
    log,
) -> dict[str, object]:
    track = parse_cue(options.cue).track(stream.track)
    prefix = stream_prefix(stream, stream.index)
    fps, fps_source = resolve_fps(stream, options)
    outputs: dict[str, str] = {}

    audio_info = None
    audio_path = None
    if stream.has_audio and not options.no_audio:
        audio_path = options.out_dir / f"{prefix}.wav"
        audio_info = extract_audio(track, stream, audio_path)
        if audio_info:
            outputs["audio"] = audio_path.name
        else:
            audio_path = None

    video_path = options.out_dir / f"{prefix}.avi"
    chunks = _chunk_spans(stream, options.chunk_frames)
    max_inflight = max(2, options.worker_count * 2)
    job_template = (
        str(options.cue),
        stream.track,
        stream.strips,
        None,
        options.pixel_format,
        options.palettized == "greyscale",
    )

    log(
        f"  [{stream.index:03d}] {stream.kind:<11} track {stream.track:02d} "
        f"lsn {stream.first_frame_lsn:<6} {stream.frame_count:>5} frames "
        f"{stream.width}x{stream.height} @ {fps} -> {video_path.name}"
    )

    with AviWriter(
        video_path,
        width=stream.width,
        height=stream.height,
        fps=fps,
        codec=options.codec,
        pixel_format=options.pixel_format,
        ffmpeg=ffmpeg,
        audio_path=audio_path if options.mux_audio else None,
    ) as writer:
        pending: deque[Future] = deque()
        for spans in chunks:
            while len(pending) >= max_inflight:
                writer.write(pending.popleft().result())
            job = (job_template[0], job_template[1], job_template[2], spans, job_template[4], job_template[5])
            pending.append(pool.submit(decode_chunk, job))
        while pending:
            writer.write(pending.popleft().result())

    outputs["video"] = video_path.name
    if options.mux_audio and audio_path is not None:
        outputs["audio_muxed"] = video_path.name

    manifest = build_manifest(
        stream,
        options.cue,
        prefix,
        fps,
        outputs=outputs,
        quant_tables=first_frame_quant_tables(track, stream),
        audio=audio_info,
        fps_source=fps_source,
    )
    manifest_path = options.out_dir / f"{prefix}.json"
    write_json(manifest_path, manifest)
    outputs["manifest"] = manifest_path.name

    if options.keep_intermediate:
        _write_packets(options, track, stream, prefix, outputs)
    return manifest


def _write_packets(
    options: ExtractOptions,
    track: CueTrack,
    stream: DiscoveredStream,
    prefix: str,
    outputs: dict[str, str],
) -> None:
    """Dump the raw RAINBOW packets and a frame table, for debugging."""
    data, base = read_cooked(track, stream.source_start, stream.source_end)
    packets_path = options.out_dir / f"{prefix}.rainbow_packets.bin"
    with packets_path.open("wb") as handle:
        for frame in stream.frames:
            handle.write(data[frame.video_start - base : frame.video_end - base])
    outputs["packets"] = packets_path.name

    frames_path = options.out_dir / f"{prefix}.frames.tsv"
    lines = ["frame_index\tvideo_output_offset\tvideo_start\tvideo_end\tvideo_size\tslot_size\taudio_size"]
    offset = 0
    for frame in stream.frames:
        lines.append(
            f"{frame.index}\t{offset}\t{frame.video_start}\t{frame.video_end}\t"
            f"{frame.video_size}\t{frame.slot_size}\t{frame.audio_size}"
        )
        offset += frame.video_size
    frames_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs["frames"] = frames_path.name


def _finish(
    options: ExtractOptions,
    cue,
    streams: list[DiscoveredStream],
    manifests: list[dict[str, object]],
    log,
) -> dict[str, object]:
    report_path = options.out_dir / "rainbow_streams.tsv"
    fields = [
        "stream_index", "track", "track_mode", "kind", "detector", "cue_index",
        "start_lsn", "sector_offset", "first_frame_lsn", "end_lsn_exclusive",
        "frames", "width", "height", "strips", "fps", "byte_size",
        "max_video_bytes", "peak_video_bytes_per_second", "audio", "chunk_table",
    ]
    rows = [
        "\t".join(
            str(value)
            for value in (
                item["prefix"].split("_")[1],
                item["track"],
                item["track_mode"],
                item["kind"],
                item["detector"],
                item["cue_index"],
                item["location"]["start_lsn"],
                item["location"]["sector_offset"],
                item["location"]["first_frame_lsn"],
                item["location"]["end_lsn_exclusive"],
                item["geometry"]["frames"],
                item["geometry"]["width"],
                item["geometry"]["height"],
                item["geometry"]["strips_per_frame"],
                item["geometry"]["fps"],
                item["location"]["byte_size"],
                item["budget"]["max_video_bytes"],
                item["budget"]["peak_video_bytes_per_second"],
                "yes" if item["audio"] else "no",
                "yes" if item["chunk_table"] else "no",
            )
        )
        for item in manifests
    ]
    options.out_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\t".join(fields) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    summary = {
        "cue": str(options.cue),
        "out_dir": str(options.out_dir.resolve()),
        "track": options.track if options.track is not None else "all_data_tracks",
        "fps": options.fps or "auto",
        "codec": options.codec,
        "dry_run": options.dry_run,
        "streams": len(streams),
        "total_frames": sum(stream.frame_count for stream in streams),
        "report": report_path.name,
        "stream_manifests": [item["prefix"] for item in manifests],
    }
    write_json(options.out_dir / "pcfx_fmv_manifest.json", summary)
    log(
        f"done: {summary['streams']} stream(s), {summary['total_frames']} frames -> "
        f"{options.out_dir}"
    )
    return summary
