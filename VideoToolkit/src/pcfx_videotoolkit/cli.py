"""Command line entry point: `pcfx-fmv`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import PcfxError, __version__
from .export.video import CODECS, DEFAULT_CODEC
from .pipeline import DEFAULT_CHUNK_FRAMES, DEFAULT_FPS, ExtractOptions, discover, extract
from .rainbow.discover import DEFAULT_MAX_FRAME_GAP, DEFAULT_MIN_FRAMES, DEFAULT_MIN_STRIPS


def _parse_fps_overrides(values: list[str] | None) -> dict[int, str]:
    overrides: dict[int, str] = {}
    for item in values or []:
        if "=" not in item:
            raise PcfxError(f"--fps-for wants STREAM=FPS, got {item!r}")
        index, fps = item.split("=", 1)
        overrides[int(index)] = fps
    return overrides


def _add_scan_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cue", type=Path, required=True, help="path to the disc's .cue")
    parser.add_argument("--track", type=int, help="only this data track (default: all)")
    parser.add_argument(
        "--min-frames",
        type=int,
        default=DEFAULT_MIN_FRAMES,
        help="shortest headerless run to accept as a stream",
    )
    parser.add_argument(
        "--min-strips",
        type=int,
        default=DEFAULT_MIN_STRIPS,
        help="fewest blocks a frame may have",
    )
    parser.add_argument(
        "--max-frame-gap",
        type=lambda value: int(value, 0),
        default=DEFAULT_MAX_FRAME_GAP,
        help="largest padding gap tolerated between frames of one stream",
    )
    parser.add_argument("--jobs", type=int, help="worker processes (default: all cores)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcfx-fmv",
        description=(
            "Extract NEC PC-FX RAINBOW FMV streams from a CUE/BIN disc image "
            "into lossless AVI files."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"pcfx-videotoolkit {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="list the FMV streams on a disc without decoding")
    _add_scan_options(scan)
    scan.set_defaults(func=cmd_scan)

    extract_cmd = sub.add_parser("extract", help="decode every FMV stream to AVI")
    _add_scan_options(extract_cmd)
    extract_cmd.add_argument("--out", type=Path, required=True, help="output directory")
    extract_cmd.add_argument(
        "--fps",
        default=None,
        help=(
            "force a frame rate for every stream, e.g. 15/1 or 30000/1001. By "
            "default it is derived from the ADPCM packet rate where a container "
            f"has audio, and falls back to {DEFAULT_FPS} where it does not."
        ),
    )
    extract_cmd.add_argument(
        "--fps-for",
        action="append",
        metavar="STREAM=FPS",
        help="override the frame rate for one stream index; repeatable",
    )
    extract_cmd.add_argument(
        "--codec",
        default=DEFAULT_CODEC,
        choices=sorted(CODECS),
        help=f"video codec inside the AVI (default {DEFAULT_CODEC}, lossless)",
    )
    extract_cmd.add_argument(
        "--alpha",
        action="store_true",
        help="write 32-bit BGRA instead of 24-bit BGR (the alpha byte is always 0)",
    )
    extract_cmd.add_argument(
        "--mux-audio",
        action="store_true",
        help="put the decoded ADPCM into the AVI instead of a separate WAV",
    )
    extract_cmd.add_argument("--no-audio", action="store_true", help="skip audio entirely")
    extract_cmd.add_argument(
        "--palettized",
        choices=("black", "greyscale"),
        default="black",
        help=(
            "how to render palettized RLE blocks. Their palette lives in KING "
            "RAM and is not on the disc, so they decode black; 'greyscale' shows "
            "the raw indices for inspection."
        ),
    )
    extract_cmd.add_argument(
        "--chunk-frames",
        type=int,
        default=DEFAULT_CHUNK_FRAMES,
        help="frames per parallel decode task",
    )
    extract_cmd.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="also write the raw RAINBOW packets and a frame table per stream",
    )
    extract_cmd.add_argument(
        "--dry-run", action="store_true", help="write manifests and the report, but no video"
    )
    extract_cmd.add_argument("--ffmpeg", help="path to an ffmpeg binary")
    extract_cmd.add_argument("-q", "--quiet", action="store_true")
    extract_cmd.set_defaults(func=cmd_extract)

    return parser


def _options_from(args: argparse.Namespace) -> ExtractOptions:
    return ExtractOptions(
        cue=args.cue,
        out_dir=getattr(args, "out", Path(".")),
        track=args.track,
        fps=getattr(args, "fps", None),
        fps_overrides=_parse_fps_overrides(getattr(args, "fps_for", None)),
        codec=getattr(args, "codec", DEFAULT_CODEC),
        pixel_format="bgra" if getattr(args, "alpha", False) else "bgr24",
        jobs=args.jobs,
        chunk_frames=getattr(args, "chunk_frames", DEFAULT_CHUNK_FRAMES),
        min_frames=args.min_frames,
        min_strips=args.min_strips,
        max_frame_gap=args.max_frame_gap,
        mux_audio=getattr(args, "mux_audio", False),
        no_audio=getattr(args, "no_audio", False),
        palettized=getattr(args, "palettized", "black"),
        keep_intermediate=getattr(args, "keep_intermediate", False),
        dry_run=getattr(args, "dry_run", False),
        ffmpeg=getattr(args, "ffmpeg", None),
        quiet=getattr(args, "quiet", False),
    )


def cmd_scan(args: argparse.Namespace) -> int:
    options = _options_from(args)
    streams = discover(options)
    if not streams:
        print("No RAINBOW streams found.")
        return 0

    print(
        f"{'idx':>3}  {'tr':>2}  {'kind':<10} {'lsn':>7} {'frames':>7} "
        f"{'size':>10}  {'geometry':<9} audio"
    )
    for stream in streams:
        print(
            f"{stream.index:>3}  {stream.track:>2}  {stream.kind:<10} "
            f"{stream.first_frame_lsn:>7} {stream.frame_count:>7} "
            f"{stream.byte_size / 1048576:>8.1f}MB  "
            f"{stream.width}x{stream.height:<5} "
            f"{'yes' if stream.has_audio else '-'}"
        )
    total = sum(stream.frame_count for stream in streams)
    print(f"\n{len(streams)} stream(s), {total} frames")
    for stream in streams:
        for note in stream.notes:
            print(f"  note [{stream.index:03d}]: {note}")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    extract(_options_from(args))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PcfxError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
