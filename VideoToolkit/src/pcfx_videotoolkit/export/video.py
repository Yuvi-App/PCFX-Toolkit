"""Writing decoded frames out as AVI via an ffmpeg pipe.
The default codec is FFV1: mathematically lossless, so looks exactly what the PC-FX would have put on screen, at a fraction of the size vs raw frames. 
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .. import PcfxError

#: codec name -> (ffmpeg encoder, extra args, lossless?)
CODECS: dict[str, tuple[str, list[str], bool]] = {
    "ffv1": (
        "ffv1",
        ["-level", "3", "-g", "1", "-coder", "1", "-context", "1", "-slicecrc", "1"],
        True,
    ),
    "huffyuv": ("huffyuv", [], True),
    "utvideo": ("utvideo", [], True),
    "rawvideo": ("rawvideo", [], True),
    "mjpeg": ("mjpeg", ["-q:v", "2"], False),
}

DEFAULT_CODEC = "ffv1"


def find_ffmpeg(explicit: Path | str | None = None) -> str:
    if explicit:
        found = shutil.which(str(explicit)) or (
            str(explicit) if Path(explicit).exists() else None
        )
        if found:
            return found
        raise PcfxError(f"ffmpeg not found at {explicit}")

    try:
        import imageio_ffmpeg

        candidate = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass

    found = shutil.which("ffmpeg")
    if found:
        return found
    raise PcfxError(
        "ffmpeg was not found. Install it and put it on PATH, or "
        "`pip install imageio-ffmpeg`, or pass --ffmpeg."
    )


class AviWriter:
    """Feeds raw frames to ffmpeg over stdin and writes one AVI."""

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: str,
        codec: str = DEFAULT_CODEC,
        pixel_format: str = "bgr24",
        ffmpeg: str | None = None,
        audio_path: Path | None = None,
    ) -> None:
        if codec not in CODECS:
            raise PcfxError(f"Unknown codec {codec!r}; choose from {sorted(CODECS)}")
        encoder, extra, _lossless = CODECS[codec]

        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg or find_ffmpeg(),
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", pixel_format,
            "-s", f"{width}x{height}",
            "-r", fps,
            "-i", "pipe:0",
        ]
        if audio_path is not None:
            command += ["-i", str(audio_path)]
        command += ["-map", "0:v:0"]
        if audio_path is not None:
            command += ["-map", "1:a:0", "-c:a", "pcm_s16le", "-shortest"]
        else:
            command += ["-an"]
        command += ["-c:v", encoder, *extra, "-f", "avi", str(path)]

        self.path = path
        self.command = command
        self.frame_bytes = width * height * (4 if pixel_format == "bgra" else 3)
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frames: bytes) -> None:
        """Write one or more whole frames of packed pixels."""
        if not frames or len(frames) % self.frame_bytes:
            raise PcfxError(
                f"got {len(frames)} bytes, expected a multiple of {self.frame_bytes}"
            )
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(frames)
        except BrokenPipeError:
            stderr = self._process.stderr.read() if self._process.stderr else b""
            raise PcfxError(
                f"ffmpeg closed the pipe writing {self.path.name}: "
                f"{stderr.decode('utf-8', errors='replace').strip()}"
            ) from None

    def close(self) -> None:
        assert self._process.stdin is not None
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass
        stderr = self._process.stderr.read() if self._process.stderr else b""
        code = self._process.wait()
        if code != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise PcfxError(f"ffmpeg failed writing {self.path.name}: {message}")

    def abort(self) -> None:
        try:
            if self._process.stdin is not None:
                self._process.stdin.close()
        except Exception:
            pass
        self._process.kill()
        self._process.wait()

    def __enter__(self) -> "AviWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()
