"""CUE sheet parsing and track geometry.
MODE1/2352 track stores 2048 user bytes at offset 16 inside each
raw sector, while MODE1/2048 tracks are already cooked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .. import PcfxError

RAW_MODE1_2352_USER_OFFSET = 16
RAW_MODE1_2352_USER_SIZE = 2048
RAW_MODE1_2352_SIZE = 2352

PCFX_MAGIC = b"PC-FX:Hu_CD-ROM"


@dataclass
class CueTrack:
    number: int
    mode: str
    file_path: Path
    file_type: str
    indexes: dict[int, int] = field(default_factory=dict)

    @property
    def is_data(self) -> bool:
        return self.mode.upper().startswith("MODE")

    @property
    def sector_size(self) -> int:
        mode = self.mode.upper()
        if mode in ("AUDIO", "MODE1/2352"):
            return 2352
        if mode == "MODE1/2048":
            return 2048
        raise PcfxError(f"Unsupported track mode: {self.mode}")

    @property
    def data_start_sector(self) -> int:
        return self.indexes.get(1, 0)

    @property
    def user_sector_size(self) -> int:
        mode = self.mode.upper()
        if mode in ("MODE1/2352", "MODE1/2048"):
            return RAW_MODE1_2352_USER_SIZE
        raise PcfxError(f"Track {self.number:02d} is not a data track")

    @property
    def user_offset_in_raw_sector(self) -> int:
        mode = self.mode.upper()
        if mode == "MODE1/2352":
            return RAW_MODE1_2352_USER_OFFSET
        if mode == "MODE1/2048":
            return 0
        raise PcfxError(f"Track {self.number:02d} is not a data track")

    @property
    def file_size(self) -> int:
        return self.file_path.stat().st_size

    @property
    def logical_sector_count(self) -> int:
        available_bytes = self.file_size - (self.data_start_sector * self.sector_size)
        if available_bytes < 0:
            raise PcfxError(f"Track {self.number:02d} INDEX 01 is beyond end of file")
        return available_bytes // self.sector_size

    def raw_offset_for_lsn(self, lsn: int) -> int:
        if lsn < 0:
            raise PcfxError("LSN must be non-negative")
        return (
            (self.data_start_sector + lsn) * self.sector_size
        ) + self.user_offset_in_raw_sector


@dataclass
class CueSheet:
    path: Path
    catalog: str | None
    tracks: list[CueTrack]

    def data_tracks(self) -> list[CueTrack]:
        return [track for track in self.tracks if track.is_data]

    def track(self, number: int) -> CueTrack:
        for track in self.tracks:
            if track.number == number:
                return track
        raise PcfxError(f"Track {number:02d} was not found in the CUE")

    def primary_data_track(self) -> CueTrack:
        tracks = self.data_tracks()
        if not tracks:
            raise PcfxError("No MODE data track found in CUE")
        return tracks[0]


def parse_msf(value: str) -> int:
    match = re.fullmatch(r"(\d+):(\d+):(\d+)", value.strip())
    if not match:
        raise PcfxError(f"Invalid MSF timestamp: {value!r}")
    minutes, seconds, frames = (int(part) for part in match.groups())
    if seconds >= 60 or frames >= 75:
        raise PcfxError(f"Invalid MSF timestamp: {value!r}")
    return (minutes * 60 * 75) + (seconds * 75) + frames


def format_msf(sectors: int) -> str:
    if sectors < 0:
        return "-" + format_msf(-sectors)
    minutes = sectors // (60 * 75)
    seconds = (sectors // 75) % 60
    frames = sectors % 75
    return f"{minutes:02d}:{seconds:02d}:{frames:02d}"


_FILE_RE = re.compile(r'^\s*FILE\s+"(.+)"\s+(\S+)\s*$', re.IGNORECASE)
_TRACK_RE = re.compile(r"^\s*TRACK\s+(\d+)\s+(\S+)\s*$", re.IGNORECASE)
_INDEX_RE = re.compile(r"^\s*INDEX\s+(\d+)\s+(\d+:\d+:\d+)\s*$", re.IGNORECASE)
_CATALOG_RE = re.compile(r"^\s*CATALOG\s+(\S+)\s*$", re.IGNORECASE)


def parse_cue(path: Path) -> CueSheet:
    path = Path(path).resolve()
    if not path.exists():
        raise PcfxError(f"CUE not found: {path}")

    catalog: str | None = None
    current_file: tuple[Path, str] | None = None
    current_track: CueTrack | None = None
    tracks: list[CueTrack] = []

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue

        match = _CATALOG_RE.match(line)
        if match:
            catalog = match.group(1)
            continue

        match = _FILE_RE.match(line)
        if match:
            file_name, file_type = match.groups()
            current_file = ((path.parent / file_name).resolve(), file_type)
            continue

        match = _TRACK_RE.match(line)
        if match:
            if current_file is None:
                raise PcfxError(f"TRACK before FILE at {path}:{line_number}")
            number_text, mode = match.groups()
            current_track = CueTrack(
                number=int(number_text),
                mode=mode,
                file_path=current_file[0],
                file_type=current_file[1],
            )
            tracks.append(current_track)
            continue

        match = _INDEX_RE.match(line)
        if match:
            if current_track is None:
                raise PcfxError(f"INDEX before TRACK at {path}:{line_number}")
            index_number, msf = match.groups()
            current_track.indexes[int(index_number)] = parse_msf(msf)
            continue

    if not tracks:
        raise PcfxError(f"No tracks parsed from CUE: {path}")

    missing = sorted({str(track.file_path) for track in tracks if not track.file_path.exists()})
    if missing:
        raise PcfxError("Missing BIN file(s):\n" + "\n".join(missing))

    return CueSheet(path=path, catalog=catalog, tracks=tracks)


def data_index_ranges(track: CueTrack) -> list[dict[str, object]]:
    """Split a data track into its CUE INDEX ranges, in LSN terms."""
    sorted_indexes = sorted(track.indexes.items())
    physical_track_sectors = track.file_size // track.sector_size
    rows: list[dict[str, object]] = []

    for position, (index_number, physical_sector) in enumerate(sorted_indexes):
        next_physical_sector = (
            sorted_indexes[position + 1][1]
            if position + 1 < len(sorted_indexes)
            else physical_track_sectors
        )
        sector_count = max(0, next_physical_sector - physical_sector)
        start_lsn = physical_sector - track.data_start_sector
        rows.append(
            {
                "track": track.number,
                "index": index_number,
                "physical_sector": physical_sector,
                "msf": format_msf(physical_sector),
                "start_lsn": start_lsn,
                "sector_count": sector_count,
                "end_lsn_exclusive": start_lsn + sector_count,
            }
        )
    return rows


def containing_index(ranges: list[dict[str, object]], lsn: int) -> int | str:
    for row in ranges:
        start = int(row["start_lsn"])
        end = int(row["end_lsn_exclusive"])
        if start <= lsn < end:
            return int(row["index"])
    return ""
