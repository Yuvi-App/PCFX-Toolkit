"""PC-FX audio: KING ADPCM decoding and WAV output."""

from __future__ import annotations

from .king_adpcm import decode_adpcm, write_wav

__all__ = ["decode_adpcm", "write_wav"]
