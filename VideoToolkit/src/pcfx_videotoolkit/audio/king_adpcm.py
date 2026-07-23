"""KING ADPCM decoding.

Ported from `SoundBox_ADPCMUpdate` in pcfxemu's `mednafen/pcfx/soundbox.c`.
It is an IMA-style adaptive delta codec, but not standard IMA: the step is
multiplied by `(nibble & 7) + 1` rather than looked up from a magnitude table,
and the predictor is a signed 15-bit accumulator.

ARS movie containers interleave one ADPCM packet per video frame; the sample
rate is stated in the container's text header ("32k Monoral", "16k Monoral").
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

STEP_SIZES = (
    16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50,
    55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143, 157,
    173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449,
    494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552,
)

STEP_INDEX_DELTAS = (
    -1, -1, -1, -1, 2, 4, 6, 8,
    -1, -1, -1, -1, 2, 4, 6, 8,
)

MAX_STEP_INDEX = 48
PREDICTOR_MAX = 0x3FFF
PREDICTOR_MIN = -0x4000


def decode_adpcm(data: bytes, step_index: int = 0, predictor: int = 0) -> np.ndarray:
    """Decode packed 4-bit KING ADPCM to signed 16-bit PCM.

    Nibbles come out of each 16-bit KRAM halfword low-first, which on the
    little-endian V810 means low nibble then high nibble of each byte.
    """
    steps = STEP_SIZES
    deltas = STEP_INDEX_DELTAS
    out = np.empty(len(data) * 2, dtype=np.int16)
    index = 0

    for byte in data:
        for nibble in (byte & 0xF, byte >> 4):
            delta = steps[step_index] * ((nibble & 0x7) + 1)
            if nibble & 0x8:
                predictor -= delta
            else:
                predictor += delta

            if predictor > PREDICTOR_MAX:
                predictor = PREDICTOR_MAX
            elif predictor < PREDICTOR_MIN:
                predictor = PREDICTOR_MIN

            step_index += deltas[nibble]
            if step_index < 0:
                step_index = 0
            elif step_index > MAX_STEP_INDEX:
                step_index = MAX_STEP_INDEX

            # The predictor is a 15-bit value; scale it to fill int16.
            out[index] = predictor * 2
            index += 1

    return out


def write_wav(
    path: Path,
    samples: np.ndarray,
    sample_rate: int,
    channels: int = 1,
) -> None:
    """Write signed 16-bit PCM as a plain RIFF/WAVE file."""
    payload = samples.astype("<i2", copy=False).tobytes()
    block_align = channels * 2
    header = b"".join(
        (
            b"RIFF",
            struct.pack("<I", 36 + len(payload)),
            b"WAVEfmt ",
            struct.pack(
                "<IHHIIHH",
                16,
                1,
                channels,
                sample_rate,
                sample_rate * block_align,
                block_align,
                16,
            ),
            b"data",
            struct.pack("<I", len(payload)),
        )
    )
    path.write_bytes(header + payload)
